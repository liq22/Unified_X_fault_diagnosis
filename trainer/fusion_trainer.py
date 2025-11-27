"""
Trainer for 1D-2D Fusion Models

This module provides a specialized trainer for 1D-2D fusion models that handles
the unique requirements of multimodal learning and explainability.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..model.Fusion1D2D import Fusion1D2D, AlignedFusion1D2D
from ..explainability.core.unified_explainer import UnifiedExplainer
from .utils import l1_reg


class Fusion1D2DTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for 1D-2D Fusion models
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.save_hyperparameters()

        # Initialize model based on fusion type
        fusion_type = getattr(args, 'fusion_type', 'early')
        if fusion_type == 'aligned':
            self.model = AlignedFusion1D2D(
                input_dim=getattr(args, 'input_dim', 4096),
                spectrogram_size=tuple(getattr(args, 'spectrogram_size', [128, 128])),
                num_classes=getattr(args, 'num_classes', 10),
                hidden_dim=getattr(args, 'hidden_dim', 128),
                dropout=getattr(args, 'dropout', 0.2),
                alignment_weight=getattr(args, 'alignment_weight', 0.1)
            )
        else:
            self.model = Fusion1D2D(
                input_dim=getattr(args, 'input_dim', 4096),
                spectrogram_size=tuple(getattr(args, 'spectrogram_size', [128, 128])),
                num_classes=getattr(args, 'num_classes', 10),
                hidden_dim=getattr(args, 'hidden_dim', 128),
                dropout=getattr(args, 'dropout', 0.2),
                fusion_type=fusion_type
            )

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()

        # Metrics
        self.acc_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )
        self.acc_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )
        self.acc_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

        self.f1_macro_train = torchmetrics.F1Score(
            task="multiclass", num_classes=args.num_classes, average='macro'
        )
        self.f1_macro_val = torchmetrics.F1Score(
            task="multiclass", num_classes=args.num_classes, average='macro'
        )
        self.f1_macro_test = torchmetrics.F1Score(
            task="multiclass", num_classes=args.num_classes, average='macro'
        )

        # Explainability
        self.explainer = None
        self.explanations = []

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch

        # Forward pass
        if hasattr(self.model, 'forward_with_features'):
            outputs = self.model.forward_with_features(x)
            if len(outputs) == 4:  # Aligned fusion
                logits, feat_1d, feat_2d, alignment_loss = outputs
                loss = self.classification_loss(logits, y.long())
                loss += getattr(self.args, 'alignment_weight', 0.1) * alignment_loss
            else:  # Early fusion
                logits, feat_1d, feat_2d = outputs
                loss = self.classification_loss(logits, y.long())
        else:
            logits = self.model(x)
            loss = self.classification_loss(logits, y.long())

        # Add L1 regularization if specified
        if hasattr(self.args, 'l1_norm') and self.args.l1_norm > 0:
            l1_loss = l1_reg(self.model)
            loss += self.args.l1_norm * l1_loss

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.acc_train.update(preds, y.long())
        self.f1_macro_train.update(preds, y.long())

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.acc_train, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_macro', self.f1_macro_train, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch

        # Forward pass
        if hasattr(self.model, 'forward_with_features'):
            outputs = self.model.forward_with_features(x)
            if len(outputs) == 4:  # Aligned fusion
                logits, feat_1d, feat_2d, alignment_loss = outputs
                loss = self.classification_loss(logits, y.long())
                loss += getattr(self.args, 'alignment_weight', 0.1) * alignment_loss
            else:  # Early fusion
                logits, feat_1d, feat_2d = outputs
                loss = self.classification_loss(logits, y.long())
        else:
            logits = self.model(x)
            loss = self.classification_loss(logits, y.long())

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.acc_val.update(preds, y.long())
        self.f1_macro_val.update(preds, y.long())

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.acc_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_macro', self.f1_macro_val, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch

        # Forward pass
        if hasattr(self.model, 'forward_with_features'):
            outputs = self.model.forward_with_features(x)
            if len(outputs) == 4:  # Aligned fusion
                logits, feat_1d, feat_2d, alignment_loss = outputs
            else:  # Early fusion
                logits, feat_1d, feat_2d = outputs
        else:
            logits = self.model(x)

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.acc_test.update(preds, y.long())
        self.f1_macro_test.update(preds, y.long())

        # Log metrics
        self.log('test_acc', self.acc_test, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.f1_macro_test, on_step=False, on_epoch=True)

        return logits

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Reset metrics
        self.acc_train.reset()
        self.f1_macro_train.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Reset metrics
        self.acc_val.reset()
        self.f1_macro_val.reset()

    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        # Reset metrics
        self.acc_test.reset()
        self.f1_macro_test.reset()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Optimizer
        optimizer = Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 0.0001)
        )

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=getattr(self.args, 'lr_patience', 5),
            min_lr=1e-6,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': type(self.model).__name__,
            'fusion_type': getattr(self.args, 'fusion_type', 'early'),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': getattr(self.args, 'input_dim', 4096),
            'spectrogram_size': getattr(self.args, 'spectrogram_size', [128, 128]),
            'num_classes': getattr(self.args, 'num_classes', 10),
            'hidden_dim': getattr(self.args, 'hidden_dim', 128)
        }

    def initialize_explainer(self, method: str = 'auto'):
        """Initialize the explainer for explainability analysis"""
        if self.explainer is None:
            self.explainer = UnifiedExplainer(
                model=self.model,
                config={'method': method},
                method=method
            )

    def explain_batch(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      method: str = 'auto',
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of samples

        Args:
            batch: Input batch (x, y)
            method: Explanation method
            **kwargs: Additional arguments

        Returns:
            List of explanations
        """
        if self.explainer is None:
            self.initialize_explainer(method)

        x, y = batch
        explanations = []

        # Generate explanations for each sample in the batch
        batch_size = x.shape[0]
        for i in range(batch_size):
            sample_input = x[i:i+1]  # Keep batch dimension
            target_class = y[i].item()

            try:
                explanation = self.explainer.explain(
                    sample_input,
                    target_class=target_class,
                    **kwargs
                )
                explanations.append({
                    'explanation': explanation,
                    'true_label': target_class,
                    'sample_idx': i
                })
            except Exception as e:
                print(f"Failed to explain sample {i}: {e}")
                explanations.append(None)

        return explanations

    def analyze_branch_contributions(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze the contribution of 1D and 2D branches

        Args:
            batch: Input batch (x, y)

        Returns:
            Analysis results
        """
        x, y = batch

        with torch.no_grad():
            if hasattr(self.model, 'forward_with_features'):
                outputs = self.model.forward_with_features(x)
                if len(outputs) == 4:  # Aligned fusion
                    logits, feat_1d, feat_2d, alignment_loss = outputs
                else:  # Early fusion
                    logits, feat_1d, feat_2d = outputs

                # Compute feature statistics
                feat_1d_norm = torch.norm(feat_1d, dim=1).mean().item()
                feat_2d_norm = torch.norm(feat_2d, dim=1).mean().item()

                # Compute similarity between branches
                feat_1d_normed = F.normalize(feat_1d, p=2, dim=1)
                feat_2d_normed = F.normalize(feat_2d, p=2, dim=1)
                similarity = F.cosine_similarity(feat_1d_normed, feat_2d_normed).mean().item()

                analysis = {
                    '1d_feature_norm': feat_1d_norm,
                    '2d_feature_norm': feat_2d_norm,
                    'branch_similarity': similarity,
                    '1d_features': feat_1d.detach().cpu().numpy(),
                    '2d_features': feat_2d.detach().cpu().numpy(),
                    'alignment_loss': alignment_loss.item() if len(outputs) == 4 else None
                }

                return analysis
            else:
                return {'error': 'Model does not support branch analysis'}


# Factory function for creating trainers
def create_fusion_trainer(args) -> Fusion1D2DTrainer:
    """
    Factory function to create a fusion trainer

    Args:
        args: Configuration arguments

    Returns:
        Fusion1D2DTrainer instance
    """
    return Fusion1D2DTrainer(args)