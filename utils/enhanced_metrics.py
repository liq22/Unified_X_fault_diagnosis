"""
Enhanced metrics collection for Fuzzy-XFD validation
Provides confusion matrices, ROC curves, per-class metrics, and rule analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
import pandas as pd

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class EnhancedMetricsCollector:
    """
    Collects and computes enhanced metrics for model evaluation
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.rule_activations = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None,
               rule_activations: Optional[torch.Tensor] = None):
        """
        Update with new batch of predictions

        Args:
            predictions: Predicted class indices (batch_size,)
            targets: Ground truth labels (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes)
            rule_activations: Fuzzy rule activation values (batch_size, num_rules)
        """
        # Convert to CPU and numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None:
            probabilities = probabilities.detach().cpu().numpy()
        if rule_activations is not None:
            rule_activations = rule_activations.detach().cpu().numpy()

        self.all_predictions.append(predictions)
        self.all_targets.append(targets)

        if probabilities is not None:
            self.all_probabilities.append(probabilities)

        if rule_activations is not None:
            self.rule_activations.append(rule_activations)

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix"""
        all_pred = np.concatenate(self.all_predictions)
        all_true = np.concatenate(self.all_targets)
        return confusion_matrix(all_true, all_pred)

    def compute_classification_report(self) -> Dict:
        """Compute detailed classification report"""
        all_pred = np.concatenate(self.all_predictions)
        all_true = np.concatenate(self.all_targets)
        return classification_report(all_true, all_pred,
                                   target_names=self.class_names,
                                   output_dict=True)

    def compute_per_class_metrics(self) -> pd.DataFrame:
        """Compute per-class precision, recall, F1-score"""
        report = self.compute_classification_report()

        metrics = []
        for class_name in self.class_names:
            if class_name in report:
                metrics.append({
                    'class': class_name,
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                })

        return pd.DataFrame(metrics)

    def compute_roc_curves(self) -> Tuple[Dict, Dict]:
        """
        Compute ROC curves for multi-class classification

        Returns:
            roc_curves: Dictionary with fpr, tpr, roc_auc for each class
            macro_metrics: Macro-averaged ROC metrics
        """
        if not self.all_probabilities:
            raise ValueError("Probabilities not stored. Enable probability tracking.")

        all_true = np.concatenate(self.all_targets)
        all_probs = np.concatenate(self.all_probabilities)

        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(all_true, classes=range(self.num_classes))

        roc_curves = {}
        macro_fpr = np.linspace(0, 1, 100)
        macro_tpr = np.zeros_like(macro_fpr)

        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)

            roc_curves[self.class_names[i]] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            }

            # Interpolate for macro average
            macro_tpr += np.interp(macro_fpr, fpr, tpr)

        # Compute macro average
        macro_tpr /= self.num_classes
        macro_auc = auc(macro_fpr, macro_tpr)

        macro_metrics = {
            'fpr': macro_fpr,
            'tpr': macro_tpr,
            'roc_auc': macro_auc
        }

        return roc_curves, macro_metrics

    def analyze_rule_activations(self) -> Dict[str, Any]:
        """
        Analyze fuzzy rule activation patterns

        Returns:
            Dictionary with rule usage statistics
        """
        if not self.rule_activations:
            return {"error": "Rule activations not stored"}

        all_rules = np.concatenate(self.rule_activations, axis=0)

        # Compute statistics
        rule_stats = {
            'mean_activation': np.mean(all_rules, axis=0),
            'std_activation': np.std(all_rules, axis=0),
            'max_activation': np.max(all_rules, axis=0),
            'min_activation': np.min(all_rules, axis=0),
            'usage_frequency': np.sum(all_rules > 0.1, axis=0) / len(all_rules)
        }

        # Find most and least used rules
        most_used = np.argmax(rule_stats['usage_frequency'])
        least_used = np.argmin(rule_stats['usage_frequency'])

        analysis = {
            'total_rules': all_rules.shape[1],
            'total_samples': len(all_rules),
            'rule_statistics': rule_stats,
            'most_used_rule': {
                'index': int(most_used),
                'usage_frequency': float(rule_stats['usage_frequency'][most_used]),
                'mean_activation': float(rule_stats['mean_activation'][most_used])
            },
            'least_used_rule': {
                'index': int(least_used),
                'usage_frequency': float(rule_stats['usage_frequency'][least_used]),
                'mean_activation': float(rule_stats['mean_activation'][least_used])
            }
        }

        return analysis

    def compute_safety_metrics(self) -> Dict[str, float]:
        """
        Compute safety-critical metrics

        Returns:
            Dictionary with safety metrics including false negative rate
        """
        cm = self.compute_confusion_matrix()

        # Calculate per-class false negative rate (FNR)
        # FNR = FN / (FN + TP)
        fn_rates = {}
        total_fn = 0
        total_tp = 0

        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            fn_rates[f'fn_rate_{self.class_names[i]}'] = fn_rate
            total_fn += fn
            total_tp += tp

        # Overall false negative rate
        overall_fn_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0

        # Identify most dangerous errors (FN for critical classes)
        # Assuming class 0 (e.g., 'Healthy') is most critical to not miss
        critical_fn_rate = fn_rates.get('fn_rate_Class_0', 0.0)

        return {
            'overall_fn_rate': overall_fn_rate,
            'critical_fn_rate': critical_fn_rate,
            **fn_rates
        }

    def save_metrics(self, save_dir: str, prefix: str = ''):
        """
        Save all computed metrics to files

        Args:
            save_dir: Directory to save metrics
            prefix: Optional prefix for filenames
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save confusion matrix
        cm = self.compute_confusion_matrix()
        np.save(save_path / f'{prefix}confusion_matrix.npy', cm)

        # Save classification report
        report = self.compute_classification_report()
        with open(save_path / f'{prefix}classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Save per-class metrics
        metrics_df = self.compute_per_class_metrics()
        metrics_df.to_csv(save_path / f'{prefix}per_class_metrics.csv', index=False)

        # Save ROC curves if probabilities available
        if self.all_probabilities:
            try:
                roc_curves, macro_metrics = self.compute_roc_curves()

                # Convert numpy arrays to lists for JSON serialization
                roc_serializable = {}
                for class_name, curve_data in roc_curves.items():
                    roc_serializable[class_name] = {
                        'fpr': curve_data['fpr'].tolist(),
                        'tpr': curve_data['tpr'].tolist(),
                        'roc_auc': float(curve_data['roc_auc'])
                    }

                roc_serializable['macro'] = {
                    'fpr': macro_metrics['fpr'].tolist(),
                    'tpr': macro_metrics['tpr'].tolist(),
                    'roc_auc': float(macro_metrics['roc_auc'])
                }

                with open(save_path / f'{prefix}roc_curves.json', 'w') as f:
                    json.dump(roc_serializable, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not compute ROC curves: {e}")

        # Save rule analysis
        rule_analysis = self.analyze_rule_activations()
        if 'error' not in rule_analysis:
            with open(save_path / f'{prefix}rule_analysis.json', 'w') as f:
                json.dump(rule_analysis, f, indent=2)

        # Save safety metrics
        safety_metrics = self.compute_safety_metrics()
        with open(save_path / f'{prefix}safety_metrics.json', 'w') as f:
            json.dump(safety_metrics, f, indent=2)

        print(f"Metrics saved to {save_path}")


class EnhancedMetricsCallback:
    """
    PyTorch Lightning callback for collecting enhanced metrics
    """

    def __init__(self, metrics_collector: EnhancedMetricsCollector):
        self.metrics_collector = metrics_collector

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of each test batch
        """
        # Extract predictions, targets, and probabilities from outputs
        if isinstance(outputs, dict):
            predictions = outputs.get('predictions', outputs.get('preds'))
            targets = outputs.get('targets', outputs.get('labels'))
            probabilities = outputs.get('probabilities', outputs.get('probs'))
            rule_activations = outputs.get('rule_activations')
        else:
            # Handle tuple/list outputs
            if len(outputs) >= 2:
                predictions, targets = outputs[:2]
                probabilities = outputs[2] if len(outputs) > 2 else None
                rule_activations = outputs[3] if len(outputs) > 3 else None
            else:
                return

        # Update metrics collector
        self.metrics_collector.update(predictions, targets, probabilities, rule_activations)

    def on_test_epoch_end(self, trainer, pl_module):
        """
        Called at the end of test epoch
        """
        # Save metrics to checkpoint directory
        save_dir = Path(trainer.checkpoint_callback.dirpath) if trainer.checkpoint_callback else Path('./results')
        self.metrics_collector.save_metrics(str(save_dir), 'test_')


def create_confusion_matrix_plot(cm: np.ndarray, class_names: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a beautiful confusion matrix plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)

    # Add summary statistics
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_roc_curves_plot(roc_curves: Dict, macro_metrics: Dict,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create ROC curves plot for multi-class classification
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class ROC curve
    for class_name, curve_data in roc_curves.items():
        if class_name == 'macro':
            continue

        ax.plot(curve_data['fpr'], curve_data['tpr'],
               label=f'{class_name} (AUC = {curve_data["roc_auc"]:.3f})',
               linewidth=2)

    # Plot macro average
    ax.plot(macro_metrics['fpr'], macro_metrics['tpr'],
           label=f'Macro Average (AUC = {macro_metrics["roc_auc"]:.3f})',
           color='black', linestyle='--', linewidth=2)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_rule_activation_plot(rule_analysis: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of rule activation patterns
    """
    if 'error' in rule_analysis:
        return None

    stats = rule_analysis['rule_statistics']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Mean activation
    axes[0, 0].bar(range(len(stats['mean_activation'])), stats['mean_activation'])
    axes[0, 0].set_title('Mean Rule Activation')
    axes[0, 0].set_xlabel('Rule Index')
    axes[0, 0].set_ylabel('Mean Activation')

    # Usage frequency
    axes[0, 1].bar(range(len(stats['usage_frequency'])), stats['usage_frequency'])
    axes[0, 1].set_title('Rule Usage Frequency')
    axes[0, 1].set_xlabel('Rule Index')
    axes[0, 1].set_ylabel('Frequency')

    # Activation distribution
    axes[1, 0].hist(stats['mean_activation'], bins=20, alpha=0.7)
    axes[1, 0].set_title('Distribution of Mean Activations')
    axes[1, 0].set_xlabel('Mean Activation')
    axes[1, 0].set_ylabel('Number of Rules')

    # Highlight most/least used rules
    most_used = rule_analysis['most_used_rule']
    least_used = rule_analysis['least_used_rule']

    axes[1, 1].scatter([least_used['index'], most_used['index']],
                      [least_used['usage_frequency'], most_used['usage_frequency']],
                      s=100, c=['red', 'green'], label=['Least Used', 'Most Used'])
    axes[1, 1].scatter(range(len(stats['usage_frequency'])),
                      stats['usage_frequency'], alpha=0.5)
    axes[1, 1].set_title('Rule Usage Analysis')
    axes[1, 1].set_xlabel('Rule Index')
    axes[1, 1].set_ylabel('Usage Frequency')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig