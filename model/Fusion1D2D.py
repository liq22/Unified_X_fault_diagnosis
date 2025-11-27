"""
1D-2D Fusion Model for Unified Fault Diagnosis Framework

This module implements 1D-2D fusion models that are compatible with the unified
fault diagnosis infrastructure, supporting both early and aligned fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from .Signal_processing import SignalProcessingModule
from .Feature_extract import FeatureExtractor
from .explainable_base import ExplainableModelMixin


class OneDBranch(nn.Module):
    """1D branch for time series feature extraction"""

    def __init__(self,
                 input_dim: int = 4096,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(OneDBranch, self).__init__()

        self.layers = nn.ModuleList()

        # Build convolutional layers
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                )
            else:
                self.layers.append(
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
                )

            # Add batch normalization and activation
            self.layers.append(nn.BatchNorm1d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D branch

        Args:
            x: Input tensor of shape (batch_size, seq_len) or (batch_size, 1, seq_len)

        Returns:
            Features of shape (batch_size, out_channels)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Apply layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch_size, out_channels, 1)
        x = x.squeeze(-1)  # (batch_size, out_channels)

        return x


class TwoDBranch(nn.Module):
    """2D branch for spectrogram feature extraction"""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (1, 128, 128),
                 base_channels: int = 32,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(TwoDBranch, self).__init__()

        self.input_shape = input_shape

        # Build convolutional layers
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]

        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)

            self.layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))

            in_channels = out_channels

            # Add max pooling after first two layers
            if i < 2:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D branch

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Features of shape (batch_size, out_channels)
        """
        # Apply layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch_size, out_channels, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, out_channels)

        return x


def create_spectrogram_from_1d(signal: torch.Tensor,
                             target_size: Tuple[int, int] = (128, 128),
                             n_fft: int = 256,
                             hop_length: Optional[int] = None,
                             win_length: Optional[int] = None) -> torch.Tensor:
    """
    Convert 1D time series signal to 2D spectrogram using STFT

    Args:
        signal: Input tensor of shape (batch_size, seq_len)
        target_size: Target spectrogram size (height, width)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length for STFT

    Returns:
        Spectrogram tensor of shape (batch_size, 1, height, width)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    batch_size, seq_len = signal.shape
    device = signal.device

    spectrograms = []

    for i in range(batch_size):
        # Compute STFT
        stft = torch.stft(
            signal[i],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(device),
            return_complex=True
        )

        # Get magnitude
        magnitude = torch.abs(stft)  # (freq_bins, time_frames)

        # Convert to log scale
        log_magnitude = torch.log1p(magnitude)

        # Resize to target size
        spectrogram = F.interpolate(
            log_magnitude.unsqueeze(0).unsqueeze(0),  # (1, 1, freq, time)
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        spectrograms.append(spectrogram.squeeze(0))  # (1, height, width)

    # Stack along batch dimension
    result = torch.stack(spectrograms, dim=0)  # (batch_size, 1, height, width)

    return result


class Fusion1D2D(nn.Module, ExplainableModelMixin):
    """
    1D-2D Fusion Model for explainable fault diagnosis

    This model combines 1D time series analysis with 2D spectrogram analysis
    using early fusion strategy. It's designed to be compatible with the unified
    fault diagnosis framework.
    """

    def __init__(self,
                 input_dim: int = 4096,
                 spectrogram_size: Tuple[int, int] = (128, 128),
                 num_classes: int = 10,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 fusion_type: str = 'early',
                 signal_processing: Optional[List[str]] = None,
                 feature_extraction: Optional[List[str]] = None):
        """
        Initialize 1D-2D Fusion model

        Args:
            input_dim: Input sequence dimension
            spectrogram_size: Target spectrogram size (height, width)
            num_classes: Number of fault classes
            hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
            fusion_type: Fusion strategy ('early', 'aligned')
            signal_processing: Signal processing operations
            feature_extraction: Feature extraction methods
        """
        super(Fusion1D2D, self).__init__()

        self.input_dim = input_dim
        self.spectrogram_size = spectrogram_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

        # Initialize signal processing and feature extraction
        if signal_processing is None:
            signal_processing = ['I']  # Identity for direct processing
        if feature_extraction is None:
            feature_extraction = []  # Use deep learning features only

        self.signal_processing = signal_processing
        self.feature_extraction = feature_extraction

        # 1D branch
        self.one_d_branch = OneDBranch(
            input_dim=input_dim,
            in_channels=1,
            out_channels=64,
            num_layers=3,
            dropout=dropout
        )

        # 2D branch
        self.two_d_branch = TwoDBranch(
            input_shape=(1, *spectrogram_size),
            base_channels=32,
            num_layers=3,
            dropout=dropout
        )

        # Fusion layers
        if fusion_type == 'early':
            # Early fusion: concatenate 1D and 2D features
            self.fusion_layers = nn.Sequential(
                nn.Linear(128, hidden_dim),  # 64 (1D) + 64 (2D) = 128
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        logits, _, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with intermediate features

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, features_1d, features_2d)
        """
        # Extract 1D features
        features_1d = self.one_d_branch(x)

        # Create 2D spectrogram from 1D signal
        spectrogram = create_spectrogram_from_1d(x, target_size=self.spectrogram_size)

        # Extract 2D features
        features_2d = self.two_d_branch(spectrogram)

        # Fusion
        if self.fusion_type == 'early':
            fused_features = torch.cat([features_1d, features_2d], dim=1)
            logits = self.fusion_layers(fused_features)

        return logits, features_1d, features_2d

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fused embeddings without classification

        Args:
            x: Input tensor

        Returns:
            Fused embeddings
        """
        features_1d = self.one_d_branch(x)
        spectrogram = create_spectrogram_from_1d(x, target_size=self.spectrogram_size)
        features_2d = self.two_d_branch(spectrogram)

        fused_features = torch.cat([features_1d, features_2d], dim=1)
        return fused_features

    def get_signal_path(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get signal processing path for explainability

        Args:
            x: Input tensor

        Returns:
            Dictionary containing signal processing information
        """
        with torch.no_grad():
            # Get intermediate features
            _, feat_1d, feat_2d = self.forward_with_features(x)

            # Create spectrogram for visualization
            spectrogram = create_spectrogram_from_1d(x, target_size=self.spectrogram_size)

            path_info = {
                'input_shape': x.shape,
                'spectrogram_shape': spectrogram.shape,
                'features_1d_shape': feat_1d.shape,
                'features_2d_shape': feat_2d.shape,
                'signal_processing': self.signal_processing,
                'feature_extraction': self.feature_extraction,
                'fusion_type': self.fusion_type,
                'branch_contributions': {
                    '1d_branch': feat_1d.detach().cpu().numpy(),
                    '2d_branch': feat_2d.detach().cpu().numpy(),
                    'spectrogram': spectrogram.detach().cpu().numpy()
                }
            }

            return path_info

    def get_explainability_info(self) -> Dict[str, Any]:
        """
        Get model explainability information

        Returns:
            Dictionary containing explainability metadata
        """
        return {
            'model_type': 'Fusion1D2D',
            'supports_signal_path': True,
            'supports_intrinsic_explanation': True,
            'modality': 'multimodal',
            'input_types': ['1d_time_series', '2d_spectrogram'],
            'explanation_methods': ['signal_path', 'integrated_gradients', 'grad_cam'],
            'fusion_strategy': self.fusion_type,
            'branches': {
                '1d_branch': {
                    'type': 'CNN_1D',
                    'layers': len(self.one_d_branch.layers) // 4,  # Conv + BN + ReLU + Dropout
                    'output_dim': 64
                },
                '2d_branch': {
                    'type': 'CNN_2D',
                    'layers': len(self.two_d_branch.layers) // 5 if hasattr(self.two_d_branch, 'layers') else 3,
                    'output_dim': 64
                }
            }
        }


class AlignedFusion1D2D(Fusion1D2D):
    """
    Aligned 1D-2D Fusion Model with semantic alignment between branches
    """

    def __init__(self,
                 input_dim: int = 4096,
                 spectrogram_size: Tuple[int, int] = (128, 128),
                 num_classes: int = 10,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 alignment_weight: float = 0.1,
                 **kwargs):
        """
        Initialize Aligned 1D-2D Fusion model

        Args:
            alignment_weight: Weight for alignment loss
        """
        super(AlignedFusion1D2D, self).__init__(
            input_dim=input_dim,
            spectrogram_size=spectrogram_size,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_type='aligned',
            **kwargs
        )

        self.alignment_weight = alignment_weight

        # Projection layers for alignment (project both to same space)
        self.projection_1d = nn.Linear(64, hidden_dim)
        self.projection_2d = nn.Linear(64, hidden_dim)

        # Fusion layers for aligned features
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Aligned features from both branches
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with alignment

        Returns:
            Tuple of (logits, features_1d, features_2d, alignment_loss)
        """
        # Extract features from both branches
        features_1d = self.one_d_branch(x)
        spectrogram = create_spectrogram_from_1d(x, target_size=self.spectrogram_size)
        features_2d = self.two_d_branch(spectrogram)

        # Project features to alignment space
        aligned_1d = self.projection_1d(features_1d)
        aligned_2d = self.projection_2d(features_2d)

        # Compute alignment loss (contrastive loss)
        alignment_loss = self.compute_alignment_loss(aligned_1d, aligned_2d)

        # Fuse aligned features
        fused_features = torch.cat([aligned_1d, aligned_2d], dim=1)
        logits = self.fusion_layers(fused_features)

        return logits, features_1d, features_2d, alignment_loss

    def compute_alignment_loss(self, feat_1d: torch.Tensor, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between 1D and 2D features

        Args:
            feat_1d: 1D branch features
            feat_2d: 2D branch features

        Returns:
            Alignment loss
        """
        # Normalize features
        feat_1d_norm = F.normalize(feat_1d, p=2, dim=1)
        feat_2d_norm = F.normalize(feat_2d, p=2, dim=1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(feat_1d_norm, feat_2d_norm, dim=1)

        # Alignment loss: maximize similarity
        alignment_loss = 1.0 - similarity.mean()

        return alignment_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        logits, _, _, _ = self.forward_with_features(x)
        return logits