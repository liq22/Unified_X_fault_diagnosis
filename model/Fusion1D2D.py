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

# 简化导入，使用现有的基础类
from .Signal_processing import (
    SignalProcessingBase, SignalProcessingModuleDict,
    FFTSignalProcessing, HilbertTransform, WaveFilters, Identity
)
from .Feature_extract import FeatureExtractor


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

        # Build layers
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(
                nn.Conv1d(in_ch, out_channels, kernel_size=3, padding=1)
            )
            self.layers.append(nn.BatchNorm1d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))

            # Add max pooling after first two layers
            if i < 2:
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D branch

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)

        Returns:
            Features of shape (batch_size, out_channels, reduced_seq_len)
        """
        # Apply layers
        for layer in self.layers:
            x = layer(x)

        return x


class TwoDBranch(nn.Module):
    """2D branch for spectrogram feature extraction"""

    def __init__(self,
                 input_dim: int = 4096,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(TwoDBranch, self).__init__()

        self.layers = nn.ModuleList()

        # Build layers
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.layers.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
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


class Fusion1D2D(nn.Module):
    """
    1D-2D Fusion Model for explainable fault diagnosis

    This model combines 1D time series analysis with 2D spectrogram analysis
    using early fusion strategy. It's designed to be compatible with the unified
    fault diagnosis framework.
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        """
        Initialize Fusion1D2D model

        Args:
            signal_processing_modules: Signal processing modules configuration
            feature_extractor_modules: Feature extractor modules configuration
            args: Arguments containing model hyperparameters
        """
        super(Fusion1D2D, self).__init__()

        # Extract parameters from args
        self.input_dim = getattr(args, 'in_dim', 4096)
        self.in_channels = getattr(args, 'in_channels', 2)
        self.out_channels = getattr(args, 'out_channels', 3)
        self.scale = getattr(args, 'scale', 4)
        self.num_classes = getattr(args, 'num_classes', 5)
        self.skip_connection = getattr(args, 'skip_connection', True)

        # Signal processing layers (using TSPN's signal processing)
        from .TSPN import SignalProcessingLayer

        self.signal_processing_layers = nn.ModuleList()
        for i in range(4):
            layer_config = getattr(args, f'layer{i+1}', ['I', 'WF', 'I'])
            # 使用与 TSPN 一致的接口：SignalProcessingModuleDict 需要传入字典，
            # 各算子模块接收完整的 args 以保持配置与设备信息一致
            module_dict = SignalProcessingModuleDict({})

            # Map config strings to actual modules
            for module_name in layer_config:
                if module_name == 'I':
                    module_dict[module_name] = Identity(args)
                elif module_name == 'WF':
                    module_dict[module_name] = WaveFilters(args)
                elif module_name == 'HT':
                    module_dict[module_name] = HilbertTransform(args)
                elif module_name == 'FFT':
                    module_dict[module_name] = FFTSignalProcessing(args)
                else:
                    module_dict[module_name] = Identity(args)

            in_ch = self.in_channels if i == 0 else self.out_channels
            out_ch = self.out_channels

            self.signal_processing_layers.append(
                SignalProcessingLayer(module_dict, in_ch, out_ch, self.skip_connection)
            )

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # 1D branch
        self.one_d_branch = OneDBranch(
            input_dim=self.input_dim,
            in_channels=self.out_channels,
            out_channels=64,
            num_layers=3
        )

        # 2D branch
        self.two_d_branch = TwoDBranch(
            input_dim=self.input_dim,
            in_channels=1,
            out_channels=64,
            num_layers=3
        )

        # Fusion layer
        fusion_dim = 64 + 64  # Concatenated features from both branches
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 13, 128),  # 64 from fusion + 13 from feature extractor
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Fusion1D2D model

        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        # Reshape input
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        # Apply signal processing layers
        for layer in self.signal_processing_layers:
            x = layer(x)

        # Store processed signal for both branches
        processed_signal = x

        # 1D branch
        one_d_features = self.one_d_branch(processed_signal)  # (batch_size, 64, reduced_seq_len)
        one_d_features = F.adaptive_avg_pool1d(one_d_features, 1).squeeze(-1)  # (batch_size, 64)

        # 2D branch - convert to spectrogram
        # Use the first channel for spectrogram conversion
        signal_1d = processed_signal[:, 0, :]  # (batch_size, seq_len)
        spectrogram = create_spectrogram_from_1d(signal_1d)  # (batch_size, 1, height, width)
        two_d_features = self.two_d_branch(spectrogram)  # (batch_size, 64)

        # Fusion
        fused_features = torch.cat([one_d_features, two_d_features], dim=1)  # (batch_size, 128)
        fused_features = self.fusion_layer(fused_features)  # (batch_size, 64)

        # Extract statistical features
        statistical_features = self.feature_extractor(processed_signal)  # (batch_size, 13)

        # Combine all features
        combined_features = torch.cat([fused_features, statistical_features], dim=1)  # (batch_size, 77)

        # Classification
        logits = self.classifier(combined_features)  # (batch_size, num_classes)

        return logits
