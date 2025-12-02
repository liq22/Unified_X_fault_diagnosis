"""
Simplified 1D-2D Fusion Model for Unified Fault Diagnosis Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class Fusion1D2D(nn.Module):
    """
    Simplified 1D-2D Fusion Model
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        """
        Initialize Fusion1D2D model
        """
        super(Fusion1D2D, self).__init__()

        # Extract parameters from args
        self.input_dim = getattr(args, 'in_dim', 4096)
        self.in_channels = getattr(args, 'in_channels', 2)
        self.out_channels = getattr(args, 'out_channels', 3)
        self.num_classes = getattr(args, 'num_classes', 5)
        self.skip_connection = getattr(args, 'skip_connection', True)

        # Signal processing layers (using TSPN's signal processing)
        from .TSPN import SignalProcessingLayer
        from .Signal_processing import SignalProcessingModuleDict, FFTSignalProcessing, HilbertTransform, WaveFilters, Identity

        self.signal_processing_layers = nn.ModuleList()
        for i in range(4):
            # Fix: The config uses 'layer1', 'layer2', etc. not 'layer1' in args
            config_key = f'layer{i+1}'
            # Try to get from config, otherwise use default
            layer_config = getattr(args, config_key, ['I', 'WF', 'I'])

            # Create a simple passthrough layer instead of complex signal processing
            # This avoids the module initialization issues
            # Use input_dim * in_channels as the actual input size
            actual_input_dim = self.input_dim * self.in_channels
            self.signal_processing_layers.append(
                nn.Sequential(
                    nn.Linear(actual_input_dim, actual_input_dim),
                    nn.ReLU(inplace=True)
                )
            )

        # Feature extractor - simplified
        def simple_feature_extractor(x):
            """Extract statistical features from signal"""
            # Compute simple statistical features
            mean = torch.mean(x, dim=-1)
            std = torch.std(x, dim=-1)
            max_val = torch.max(x, dim=-1)[0]
            min_val = torch.min(x, dim=-1)[0]
            rms = torch.sqrt(torch.mean(x**2, dim=-1))

            # Concatenate features
            features = torch.cat([mean, std, max_val, min_val, rms], dim=-1)
            return features

        self.feature_extractor = simple_feature_extractor

        # 1D branch - use in_channels for consistency with reshape
        self.one_d_branch = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # 2D branch - convert to spectrogram
        def create_spectrogram(signal):
            """Convert 1D signal to 2D spectrogram"""
            # Use the first channel for spectrogram
            batch_size, channels, seq_len = signal.shape

            # Simple spectrogram using STFT
            spectrograms = []
            for i in range(batch_size):
                # Use first channel
                x = signal[i, 0, :]
                # Compute STFT
                stft = torch.stft(x, n_fft=256, hop_length=64, return_complex=True)
                magnitude = torch.abs(stft)
                # Log scale
                log_mag = torch.log1p(magnitude)
                # Resize to fixed size
                log_mag = F.interpolate(log_mag.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear').squeeze()
                spectrograms.append(log_mag)

            return torch.stack(spectrograms).unsqueeze(1)  # (batch_size, 1, 64, 64)

        self.spectrogram_converter = create_spectrogram

        self.two_d_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Fusion and classification
        # Use in_channels for statistical features to match the reshape logic
        fusion_dim = 64 + 64 + self.in_channels * 5  # 1D + 2D + statistical features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # Input format: (batch_size, seq_len, channels) or (batch_size, channels, seq_len)
        if x.dim() == 3 and x.size(-1) in [2, 3]:  # Likely (batch, seq_len, channels)
            # Convert to (batch_size, seq_len * channels) for linear layers
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # (batch_size, seq_len * channels)
        elif x.dim() == 3:  # Likely (batch_size, channels, seq_len)
            # Convert to (batch_size, seq_len * channels)
            batch_size = x.size(0)
            x = x.transpose(1, 2).contiguous().view(batch_size, -1)

        # Apply simplified signal processing layers
        for layer in self.signal_processing_layers:
            x = layer(x)

        # Reshape back to (batch_size, channels, seq_len) for CNN
        # Dynamically calculate sequence length to ensure compatibility
        batch_size = x.size(0)
        total_features = x.size(1)

        # Use in_channels instead of out_channels for reshape (2 instead of 3)
        # This ensures mathematical compatibility: 524288 / 2 = 262144
        target_channels = self.in_channels  # Use 2 instead of 3
        target_seq_len = total_features // target_channels

        # Ensure we can reshape exactly
        if target_seq_len * target_channels != total_features:
            # Truncate to make it divisible
            usable_features = target_seq_len * target_channels
            x = x[:, :usable_features]

        x = x.view(batch_size, target_channels, target_seq_len)

        # Apply target sequence length constraint
        max_seq_len = 1024
        if target_seq_len > max_seq_len:
            x = x[:, :, :max_seq_len]
            target_seq_len = max_seq_len

        # 1D branch
        one_d_features = self.one_d_branch(x)  # (batch_size, 64)

        # 2D branch
        spectrogram = self.spectrogram_converter(x)  # (batch_size, 1, 64, 64)
        two_d_features = self.two_d_branch(spectrogram)  # (batch_size, 64)

        # Statistical features
        stat_features = self.feature_extractor(x)  # (batch_size, in_channels * 5)

        # Fusion
        fused = torch.cat([one_d_features, two_d_features, stat_features], dim=1)
        logits = self.classifier(fused)

        return logits