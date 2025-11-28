"""
Simplified Fuzzy Logic Network for Unified Fault Diagnosis Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np


class FuzzyLogicNetwork(nn.Module):
    """
    Simplified Fuzzy Logic Network for explainable fault diagnosis
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        """
        Initialize FuzzyLogicNetwork
        """
        super(FuzzyLogicNetwork, self).__init__()

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
            layer_config = getattr(args, f'layer{i+1}', ['I', 'WF', 'I'])
            module_dict = SignalProcessingModuleDict({})

            # Map config strings to actual modules
            for idx, module_name in enumerate(layer_config):
                # 使用唯一键避免覆盖
                unique_key = f"{module_name}_{idx}"
                if module_name == 'I':
                    module_dict[unique_key] = Identity(args)
                elif module_name == 'WF':
                    module_dict[unique_key] = WaveFilters(args)
                elif module_name == 'HT':
                    module_dict[unique_key] = HilbertTransform(args)
                elif module_name == 'FFT':
                    module_dict[unique_key] = FFTSignalProcessing(args)
                else:
                    module_dict[unique_key] = Identity(args)

            in_ch = self.in_channels if i == 0 else self.out_channels
            out_ch = self.out_channels

            self.signal_processing_layers.append(
                SignalProcessingLayer(module_dict, in_ch, out_ch, self.skip_connection)
            )

        # Simplified fuzzy logic implementation
        # Feature reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.out_channels * 64, 32),  # Reduced dimensions to match fuzzy features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Fuzzy membership functions
        self.num_fuzzy_features = 32  # Match feature_reducer output
        self.num_membership_functions = 3  # Low, Medium, High
        self.num_fuzzy_rules = 10

        # Membership function parameters (centers and widths)
        self.centers = nn.Parameter(
            torch.randn(self.num_fuzzy_features, self.num_membership_functions) * 0.5
        )
        self.widths = nn.Parameter(
            torch.ones(self.num_fuzzy_features, self.num_membership_functions) * 0.3
        )

        # Rule weights
        self.rule_weights = nn.Parameter(
            torch.ones(self.num_fuzzy_rules, self.num_fuzzy_features) / self.num_fuzzy_features
        )
        self.rule_outputs = nn.Parameter(
            torch.randn(self.num_fuzzy_rules, self.num_classes) * 0.1
        )

        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(self.num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes)
        )

    def compute_membership(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fuzzy membership values using Gaussian functions
        """
        # Expand dimensions
        x_expanded = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_features, num_membership_functions)
        widths_expanded = torch.abs(self.widths.unsqueeze(0))  # (1, num_features, num_membership_functions)

        # Gaussian membership function
        membership = torch.exp(
            -((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2)
        )

        return membership  # (batch_size, num_features, num_membership_functions)

    def apply_rules(self, membership_values: torch.Tensor) -> torch.Tensor:
        """
        Apply fuzzy rules
        """
        # Aggregate membership values across membership functions (dim=2)
        # membership_values: (batch_size, num_features, num_memberships) -> (batch_size, num_features)
        aggregated_membership = torch.mean(membership_values, dim=2)

        # Weighted sum for rule activation
        rule_activation = torch.sum(
            aggregated_membership.unsqueeze(1) * self.rule_weights.unsqueeze(0), dim=2
        )  # (batch_size, num_rules)

        # Normalize rule activations
        rule_activation = F.softmax(rule_activation, dim=-1)

        # Apply rule outputs
        rule_outputs = rule_activation.unsqueeze(-1) * self.rule_outputs.unsqueeze(0)
        final_output = torch.sum(rule_outputs, dim=1)  # (batch_size, num_classes)

        return final_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        # TSPN's SignalProcessingLayer expects (batch_size, seq_len, channels) format
        # Keep input as is: (batch_size, seq_len, channels)

        # Apply signal processing layers
        for layer in self.signal_processing_layers:
            x = layer(x)

        # Convert to (batch_size, channels, seq_len) for pooling
        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        # Reduce dimension for fuzzy processing
        # Global average pooling to reduce sequence dimension
        x_pooled = F.adaptive_avg_pool1d(x, 64)  # (batch_size, out_channels, 64)
        x_flat = x_pooled.reshape(x_pooled.size(0), -1)  # (batch_size, out_channels * 64)

        # Apply feature reduction
        reduced_features = self.feature_reducer(x_flat)  # (batch_size, 64)

        # Apply fuzzy membership functions
        membership_values = self.compute_membership(reduced_features)

        # Apply fuzzy rules
        fuzzy_output = self.apply_rules(membership_values)

        # Final classification
        logits = self.classifier(fuzzy_output)

        return logits