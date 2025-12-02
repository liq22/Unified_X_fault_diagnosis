"""
Fuzzy Logic Network for Unified Fault Diagnosis Framework

This module implements a fuzzy logic-based model that combines traditional fuzzy
reasoning with neural networks for explainable fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np

# 导入现有的基础类与特征提取器
from .Signal_processing import (
    SignalProcessingBase,
    SignalProcessingModuleDict,
    FFTSignalProcessing,
    HilbertTransform,
    WaveFilters,
    Identity,
)
from .Feature_extract import FeatureExtractor


class FuzzyMembershipFunction(nn.Module):
    """
    Fuzzy membership function implementation
    """

    def __init__(self, num_features: int, num_membership_functions: int = 3):
        super(FuzzyMembershipFunction, self).__init__()
        self.num_features = num_features
        self.num_membership_functions = num_membership_functions

        # Learnable parameters for Gaussian membership functions
        # Each feature has num_membership_functions Gaussian functions
        self.centers = nn.Parameter(
            torch.randn(num_features, num_membership_functions)
        )
        self.widths = nn.Parameter(
            torch.ones(num_features, num_membership_functions) * 0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fuzzy membership values

        Args:
            x: Input tensor of shape (batch_size, num_features)

        Returns:
            Membership values of shape (batch_size, num_features, num_membership_functions)
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_features, num_membership_functions)
        widths_expanded = torch.abs(self.widths.unsqueeze(0))  # (1, num_features, num_membership_functions)

        # Gaussian membership function
        membership = torch.exp(
            -((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2)
        )

        return membership


class FuzzyRule(nn.Module):
    """
    Fuzzy rule implementation using first-order predicate logic
    """

    def __init__(self, num_features: int, num_rules: int, num_classes: int):
        super(FuzzyRule, self).__init__()
        self.num_features = num_features
        self.num_rules = num_rules
        self.num_classes = num_classes

        # Rule antecedents (which features and membership functions are used)
        # Using learnable weights for soft rule selection
        self.rule_weights = nn.Parameter(
            torch.ones(num_rules, num_features) / num_features
        )

        # Rule consequents (class outputs)
        self.rule_consequents = nn.Parameter(
            torch.randn(num_rules, num_classes) * 0.1
        )

    def forward(self, membership_values: torch.Tensor) -> torch.Tensor:
        """
        Apply fuzzy rules

        Args:
            membership_values: Shape (batch_size, num_features, num_membership_functions)

        Returns:
            Rule outputs of shape (batch_size, num_rules, num_classes)
        """
        batch_size = membership_values.size(0)

        # Aggregate membership values for each rule
        # Using weighted product as t-norm for fuzzy AND operation
        rule_strengths = []
        for rule_idx in range(self.num_rules):
            # Get membership values for this rule
            rule_membership = membership_values * self.rule_weights[rule_idx].unsqueeze(0).unsqueeze(-1)

            # Apply t-norm (product)
            rule_strength = torch.prod(torch.sum(rule_membership, dim=1), dim=1, keepdim=True)  # (batch_size, 1)
            rule_strengths.append(rule_strength)

        rule_strengths = torch.cat(rule_strengths, dim=1)  # (batch_size, num_rules)

        # Apply rule consequents
        rule_outputs = rule_strengths.unsqueeze(-1) * self.rule_consequents.unsqueeze(0)  # (batch_size, num_rules, num_classes)

        return rule_outputs, rule_strengths


class FuzzyLogicNetwork(nn.Module):
    """
    Fuzzy Logic Network for explainable fault diagnosis

    This model combines traditional fuzzy logic reasoning with neural network
    components for better interpretability while maintaining good performance.
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        """
        Initialize FuzzyLogicNetwork

        Args:
            signal_processing_modules: Signal processing modules configuration
            feature_extractor_modules: Feature extractor modules configuration
            args: Arguments containing model hyperparameters
        """
        super(FuzzyLogicNetwork, self).__init__()

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
            # 与 TSPN/Fusion1D2D 保持一致：显式传入空字典，并为算子提供 args
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

        # Feature extractor for statistical features
        self.feature_extractor = FeatureExtractor()

        # Dimension reduction layer (to reduce computational complexity)
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.out_channels * self.input_dim // 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Fuzzy membership functions
        self.num_fuzzy_features = 64  # After reduction + statistical features
        self.num_membership_functions = 3  # Low, Medium, High
        self.num_fuzzy_rules = 20  # Number of fuzzy rules

        self.fuzzy_membership = FuzzyMembershipFunction(
            num_features=self.num_fuzzy_features,
            num_membership_functions=self.num_membership_functions
        )

        # Fuzzy rules using first-order predicate logic
        self.fuzzy_rules = FuzzyRule(
            num_features=self.num_fuzzy_features,
            num_rules=self.num_fuzzy_rules,
            num_classes=self.num_classes
        )

        # Output layer for defuzzification
        self.defuzzification = nn.Sequential(
            nn.Linear(self.num_classes, self.num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FuzzyLogicNetwork

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

        # Reduce dimension for fuzzy processing
        # Global average pooling to reduce sequence dimension
        x_pooled = F.adaptive_avg_pool1d(x, self.input_dim // 16)  # (batch_size, out_channels, reduced_seq_len)
        x_flat = x_pooled.view(x_pooled.size(0), -1)  # (batch_size, out_channels * reduced_seq_len)

        # Apply feature reduction
        reduced_features = self.feature_reducer(x_flat)  # (batch_size, 64)

        # Extract statistical features
        statistical_features = self.feature_extractor(x)  # (batch_size, 13)

        # Combine features
        # Pad statistical features to match dimension
        padded_stats = F.pad(statistical_features, (0, 64 - 13))
        combined_features = reduced_features + padded_stats  # (batch_size, 64)

        # Apply fuzzy membership functions
        membership_values = self.fuzzy_membership(combined_features)  # (batch_size, 64, 3)

        # Apply fuzzy rules
        rule_outputs, rule_strengths = self.fuzzy_rules(membership_values)  # (batch_size, num_rules, num_classes)

        # Aggregate rule outputs (weighted average)
        final_output = torch.sum(rule_outputs, dim=1)  # (batch_size, num_classes)

        # Apply defuzzification
        logits = self.defuzzification(final_output)  # (batch_size, num_classes)

        return logits

    def get_rule_explanations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get fuzzy rule explanations for interpretability

        Args:
            x: Input tensor

        Returns:
            Dictionary containing membership values, rule strengths, and rule outputs
        """
        with torch.no_grad():
            # Forward pass to get explanations
            if x.dim() == 3:
                x = x.transpose(1, 2)

            for layer in self.signal_processing_layers:
                x = layer(x)

            x_pooled = F.adaptive_avg_pool1d(x, self.input_dim // 16)
            x_flat = x_pooled.view(x_pooled.size(0), -1)
            reduced_features = self.feature_reducer(x_flat)
            statistical_features = self.feature_extractor(x)
            padded_stats = F.pad(statistical_features, (0, 64 - 13))
            combined_features = reduced_features + padded_stats

            membership_values = self.fuzzy_membership(combined_features)
            rule_outputs, rule_strengths = self.fuzzy_rules(membership_values)

            return {
                'membership_values': membership_values,
                'rule_strengths': rule_strengths,
                'rule_outputs': rule_outputs,
                'features': combined_features
            }
