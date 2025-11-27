"""
Mixture of Experts (MoE) Model for Fault Diagnosis

This module implements a physics-constrained Mixture of Experts model that combines
multiple specialized experts for interpretable fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from .explainable_base import ExplainableMixin
from .TSPN import SignalProcessingLayer


class FeatureExtractor(nn.Module):
    """Simple statistical feature extractor for routing and expert processing."""

    def __init__(self):
        super().__init__()
        self.feature_names = [
            'mean', 'std', 'var', 'rms', 'peak', 'peak2peak', 'abs_mean',
            'skewness', 'kurtosis', 'impulse_factor', 'clearance_factor',
            'shape_factor', 'crest_factor', 'margin_factor', 'energy'
        ]

    def forward(self, x):
        """
        Extract statistical features from input signal.

        Args:
            x: Input signal [batch_size, signal_length]

        Returns:
            features: Statistical features [batch_size, 15]
        """
        # Ensure x is 2D
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        device = x.device

        features = []

        for signal in x:
            # Basic statistics
            mean_val = torch.mean(signal)
            std_val = torch.std(signal)
            var_val = torch.var(signal)
            rms_val = torch.sqrt(torch.mean(signal ** 2))

            # Peak values
            peak_val = torch.max(torch.abs(signal))
            min_val = torch.min(signal)
            peak2peak_val = peak_val - min_val
            abs_mean_val = torch.mean(torch.abs(signal))

            # Higher order moments
            centered = signal - mean_val
            std_val = torch.std(centered) + 1e-8  # Avoid division by zero

            skewness = torch.mean(centered ** 3) / (std_val ** 3)
            kurtosis = torch.mean(centered ** 4) / (std_val ** 4)

            # Engineering features
            impulse_factor = peak_val / abs_mean_val
            clearance_factor = peak_val / (torch.mean(torch.sqrt(torch.abs(signal))) ** 2 + 1e-8)
            shape_factor = rms_val / abs_mean_val
            crest_factor = peak_val / rms_val
            margin_factor = peak_val / (torch.mean(signal ** 2) ** 0.5 + 1e-8)
            energy = torch.sum(signal ** 2)

            signal_features = torch.tensor([
                mean_val.item(), std_val.item(), var_val.item(), rms_val.item(),
                peak_val.item(), peak2peak_val.item(), abs_mean_val.item(),
                skewness.item(), kurtosis.item(), impulse_factor.item(),
                clearance_factor.item(), shape_factor.item(), crest_factor.item(),
                margin_factor.item(), energy.item()
            ], device=device)

            features.append(signal_features)

        return torch.stack(features, dim=0)

    def get_feature_names(self):
        """Return the list of feature names."""
        return self.feature_names


class BaseExpert(nn.Module, ABC):
    """Base class for all expert modules."""

    def __init__(self, expert_id: str, feature_dim: int = 64):
        super().__init__()
        self.expert_id = expert_id
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the expert.

        Args:
            x: Input signal [batch_size, signal_length]

        Returns:
            Tuple of (expert_output, expert_metadata)
        """
        pass

    @abstractmethod
    def get_expert_info(self) -> Dict[str, Any]:
        """Get expert description and capabilities."""
        pass


class LowFrequencyExpert(BaseExpert):
    """Expert specializing in low-frequency fault detection."""

    def __init__(self, feature_dim: int = 64, cutoff_freq: float = 500.0):
        super().__init__("low_freq", feature_dim)
        self.cutoff_freq = cutoff_freq

        # Signal processing layers for low frequency
        self.signal_layers = nn.ModuleList([
            SignalProcessingLayer(['I', 'WF'], 1, 16, skip_connection=True),
            SignalProcessingLayer(['FFT', 'I'], 16, 32, skip_connection=True),
        ])

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Expert-specific network
        self.expert_net = nn.Sequential(
            nn.Linear(15, 32),  # 15 statistical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim)
        )

        # Low-pass filter parameters
        self.register_buffer('filter_freq', torch.tensor(cutoff_freq))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = x.shape[0]

        # Process through signal layers
        x_in = x.unsqueeze(-1)  # [batch_size, length, 1]

        # Low-pass filtering using FFT
        x_fft = torch.fft.fft(x_in, dim=1)
        freq_bins = x_fft.shape[1]

        # Create low-pass filter mask
        cutoff_idx = int(self.cutoff_freq * freq_bins / 12000.0)  # Assuming 12kHz sampling
        filter_mask = torch.zeros(freq_bins, 1, device=x.device)
        filter_mask[:cutoff_idx] = 1.0

        # Apply filter
        x_fft_filtered = x_fft * filter_mask
        x_filtered = torch.fft.ifft(x_fft_filtered, dim=1).real

        # Extract features
        features = self.feature_extractor(x_filtered.squeeze(-1))  # [batch_size, 15]

        # Process through expert network
        expert_features = self.expert_net(features)

        # Metadata for explainability
        metadata = {
            'expert_type': 'low_frequency',
            'cutoff_freq': self.cutoff_freq,
            'filtered_signal': x_filtered,
            'filter_mask': filter_mask,
            'feature_stats': {
                'mean': torch.mean(features),
                'std': torch.std(features),
                'rms': torch.sqrt(torch.mean(x_filtered ** 2))
            }
        }

        return expert_features, metadata

    def get_expert_info(self) -> Dict[str, Any]:
        return {
            'expert_id': self.expert_id,
            'expert_name': 'LowFrequencyExpert',
            'target_faults': ['转子不平衡', '基础振动', '低频机械松动'],
            'physical_mechanism': 'Low-frequency energy concentration',
            'frequency_range': f'0-{self.cutoff_freq} Hz',
            'strengths': ['Sensitive to low-frequency faults', 'Strong noise immunity'],
            'parameters': {
                'feature_dim': self.feature_dim,
                'cutoff_freq': self.cutoff_freq
            }
        }


class HarmonicExpert(BaseExpert):
    """Expert specializing in harmonic fault detection."""

    def __init__(self, feature_dim: int = 64):
        super().__init__("harmonic", feature_dim)

        # Signal processing layers for harmonic analysis
        self.signal_layers = nn.ModuleList([
            SignalProcessingLayer(['HT', 'WF'], 1, 16, skip_connection=True),
            SignalProcessingLayer(['FFT', 'I'], 16, 32, skip_connection=True),
        ])

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Harmonic analysis network
        self.harmonic_net = nn.Sequential(
            nn.Linear(15, 32),  # 15 statistical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim)
        )

        # Harmonic detection layers
        self.harmonic_detector = nn.Sequential(
            nn.Linear(128, 64),  # Spectrum features
            nn.ReLU(),
            nn.Linear(64, feature_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = x.shape[0]

        # Hilbert transform for envelope
        x_complex = torch.fft.fft(x)
        x_analytic = torch.fft.ifft(torch.cat([
            x_complex[:x_complex.shape[0]//2],
            torch.zeros_like(x_complex[:x_complex.shape[0]//2])
        ], dim=0))

        # Extract features
        features = self.feature_extractor(x)

        # Harmonic analysis using FFT
        x_fft = torch.fft.fft(x)
        spectrum = torch.abs(x_fft[:, :x.shape[1]//2])

        # Normalize spectrum to fixed length
        if spectrum.shape[-1] < 128:
            padding_size = 128 - spectrum.shape[-1]
            spectrum = torch.cat([
                spectrum,
                torch.zeros(batch_size, padding_size, device=x.device)
            ], dim=-1)
        else:
            spectrum = spectrum[:, :128]

        harmonic_features = self.harmonic_detector(spectrum)

        # Combine features
        expert_features = self.harmonic_net(features)

        # Metadata
        metadata = {
            'expert_type': 'harmonic',
            'spectrum_magnitude': spectrum,
            'envelope_signal': torch.abs(x_analytic),
            'feature_stats': {
                'mean': torch.mean(features),
                'std': torch.std(features),
                'spectral_centroid': torch.sum(spectrum * torch.arange(spectrum.shape[-1], device=x.device), dim=-1) / (torch.sum(spectrum, dim=-1) + 1e-8)
            }
        }

        return expert_features, metadata

    def get_expert_info(self) -> Dict[str, Any]:
        return {
            'expert_id': self.expert_id,
            'expert_name': 'HarmonicExpert',
            'target_faults': ['齿轮故障', '滚动体故障', '轴承磨损'],
            'physical_mechanism': 'Harmonic frequency patterns',
            'frequency_range': 'Mid-frequency harmonics',
            'strengths': ['Detects periodic patterns', 'Good for gear/bearing faults'],
            'parameters': {
                'feature_dim': self.feature_dim
            }
        }


class EnvelopeExpert(BaseExpert):
    """Expert specializing in envelope analysis for impact detection."""

    def __init__(self, feature_dim: int = 64):
        super().__init__("envelope", feature_dim)

        # Signal processing layers for envelope analysis
        self.signal_layers = nn.ModuleList([
            SignalProcessingLayer(['HT', 'WF'], 1, 16, skip_connection=True),
            SignalProcessingLayer(['I', 'HT'], 16, 32, skip_connection=True),
        ])

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Envelope analysis network
        self.envelope_net = nn.Sequential(
            nn.Linear(15, 32),  # 15 statistical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Extract features
        features = self.feature_extractor(x)

        # Envelope analysis using Hilbert transform
        x_analytic = torch.fft.fft(x)
        x_analytic = torch.fft.ifft(torch.cat([
            2 * x_analytic[:, :x_analytic.shape[1]//2],
            torch.zeros_like(x_analytic[:, :x_analytic.shape[1]//2])
        ], dim=1))
        envelope = torch.abs(x_analytic)

        # Extract envelope features
        envelope_features = self.feature_extractor(envelope)

        # Process through expert network
        expert_features = self.envelope_net(features)

        # Metadata
        metadata = {
            'expert_type': 'envelope',
            'envelope_signal': envelope,
            'envelope_features': envelope_features,
            'feature_stats': {
                'mean': torch.mean(features),
                'std': torch.std(features),
                'kurtosis': torch.mean((features - torch.mean(features))**4) / (torch.std(features)**4 + 1e-8),
                'peak_factor': torch.max(torch.abs(features)) / (torch.sqrt(torch.mean(features**2)) + 1e-8)
            }
        }

        return expert_features, metadata

    def get_expert_info(self) -> Dict[str, Any]:
        return {
            'expert_id': self.expert_id,
            'expert_name': 'EnvelopeExpert',
            'target_faults': ['外圈故障', '内圈故障', '冲击故障'],
            'physical_mechanism': 'Impact detection through envelope analysis',
            'frequency_range': 'High-frequency impacts',
            'strengths': ['Sensitive to impacts', 'Good for bearing faults'],
            'parameters': {
                'feature_dim': self.feature_dim
            }
        }


class StatisticalRouter(nn.Module):
    """Router that uses statistical features for expert selection."""

    def __init__(self, num_experts: int, feature_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.temperature = temperature

        # Feature extractor for routing
        self.feature_extractor = FeatureExtractor()

        # Routing network
        self.router_net = nn.Sequential(
            nn.Linear(15, 32),  # 15 statistical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts)
        )

        # Regularization parameters
        self.load_balance_weight = 0.1
        self.sparsity_weight = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        batch_size = x.shape[0]

        # Extract statistical features
        stats_features = self.feature_extractor(x)  # [batch_size, 15]

        # Compute routing logits
        routing_logits = self.router_net(stats_features)  # [batch_size, num_experts]

        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature

        # Compute routing weights
        routing_weights = F.softmax(routing_logits, dim=-1)  # [batch_size, num_experts]

        # Routing info for explainability
        routing_info = {
            'logits': routing_logits,
            'entropy': -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1),
            'dominant_expert': torch.argmax(routing_weights, dim=-1)
        }

        return routing_weights, stats_features, routing_info


class MoEModel(nn.Module, ExplainableMixin):
    """
    Mixture of Experts Model for Interpretable Fault Diagnosis

    This model combines multiple specialized experts with an intelligent routing system
    to provide both high accuracy and explainability in fault diagnosis.
    """

    def __init__(self,
                 num_classes: int = 10,
                 feature_dim: int = 64,
                 num_experts: int = 3,
                 routing_temperature: float = 1.0,
                 use_load_balance: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.use_load_balance = use_load_balance

        # Initialize experts
        self.experts = nn.ModuleList([
            LowFrequencyExpert(feature_dim),
            HarmonicExpert(feature_dim),
            EnvelopeExpert(feature_dim)
        ])

        # Initialize router
        self.router = StatisticalRouter(num_experts, feature_dim, routing_temperature)

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Explainability storage
        self._last_forward_metadata = {}

    def forward(self, x: torch.Tensor, return_explanations: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the MoE model.

        Args:
            x: Input signal [batch_size, signal_length]
            return_explanations: Whether to return explanation data

        Returns:
            Tuple of (logits, metadata)
        """
        batch_size = x.shape[0]

        # 1. Routing decision
        routing_weights, routing_features, routing_info = self.router(x)

        # 2. Expert parallel processing
        expert_outputs = []
        expert_metadata = []

        for expert in self.experts:
            expert_output, expert_meta = expert(x)
            expert_outputs.append(expert_output)
            expert_metadata.append(expert_meta)

        # 3. Combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, feature_dim]
        routing_weights = routing_weights.unsqueeze(-1)      # [batch_size, num_experts, 1]

        # Weighted combination
        fused_features = torch.sum(expert_outputs * routing_weights, dim=1)  # [batch_size, feature_dim]
        fused_features = self.fusion_net(fused_features)

        # 4. Classification
        logits = self.classifier(fused_features)

        # 5. Collect metadata
        metadata = {
            'routing_weights': routing_weights.squeeze(-1),  # [batch_size, num_experts]
            'expert_outputs': expert_outputs,
            'fused_features': fused_features,
            'routing_info': routing_info,
            'expert_metadata': expert_metadata,
            'logits': logits
        }

        # 6. Generate explanations
        if return_explanations:
            explanations = self._generate_explanations(x, metadata)
            metadata['explanations'] = explanations

        # Store for explainability methods
        self._last_forward_metadata = metadata

        return logits, metadata

    def get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """Get the signal transformation path through the MoE model."""
        if not self._last_forward_metadata:
            # Run forward pass if no metadata available
            self.forward(input_data, return_explanations=True)

        metadata = self._last_forward_metadata
        routing_weights = metadata['routing_weights']
        expert_metadata = metadata['expert_metadata']

        path = []

        # Add routing information
        for i, weight in enumerate(routing_weights):
            path.append({
                'stage': 'routing',
                'expert_weights': weight.detach().cpu().numpy(),
                'dominant_expert': int(torch.argmax(weight)),
                'expert_confidence': float(torch.max(weight))
            })

        # Add expert processing information
        for i, expert_meta in enumerate(expert_metadata):
            path.append({
                'stage': f'expert_{i}',
                'expert_type': expert_meta['expert_type'],
                'feature_stats': {k: float(v) if torch.is_tensor(v) else v
                                for k, v in expert_meta['feature_stats'].items()}
            })

        return path

    def get_operator_graph(self) -> Dict[str, Any]:
        """Get the operator graph structure of the MoE model."""
        return {
            'model_type': 'MixtureOfExperts',
            'num_experts': self.num_experts,
            'experts': [expert.get_expert_info() for expert in self.experts],
            'router_type': 'StatisticalRouter',
            'connections': [
                {'from': 'router', 'to': f'expert_{i}', 'type': 'routing_weights'}
                for i in range(self.num_experts)
            ] + [
                {'from': f'expert_{i}', 'to': 'fusion', 'type': 'weighted_combination'}
                for i in range(self.num_experts)
            ] + [
                {'from': 'fusion', 'to': 'classifier', 'type': 'direct'}
            ]
        }

    def get_attention_maps(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention weights (routing weights in MoE context)."""
        if not self._last_forward_metadata:
            self.forward(input_data, return_explanations=True)

        return {
            'routing_weights': self._last_forward_metadata['routing_weights'],
            'expert_importance': torch.mean(self._last_forward_metadata['routing_weights'], dim=0)
        }

    def _generate_explanations(self, x: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanations for the model's decision."""
        routing_weights = metadata['routing_weights']
        expert_metadata = metadata['expert_metadata']

        # Expert activation analysis
        expert_activations = torch.mean(routing_weights, dim=0)
        most_active_expert = int(torch.argmax(expert_activations))

        # Path signatures
        path_signatures = []
        for i in range(x.shape[0]):
            signature = {
                'sample_id': i,
                'expert_weights': routing_weights[i].detach().cpu().numpy().tolist(),
                'dominant_expert': int(torch.argmax(routing_weights[i])),
                'confidence': float(torch.max(routing_weights[i])),
                'routing_entropy': float(-torch.sum(routing_weights[i] * torch.log(routing_weights[i] + 1e-8)))
            }
            path_signatures.append(signature)

        # Expert descriptions
        expert_descriptions = [expert.get_expert_info() for expert in self.experts]

        explanations = {
            'path_signatures': path_signatures,
            'expert_activations': {
                'mean_weights': expert_activations.detach().cpu().numpy().tolist(),
                'most_active_expert': most_active_expert,
                'expert_descriptions': expert_descriptions
            },
            'routing_analysis': {
                'entropy': float(torch.mean(metadata['routing_info']['entropy'])),
                'dominant_expert_distribution': torch.mode(metadata['routing_info']['dominant_expert']).values.tolist()
            }
        }

        return explanations

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': 'MoEModel',
            'version': '1.0.0',
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'num_experts': self.num_experts,
            'experts': [expert.get_expert_info() for expert in self.experts],
            'explainable_features': [
                'Expert routing visualization',
                'Path signature analysis',
                'Expert contribution analysis',
                'Physical mechanism interpretation'
            ]
        }