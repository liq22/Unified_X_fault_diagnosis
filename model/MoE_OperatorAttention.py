"""
MoE + Operator Attention Fusion Model

This module implements a novel fusion approach that combines the
Mixture of Experts paradigm with Operator Attention mechanisms
to achieve both high accuracy and comprehensive explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod

from .explainable_base import ExplainableMixin
from .MoE import MoEModel, BaseExpert, StatisticalRouter, FeatureExtractor
from .TSPN_OperatorAttention import TSPNWithOperatorAttention as TSPN_OperatorAttention
from .TSPN import SignalProcessingLayer


class AttentionAwareExpert(BaseExpert):
    """
    Expert that incorporates attention mechanisms into its decision making.

    Combines the specialization of MoE experts with the interpretability
    of attention mechanisms.
    """

    def __init__(self, base_expert: BaseExpert, feature_dim: int = 64):
        super().__init__(f"attention_{base_expert.expert_id}", feature_dim)

        # Store the base expert
        self.base_expert = base_expert

        # Attention mechanism for the expert
        self.attention_dim = feature_dim
        self.query_projection = nn.Linear(feature_dim, self.attention_dim)
        self.key_projection = nn.Linear(feature_dim, self.attention_dim)
        self.value_projection = nn.Linear(feature_dim, self.attention_dim)

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim + self.attention_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

        # Attention for interpretability
        self.signal_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,  # Use attention_dim instead of feature_dim
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with attention mechanisms.

        Args:
            x: Input signal [batch_size, signal_length]

        Returns:
            Tuple of (expert_output, expert_metadata)
        """
        # Get base expert output
        base_output, base_metadata = self.base_expert(x)

        batch_size = base_output.shape[0]

        # Compute attention weights
        queries = self.query_projection(base_output)
        keys = self.key_projection(base_output)
        values = self.value_projection(base_output)

        # Self-attention within the expert
        queries = queries.unsqueeze(1)  # [batch_size, 1, attention_dim]
        keys = keys.unsqueeze(1)
        values = values.unsqueeze(1)

        attended_output, attention_weights = self.signal_attention(queries, keys, values)
        attended_output = attended_output.squeeze(1)  # [batch_size, attention_dim]

        # Fuse base output and attention output
        combined_features = torch.cat([base_output, attended_output], dim=-1)
        final_output = self.fusion_net(combined_features)

        # Enhanced metadata with attention information
        metadata = {
            'expert_type': f'attention_{base_metadata["expert_type"]}',
            'base_expert_output': base_output,
            'attention_weights': attention_weights.squeeze(1),  # [batch_size, 1, 1]
            'attended_output': attended_output,
            'attention_query': queries.squeeze(1),
            'attention_key': keys.squeeze(1),
            'attention_value': values.squeeze(1),
            'base_metadata': base_metadata,
            'attention_entropy': self._compute_attention_entropy(attention_weights),
            'feature_importance': self._compute_feature_importance(attention_weights, base_output)
        }

        return final_output, metadata

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights for interpretability."""
        # attention_weights: [batch_size, 1, 1]
        attention_probs = F.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        return torch.mean(entropy)

    def _compute_feature_importance(self, attention_weights: torch.Tensor,
                                  features: torch.Tensor) -> torch.Tensor:
        """Compute feature importance based on attention weights."""
        # Simple approach: use attention weights to weight feature importance
        attention_importance = torch.abs(attention_weights.squeeze(-1))  # [batch_size, 1]
        feature_importance = torch.mean(features * attention_importance, dim=0)  # [feature_dim]
        return feature_importance

    def get_expert_info(self) -> Dict[str, Any]:
        """Get enhanced expert information."""
        base_info = self.base_expert.get_expert_info()
        base_info['expert_id'] = self.expert_id
        base_info['expert_name'] = f"Attention{base_info['expert_name']}"
        base_info['enhancements'] = [
            'Self-attention mechanism',
            'Feature importance analysis',
            'Attention entropy tracking',
            'Enhanced interpretability'
        ]
        base_info['attention_heads'] = 4
        base_info['attention_dim'] = self.attention_dim
        return base_info


class OperatorAwareRouter(StatisticalRouter):
    """
    Enhanced router that uses operator attention insights for routing decisions.
    """

    def __init__(self, num_experts: int, feature_dim: int = 64,
                 temperature: float = 1.0, use_operator_attention: bool = True):
        super().__init__(num_experts, feature_dim, temperature)

        self.use_operator_attention = use_operator_attention

        if use_operator_attention:
            # Operator attention integration
            self.operator_attention_dim = 16  # Fixed dimension
            self.operator_gate = nn.Linear(15, self.operator_attention_dim)  # 15 statistical features
            self.expert_operator_weights = nn.Parameter(
                torch.randn(num_experts, self.operator_attention_dim)
            )

        # Enhanced routing with attention insights
        self.attention_router = nn.Sequential(
            nn.Linear(15 + (self.operator_attention_dim if use_operator_attention else 0), 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Enhanced forward pass with operator attention."""
        # Extract statistical features
        stats_features = self.feature_extractor(x)  # [batch_size, 15]

        if self.use_operator_attention:
            # Compute operator-aware features
            operator_features = torch.tanh(self.operator_gate(stats_features))

            # Compute expert-operator compatibility
            expert_operator_scores = torch.matmul(
                operator_features, self.expert_operator_weights.T
            )  # [batch_size, num_experts]

            # Combine features
            combined_features = torch.cat([stats_features, operator_features], dim=-1)
        else:
            combined_features = stats_features
            expert_operator_scores = None

        # Enhanced routing with attention
        routing_logits = self.attention_router(combined_features)

        # Incorporate operator compatibility if available
        if self.use_operator_attention and expert_operator_scores is not None:
            routing_logits = routing_logits + 0.1 * expert_operator_scores

        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature

        # Compute routing weights
        routing_weights = F.softmax(routing_logits, dim=-1)

        # Enhanced routing info
        routing_info = {
            'logits': routing_logits,
            'entropy': -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1),
            'dominant_expert': torch.argmax(routing_weights, dim=-1)
        }

        if self.use_operator_attention:
            routing_info.update({
                'operator_features': operator_features,
                'expert_operator_scores': expert_operator_scores
            })

        return routing_weights, stats_features, routing_info


class MoEOperatorAttentionFusion(nn.Module, ExplainableMixin):
    """
    Fusion model that combines MoE and Operator Attention paradigms.

    This model achieves:
    1. Expert specialization through MoE
    2. Attention-based interpretability
    3. Operator-aware routing
    4. Enhanced decision transparency
    """

    def __init__(self,
                 num_classes: int = 10,
                 feature_dim: int = 64,
                 num_experts: int = 3,
                 routing_temperature: float = 1.0,
                 use_operator_attention: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.use_operator_attention = use_operator_attention

        # Create base experts
        from .MoE import LowFrequencyExpert, HarmonicExpert, EnvelopeExpert
        base_experts = [
            LowFrequencyExpert(feature_dim),
            HarmonicExpert(feature_dim),
            EnvelopeExpert(feature_dim)
        ]

        # Wrap experts with attention mechanisms
        self.experts = nn.ModuleList([
            AttentionAwareExpert(expert, feature_dim) for expert in base_experts
        ])

        # Operator-aware router
        self.router = OperatorAwareRouter(
            num_experts, feature_dim, routing_temperature, use_operator_attention
        )

        # Cross-expert attention mechanism
        self.cross_expert_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Operator attention module for signal processing (simplified for demo)
        if use_operator_attention:
            # For demonstration, create a simple operator attention module
            self.signal_operator_attention = nn.Sequential(
                nn.Linear(4096, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            self.signal_operator_attention = None

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * (2 if use_operator_attention else 1), feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
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
        Forward pass through the fusion model.

        Args:
            x: Input signal [batch_size, signal_length]
            return_explanations: Whether to return explanation data

        Returns:
            Tuple of (logits, metadata)
        """
        batch_size = x.shape[0]

        # 1. Enhanced routing decision
        routing_weights, routing_features, routing_info = self.router(x)

        # 2. Expert parallel processing with attention
        expert_outputs = []
        expert_metadata = []

        for expert in self.experts:
            expert_output, expert_meta = expert(x)
            expert_outputs.append(expert_output)
            expert_metadata.append(expert_meta)

        # Stack expert outputs for cross-attention
        expert_outputs_tensor = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, feature_dim]

        # 3. Cross-expert attention
        expert_outputs_attended, cross_attention_weights = self.cross_expert_attention(
            expert_outputs_tensor, expert_outputs_tensor, expert_outputs_tensor
        )

        # 4. Combine expert outputs with attention
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]

        # Weighted combination of attended expert outputs
        attended_fused_features = torch.sum(
            expert_outputs_attended * routing_weights_expanded, dim=1
        )  # [batch_size, feature_dim]

        # Original weighted combination for comparison
        original_fused_features = torch.sum(
            expert_outputs_tensor * routing_weights_expanded, dim=1
        )  # [batch_size, feature_dim]

        # 5. Operator attention integration (if enabled)
        if self.use_operator_attention and self.signal_operator_attention is not None:
            # Ensure correct input shape for operator attention (simplified)
            x_op = x.float()
            if len(x_op.shape) == 2 and x_op.shape[-1] != 4096:
                # Pad or resize to expected dimension
                if x_op.shape[-1] < 4096:
                    padding = 4096 - x_op.shape[-1]
                    x_op = F.pad(x_op, (0, padding))
                else:
                    x_op = x_op[:, :4096]

            op_features = self.signal_operator_attention(x_op)

            # Combine attended MoE features with operator attention features
            combined_features = torch.cat([attended_fused_features, op_features], dim=-1)
        else:
            combined_features = attended_fused_features

        # 6. Final fusion and classification
        if self.use_operator_attention:
            final_features = self.fusion_net(combined_features)
        else:
            final_features = attended_fused_features

        logits = self.classifier(final_features)

        # 7. Collect comprehensive metadata
        metadata = {
            'routing_weights': routing_weights,
            'expert_outputs': expert_outputs_tensor,
            'expert_outputs_attended': expert_outputs_attended,
            'cross_attention_weights': cross_attention_weights,
            'attended_fused_features': attended_fused_features,
            'original_fused_features': original_fused_features,
            'final_features': final_features,
            'routing_info': routing_info,
            'expert_metadata': expert_metadata,
            'logits': logits,
            'fusion_contribution': self._compute_fusion_contribution(
                attended_fused_features, original_fused_features
            )
        }

        # 8. Add operator attention metadata if available
        if self.use_operator_attention and self.signal_operator_attention is not None:
            metadata['operator_attention_features'] = op_features if 'op_features' in locals() else None

        # 9. Generate explanations
        if return_explanations:
            explanations = self._generate_comprehensive_explanations(x, metadata)
            metadata['explanations'] = explanations

        # Store for explainability methods
        self._last_forward_metadata = metadata

        return logits, metadata

    def _compute_fusion_contribution(self, attended_features: torch.Tensor,
                                   original_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the contribution of attention fusion."""
        # Compute similarity between attended and original features
        similarity = F.cosine_similarity(attended_features, original_features, dim=-1)

        # Compute improvement magnitude
        improvement = torch.norm(attended_features - original_features, dim=-1)

        return {
            'cosine_similarity': similarity,
            'improvement_magnitude': improvement,
            'avg_similarity': torch.mean(similarity),
            'avg_improvement': torch.mean(improvement)
        }

    def _generate_comprehensive_explanations(self, x: torch.Tensor,
                                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanations combining MoE and attention insights."""
        routing_weights = metadata['routing_weights']
        expert_metadata = metadata['expert_metadata']
        cross_attention = metadata['cross_attention_weights']

        # MoE-based explanations
        expert_activations = torch.mean(routing_weights, dim=0)
        most_active_expert = int(torch.argmax(expert_activations))

        # Attention-based explanations
        avg_cross_attention = torch.mean(cross_attention, dim=1)  # [batch_size, num_experts, num_experts]

        # Expert-specific attention entropy
        expert_attention_entropies = []
        for expert_meta in expert_metadata:
            expert_attention_entropies.append(expert_meta['attention_entropy'].item())

        explanations = {
            'moe_routing': {
                'expert_activations': expert_activations.detach().cpu().numpy().tolist(),
                'most_active_expert': most_active_expert,
                'routing_balance': 1.0 - torch.std(expert_activations).item()
            },
            'attention_patterns': {
                'cross_expert_attention': avg_cross_attention.detach().cpu().numpy().tolist(),
                'expert_attention_entropies': expert_attention_entropies,
                'attention_similarity': metadata['fusion_contribution']['cosine_similarity'].detach().cpu().numpy().tolist()
            },
            'fusion_analysis': {
                'fusion_improvement': metadata['fusion_contribution']['avg_improvement'].item(),
                'similarity_score': metadata['fusion_contribution']['avg_similarity'].item()
            },
            'expert_insights': []
        }

        # Individual expert insights
        for i, expert_meta in enumerate(expert_metadata):
            expert_insight = {
                'expert_id': i,
                'expert_type': expert_meta['expert_type'],
                'attention_entropy': expert_meta['attention_entropy'].item(),
                'feature_importance': expert_meta['feature_importance'].detach().cpu().numpy().tolist(),
                'routing_weight_mean': torch.mean(routing_weights[:, i]).item(),
                'base_expert_type': expert_meta['base_metadata']['expert_type']
            }
            explanations['expert_insights'].append(expert_insight)

        return explanations

    def get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """Get the signal transformation path through the fusion model."""
        if not self._last_forward_metadata:
            self.forward(input_data, return_explanations=True)

        metadata = self._last_forward_metadata
        path = []

        # 1. Routing stage
        routing_info = metadata['routing_info']
        path.append({
            'stage': 'operator_aware_routing',
            'attention_features': routing_info.get('operator_features', None),
            'expert_operator_scores': routing_info.get('expert_operator_scores', None)
        })

        # 2. Expert processing stage
        for i, expert_meta in enumerate(metadata['expert_metadata']):
            path.append({
                'stage': f'attention_expert_{i}',
                'expert_type': expert_meta['expert_type'],
                'attention_weights': expert_meta['attention_weights'].detach().cpu().numpy().tolist(),
                'attention_entropy': expert_meta['attention_entropy'].item()
            })

        # 3. Cross-attention fusion stage
        path.append({
            'stage': 'cross_expert_attention',
            'attention_weights': metadata['cross_attention_weights'].detach().cpu().numpy().tolist()
        })

        # 4. Final fusion stage
        path.append({
            'stage': 'final_fusion',
            'fusion_improvement': metadata['fusion_contribution']['avg_improvement'].item(),
            'similarity_score': metadata['fusion_contribution']['avg_similarity'].item()
        })

        return path

    def get_operator_graph(self) -> Dict[str, Any]:
        """Get the operator graph structure of the fusion model."""
        return {
            'model_type': 'MoE_OperatorAttention_Fusion',
            'num_experts': self.num_experts,
            'attention_heads': 4,
            'components': {
                'experts': [expert.get_expert_info() for expert in self.experts],
                'router': 'OperatorAwareRouter',
                'cross_attention': 'MultiheadAttention',
                'signal_operator': 'TSPN_OperatorAttention' if self.use_operator_attention else None
            },
            'connections': [
                {'from': 'router', 'to': 'experts', 'type': 'operator_aware_routing'},
                {'from': 'experts', 'to': 'cross_attention', 'type': 'expert_outputs'},
                {'from': 'cross_attention', 'to': 'fusion', 'type': 'attended_features'},
                {'from': 'signal_operator', 'to': 'fusion', 'type': 'operator_features', 'optional': True},
                {'from': 'fusion', 'to': 'classifier', 'type': 'final_features'}
            ],
            'explainability_features': [
                'Expert routing with operator awareness',
                'Self-attention within experts',
                'Cross-expert attention patterns',
                'Fusion contribution analysis',
                'Feature importance through attention',
                'Attention entropy tracking'
            ]
        }

    def get_attention_maps(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get comprehensive attention maps from the fusion model."""
        if not self._last_forward_metadata:
            self.forward(input_data, return_explanations=True)

        metadata = self._last_forward_metadata

        attention_maps = {
            'routing_weights': metadata['routing_weights'],
            'cross_expert_attention': metadata['cross_attention_weights'],
            'expert_attentions': {}
        }

        # Individual expert attention maps
        for i, expert_meta in enumerate(metadata['expert_metadata']):
            attention_maps['expert_attentions'][f'expert_{i}'] = {
                'self_attention': expert_meta['attention_weights'],
                'feature_importance': expert_meta['feature_importance']
            }

        # Operator attention maps if available
        if self.use_operator_attention and 'operator_attention_metadata' in metadata:
            op_metadata = metadata['operator_attention_metadata']
            if 'attention_weights' in op_metadata:
                attention_maps['operator_attention'] = op_metadata['attention_weights']

        return attention_maps

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': 'MoE_OperatorAttention_Fusion',
            'version': '1.0.0',
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'num_experts': self.num_experts,
            'use_operator_attention': self.use_operator_attention,
            'experts': [expert.get_expert_info() for expert in self.experts],
            'attention_config': {
                'attention_heads': 4,
                'cross_expert_attention': True,
                'self_attention_per_expert': True
            },
            'router_type': 'OperatorAwareRouter',
            'fusion_type': 'Attention_Enhanced_Fusion',
            'explainability_features': [
                'Dual-level attention (expert + cross-expert)',
                'Operator-aware routing decisions',
                'Fusion contribution analysis',
                'Attention entropy tracking',
                'Feature importance via attention',
                'Path signature analysis'
            ],
            'advantages': [
                'Combines expert specialization with attention interpretability',
                'Operator-aware routing for better expert selection',
                'Cross-expert attention for collaborative decision making',
                'Comprehensive explainability from multiple perspectives'
            ]
        }