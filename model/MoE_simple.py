"""
Simplified MoE Model for Unified Fault Diagnosis Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class MoEModel(nn.Module):
    """
    Simplified MoE Model for fault diagnosis
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        super(MoEModel, self).__init__()

        # Extract parameters
        self.input_dim = getattr(args, 'in_dim', 4096)
        self.in_channels = getattr(args, 'in_channels', 2)
        self.out_channels = getattr(args, 'out_channels', 3)
        self.num_classes = getattr(args, 'num_classes', 5)
        self.num_experts = 3

        # Simplified signal processing
        # Use a simple CNN instead of complex TSPN layers
        self.signal_processor = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.out_channels * self.input_dim // 16, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 10)  # Expert output
            ) for _ in range(self.num_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.out_channels * self.input_dim // 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_experts)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        # Process signal
        x = self.signal_processor(x)

        # Global pooling
        x = F.adaptive_avg_pool1d(x, self.input_dim // 16)  # Reduce sequence length
        x = x.view(x.size(0), -1)  # Flatten

        # Compute gate scores
        gate_scores = self.gate(x)  # (batch_size, num_experts)
        gate_weights = F.softmax(gate_scores, dim=-1)

        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, 10)

        # Weighted sum of expert outputs
        moe_output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)  # (batch_size, 10)

        # Final classification
        logits = self.classifier(moe_output)

        return logits