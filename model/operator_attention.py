"""
Operator Attention Module for Transparent Signal Processing Networks

This module implements the Operator Attention mechanism as described in the TII paper.
The core idea is to replace the simple weighted combination of signal processing operators
with an attention-based approach that dynamically weights operators based on input signals.

Based on the theory from: Paper/TII_operator_attention/Operator_Attention_Theory_Analysis.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class SimpleOperatorAttention(nn.Module):
    """
    Simplified Operator Attention implementation for minimal viable version.

    This implements the core Operator Attention mechanism with:
    - 4 basic operators: FFT, HT, WF, I
    - Gated attention based on global signal features
    - Weighted fusion of operator outputs

    Mathematical formulation:
    - Input: X ∈ R^(B×L×C)
    - Global features: F_global = GlobalAvgPool(X) ∈ R^(B×C)
    - Gate weights: g = σ(W_g * F_global + b_g) ∈ R^(B×K)
    - Attention weights: α = softmax(g, dim=1) ∈ R^(B×K)
    - Output: Y = Σ_k α_k * o_k(X)

    Where K is the number of operators (4 in this simplified version).
    """

    def __init__(self, in_channels, embed_dim=64, hidden_dim=128, temperature=1.0,
                 sparse_regularization=0.01, device='cpu'):
        """
        Initialize Simple Operator Attention module.

        Args:
            in_channels (int): Number of input channels
            embed_dim (int): Dimension of operator embeddings
            hidden_dim (int): Hidden dimension of gate network
            temperature (float): Temperature parameter for attention distribution
            sparse_regularization (float): Weight for L1 sparsity regularization
            device (str): Device for computation ('cpu' or 'cuda')
        """
        super(SimpleOperatorAttention, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.sparse_regularization = sparse_regularization
        self.device = device

        # Number of operators in the simplified version
        self.num_operators = 4
        self.operator_names = ['FFT', 'HT', 'WF', 'I']

        # Initialize operator embeddings (learnable)
        self.operator_embeddings = nn.Parameter(
            torch.randn(self.num_operators, embed_dim, device=device) * 0.1
        )

        # Gate network for computing operator importance weights
        self.gate_network = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_operators),
            nn.Sigmoid()
        ).to(device)

        # Temperature parameter (learnable but constrained)
        self.temperature_param = nn.Parameter(torch.tensor(temperature, device=device))

        # Store attention weights for interpretability analysis
        self.last_attention_weights = None

    def compute_gate_weights(self, x):
        """
        Compute gate weights based on global signal features.

        Args:
            x (torch.Tensor): Input signal tensor of shape (B, L, C)

        Returns:
            torch.Tensor: Gate weights of shape (B, K)
        """
        # Global average pooling as basic signal feature
        global_features = torch.mean(x, dim=1)  # (B, C)

        # Compute gate weights through the gate network
        gate_weights = self.gate_network(global_features)  # (B, K)

        return gate_weights

    def compute_attention_weights(self, gate_weights):
        """
        Compute final attention weights from gate weights using temperature-scaled softmax.

        Args:
            gate_weights (torch.Tensor): Gate weights of shape (B, K)

        Returns:
            torch.Tensor: Normalized attention weights of shape (B, K)
        """
        # Apply temperature scaling
        scaled_weights = gate_weights / (self.temperature_param + 1e-8)

        # Apply softmax for normalization
        attention_weights = F.softmax(scaled_weights, dim=1)

        return attention_weights

    def apply_operators(self, x, operator_modules):
        """
        Apply each operator to the input signal.

        Args:
            x (torch.Tensor): Input signal tensor of shape (B, L, C)
            operator_modules (dict): Dictionary of operator modules

        Returns:
            list: List of operator outputs
        """
        operator_outputs = []

        for name in self.operator_names:
            if name in operator_modules:
                operator = operator_modules[name]
                op_output = operator(x)

                # Handle complex outputs (e.g., from FFT)
                if torch.is_complex(op_output):
                    # Convert complex to real by concatenating real and imaginary parts
                    real_output = op_output.real
                    imag_output = op_output.imag

                    # Take real part and ensure correct dimension
                    if real_output.shape[-1] == self.in_channels:
                        op_output = real_output
                    else:
                        # If dimensions don't match, truncate or pad
                        if real_output.shape[-1] > self.in_channels:
                            op_output = real_output[..., :self.in_channels]
                        else:
                            padding = self.in_channels - real_output.shape[-1]
                            op_output = F.pad(real_output, (0, padding))

                operator_outputs.append(op_output)
            else:
                # If operator not found, use identity
                operator_outputs.append(x)

        return operator_outputs

    def weighted_fusion(self, operator_outputs, attention_weights):
        """
        Perform weighted fusion of operator outputs based on attention weights.

        Args:
            operator_outputs (list): List of operator outputs
            attention_weights (torch.Tensor): Attention weights of shape (B, K)

        Returns:
            torch.Tensor: Fused output of shape (B, L, C)
        """
        batch_size, seq_len, channels = operator_outputs[0].shape

        # Find target sequence length (use the most common length)
        seq_lengths = [output.shape[1] for output in operator_outputs]
        target_seq_len = max(set(seq_lengths), key=seq_lengths.count)

        # Initialize output tensor
        output = torch.zeros(batch_size, target_seq_len, channels,
                           device=operator_outputs[0].device, dtype=operator_outputs[0].dtype)

        # Weighted sum of operator outputs
        for i, op_output in enumerate(operator_outputs):
            # Handle dimension mismatch by padding or truncating
            if op_output.shape[1] != target_seq_len:
                if op_output.shape[1] > target_seq_len:
                    # Truncate
                    op_output = op_output[:, :target_seq_len, :]
                else:
                    # Pad
                    pad_size = target_seq_len - op_output.shape[1]
                    op_output = F.pad(op_output, (0, 0, 0, pad_size))

            # Reshape attention weights for broadcasting
            weight = attention_weights[:, i:i+1, None]  # (B, 1, 1)
            output += weight * op_output

        return output

    def compute_sparsity_loss(self, attention_weights):
        """
        Compute L1 sparsity regularization loss for attention weights.

        Args:
            attention_weights (torch.Tensor): Attention weights of shape (B, K)

        Returns:
            torch.Tensor: Sparsity loss value
        """
        return torch.mean(torch.norm(attention_weights, p=1, dim=1))

    def forward(self, x, operator_modules):
        """
        Forward pass of Simple Operator Attention.

        Args:
            x (torch.Tensor): Input signal tensor of shape (B, L, C)
            operator_modules (dict): Dictionary of operator modules

        Returns:
            tuple: (output, attention_weights, sparsity_loss)
                - output: Fused signal output of shape (B, L, C)
                - attention_weights: Operator attention weights of shape (B, K)
                - sparsity_loss: L1 regularization loss
        """
        # 1. Compute gate weights based on input signal
        gate_weights = self.compute_gate_weights(x)  # (B, K)

        # 2. Compute normalized attention weights
        attention_weights = self.compute_attention_weights(gate_weights)  # (B, K)

        # 3. Apply each operator to the input signal
        operator_outputs = self.apply_operators(x, operator_modules)  # List of (B, L, C)

        # 4. Perform weighted fusion
        output = self.weighted_fusion(operator_outputs, attention_weights)  # (B, L, C)

        # 5. Compute sparsity regularization loss
        sparsity_loss = self.compute_sparsity_loss(attention_weights)

        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights.detach().cpu().numpy()

        return output, attention_weights, sparsity_loss

    def get_attention_weights(self):
        """
        Get the last computed attention weights for analysis.

        Returns:
            numpy.ndarray: Attention weights of shape (B, K)
        """
        return self.last_attention_weights

    def get_operator_importance(self):
        """
        Compute average operator importance across a batch.

        Returns:
            dict: Operator names mapped to their average importance scores
        """
        if self.last_attention_weights is None:
            return None

        avg_importance = np.mean(self.last_attention_weights, axis=0)
        return dict(zip(self.operator_names, avg_importance))


class OperatorLibrary(nn.Module):
    """
    Library of signal processing operators for use with Operator Attention.

    This class manages the collection of signal processing operators that
    can be used within the Operator Attention mechanism.
    """

    def __init__(self, args, operator_names=None):
        """
        Initialize operator library.

        Args:
            args: Configuration arguments
            operator_names (list): List of operator names to include
        """
        super(OperatorLibrary, self).__init__()

        if operator_names is None:
            operator_names = ['FFT', 'HT', 'WF', 'I']

        self.operator_names = operator_names
        self.args = args

        # Import signal processing modules
        from .Signal_processing import (
            FFTSignalProcessing, HilbertTransform, WaveFilters, Identity
        )

        # Create operator modules
        self.operators = nn.ModuleDict()

        # Create args for FFT (output dimension needs adjustment)
        fft_args = self._create_fft_args()

        operator_map = {
            'FFT': (FFTSignalProcessing, fft_args),
            'HT': (HilbertTransform, args),
            'WF': (WaveFilters, args),
            'I': (Identity, args)
        }

        for name in operator_names:
            if name in operator_map:
                operator_class, op_args = operator_map[name]
                self.operators[name] = operator_class(op_args)

        # Ensure all operators are on the correct device
        self.to(args.device)

    def _create_fft_args(self):
        """Create modified args for FFT operator (output dimension adjustment)."""
        import copy
        fft_args = copy.deepcopy(self.args)
        fft_args.out_dim = self.args.in_dim // 2 + 1
        return fft_args

    def forward(self, x, operator_name=None):
        """
        Apply specific operator or return all operators.

        Args:
            x (torch.Tensor): Input tensor
            operator_name (str): Specific operator to apply, if None returns dict

        Returns:
            Union[torch.Tensor, dict]: Output tensor or dictionary of all outputs
        """
        if operator_name is not None:
            if operator_name in self.operators:
                return self.operators[operator_name](x)
            else:
                raise ValueError(f"Operator '{operator_name}' not found in library")
        else:
            outputs = {}
            for name, operator in self.operators.items():
                outputs[name] = operator(x)
            return outputs


def create_simple_operator_attention_config():
    """
    Create a simple configuration dictionary for Operator Attention.

    Returns:
        dict: Configuration dictionary
    """
    config = {
        'operator_attention': {
            'embed_dim': 64,
            'hidden_dim': 128,
            'temperature': 1.0,
            'sparse_regularization': 0.01,
            'use_operator_attention': True
        },
        'operators': {
            'enabled': ['FFT', 'HT', 'WF', 'I'],
            'learnable_embedding': True
        }
    }
    return config


if __name__ == "__main__":
    # Simple test for the operator attention module
    print("Testing Simple Operator Attention Module...")

    # Create mock args
    class MockArgs:
        def __init__(self):
            self.device = 'cpu'
            self.in_dim = 1024
            self.out_dim = 1024
            self.in_channels = 2
            self.out_channels = 2
            self.scale = 2
            self.f_c_mu = 0.1
            self.f_c_sigma = 0.01
            self.f_b_mu = 0.1
            self.f_b_sigma = 0.01

    # Test data
    batch_size, seq_len, channels = 4, 1024, 2
    x = torch.randn(batch_size, seq_len, channels)

    # Create operator attention module
    args = MockArgs()
    operator_attention = SimpleOperatorAttention(
        in_channels=channels,
        embed_dim=32,
        hidden_dim=64,
        device=args.device
    )

    # Create operator library
    operator_lib = OperatorLibrary(args)

    # Forward pass
    try:
        output, attention_weights, sparsity_loss = operator_attention(x, operator_lib.operators)

        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape}")
        print(f"✅ Sparsity loss: {sparsity_loss.item():.6f}")
        print(f"✅ Attention sum per sample: {attention_weights.sum(dim=1)}")

        # Test operator importance extraction
        importance = operator_attention.get_operator_importance()
        print(f"✅ Operator importance: {importance}")

        print("\n🎉 Simple Operator Attention module test passed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()