"""
Transparent Signal Processing Network with Operator Attention

This module extends the original TSPN model to incorporate the Operator Attention mechanism.
It replaces the simple weighted combination of signal processing operators with an
attention-based approach that dynamically weights operators based on input signals.

Key Features:
- Operator Attention replaces traditional signal processing layers
- Maintains TSPN's transparency while adding adaptive operator selection
- Compatible with existing TSPN training infrastructure
- Provides interpretable operator weights for analysis

Based on the theory from: Paper/TII_operator_attention/Operator_Attention_Theory_Analysis.md
"""

import torch
import torch.nn as nn
from einops import rearrange
import copy

# Import TSPN components
from .TSPN import FeatureExtractorlayer, Classifier, Transparent_Signal_Processing_Network
from .operator_attention import SimpleOperatorAttention, OperatorLibrary


class OperatorAttentionSignalProcessingLayer(nn.Module):
    """
    A signal processing layer that uses Operator Attention instead of static weights.

    This layer replaces the original SignalProcessingLayer's static linear combination
    with dynamic operator attention based on the input signal characteristics.
    """

    def __init__(self, input_channels, output_channels, operator_attention_args, skip_connection=True):
        """
        Initialize Operator Attention Signal Processing Layer.

        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            operator_attention_args (dict): Arguments for Operator Attention module
            skip_connection (bool): Whether to use skip connection
        """
        super(OperatorAttentionSignalProcessingLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(input_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_skip_connection = skip_connection

        # Create dimension adapter for channel matching
        self.channel_adapter = nn.Linear(input_channels, output_channels)

        # Initialize Operator Attention module
        self.operator_attention = SimpleOperatorAttention(
            in_channels=input_channels,
            embed_dim=operator_attention_args.get('embed_dim', 64),
            hidden_dim=operator_attention_args.get('hidden_dim', 128),
            temperature=operator_attention_args.get('temperature', 1.0),
            sparse_regularization=operator_attention_args.get('sparse_regularization', 0.01),
            device=operator_attention_args.get('device', 'cpu')
        )

        # Skip connection if enabled
        if skip_connection:
            self.skip_connection = nn.Linear(input_channels, output_channels)

        # Store operator attention loss
        self.attention_loss = 0.0

    def forward(self, x, operator_modules):
        """
        Forward pass with Operator Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)
            operator_modules (dict): Dictionary of signal processing operators

        Returns:
            torch.Tensor: Output tensor of shape (B, L, C_out)
        """
        # Signal normalization
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')

        # Apply Operator Attention
        x_att, attention_weights, attention_loss = self.operator_attention(normed_x, operator_modules)

        # Channel adaptation if needed
        if x_att.shape[-1] != self.output_channels:
            x_att = self.channel_adapter(x_att)

        # Store attention loss for regularization
        self.attention_loss = attention_loss

        # Add skip connection if enabled
        if self.use_skip_connection:
            skip_output = self.skip_connection(normed_x)
            x_att = x_att + skip_output

        return x_att

    def get_attention_loss(self):
        """Get the attention sparsity loss."""
        return self.attention_loss


class TSPNWithOperatorAttention(Transparent_Signal_Processing_Network):
    """
    TSPN variant that incorporates Operator Attention mechanism.

    This class extends the original TSPN to use Operator Attention for adaptive
    operator selection, enhancing both performance and interpretability.
    """

    def __init__(self, signal_processing_modules, feature_extractor, args):
        """
        Initialize TSPN with Operator Attention.

        Args:
            signal_processing_modules: Signal processing modules (ignored, using Operator Library)
            feature_extractor: Feature extractor modules
            args: Configuration arguments with operator_attention settings
        """
        # Initialize parent class structure but we'll override the signal processing layers
        super(TSPNWithOperatorAttention, self).__init__(signal_processing_modules, feature_extractor, args)

        # Extract operator attention settings
        self.use_operator_attention = args.get('use_operator_attention', False)
        self.operator_attention_config = args.get('operator_attention', {})

        if self.use_operator_attention:
            self.init_operator_attention_layers()
            self.init_operator_library()

    def init_operator_attention_layers(self):
        """Initialize Operator Attention layers instead of traditional signal processing layers."""
        print('# build operator attention signal processing layers')

        in_channels = self.args.in_channels
        out_channels = int(self.args.out_channels * self.args.scale)

        self.signal_processing_layers = nn.ModuleList()

        # Create operator attention arguments
        op_att_args = {
            'embed_dim': self.operator_attention_config.get('embed_dim', 64),
            'hidden_dim': self.operator_attention_config.get('hidden_dim', 128),
            'temperature': self.operator_attention_config.get('temperature', 1.0),
            'sparse_regularization': self.operator_attention_config.get('sparse_regularization', 0.01),
            'device': self.args.device
        }

        for i in range(self.layer_num):
            # Create operator attention layer
            op_att_layer = OperatorAttentionSignalProcessingLayer(
                input_channels=in_channels,
                output_channels=out_channels,
                operator_attention_args=op_att_args,
                skip_connection=self.args.skip_connection
            ).to(self.args.device)

            self.signal_processing_layers.append(op_att_layer)
            in_channels = out_channels

        self.channel_for_feature = out_channels

    def init_operator_library(self):
        """Initialize the operator library."""
        # Get enabled operators from config
        enabled_operators = self.operator_attention_config.get('enabled_operators', ['FFT', 'HT', 'WF', 'I'])

        # Create operator library
        self.operator_library = OperatorLibrary(self.args, enabled_operators)

    def forward(self, x):
        """
        Forward pass of TSPN with Operator Attention.

        Args:
            x (torch.Tensor): Input signal tensor of shape (B, L, C)

        Returns:
            tuple: (output, attention_info) where output is the classification result
                   and attention_info contains attention weights for interpretability
        """
        total_attention_loss = 0.0
        attention_weights_list = []

        # Operator attention signal processing layers
        if self.use_operator_attention:
            for layer in self.signal_processing_layers:
                x = layer(x, self.operator_library.operators)
                total_attention_loss += layer.get_attention_loss()
                attention_weights_list.append(layer.operator_attention.get_attention_weights())
        else:
            # Fall back to original TSPN signal processing
            for layer in self.signal_processing_layers:
                x = layer(x)

        # Feature extraction (ensure correct input format)
        # The feature extractor expects input shape (B, C, L) but we have (B, L, C)
        x = rearrange(x, 'b l c -> b c l')
        x = self.feature_extractor_layers(x)

        # Classification
        x = self.clf(x)

        # Prepare attention info for interpretability
        attention_info = {
            'attention_weights': attention_weights_list,
            'total_attention_loss': total_attention_loss,
            'operator_importance': self.get_operator_importance()
        }

        return x, attention_info

    def get_operator_importance(self):
        """Get operator importance scores for interpretability."""
        if not self.use_operator_attention or len(self.signal_processing_layers) == 0:
            return None

        # Collect importance from all layers
        all_importance = []
        for layer in self.signal_processing_layers:
            layer_importance = layer.operator_attention.get_operator_importance()
            if layer_importance is not None:
                all_importance.append(layer_importance)

        # Average importance across layers
        if all_importance:
            avg_importance = {}
            for key in all_importance[0].keys():
                avg_importance[key] = sum([imp[key] for imp in all_importance]) / len(all_importance)
            return avg_importance

        return None

    def get_attention_summary(self):
        """Get a summary of attention patterns for analysis."""
        if not self.use_operator_attention:
            return None

        summary = {
            'num_layers': len(self.signal_processing_layers),
            'operator_importance': self.get_operator_importance(),
            'enabled_operators': list(self.operator_library.operators.keys())
        }

        return summary


def create_tspn_with_operator_attention(signal_processing_modules, feature_extractor, args):
    """
    Factory function to create TSPN with Operator Attention.

    Args:
        signal_processing_modules: Original signal processing modules (for compatibility)
        feature_extractor: Feature extractor modules
        args: Configuration arguments

    Returns:
        TSPNWithOperatorAttention: Configured model instance
    """
    return TSPNWithOperatorAttention(signal_processing_modules, feature_extractor, args)


def create_config_with_operator_attention(base_args, operator_attention_config):
    """
    Create configuration that includes operator attention settings.

    Args:
        base_args: Base TSPN configuration
        operator_attention_config: Operator attention specific configuration

    Returns:
        dict: Combined configuration
    """
    # Deep copy base args to avoid modification
    config = copy.deepcopy(base_args.__dict__ if hasattr(base_args, '__dict__') else base_args)

    # Add operator attention configuration
    config.update({
        'use_operator_attention': True,
        'operator_attention': operator_attention_config
    })

    return config


if __name__ == "__main__":
    # Test the TSPN with Operator Attention
    print("Testing TSPN with Operator Attention...")

    # Create mock args
    class MockArgs:
        def __init__(self):
            self.device = 'cpu'
            self.in_dim = 1024
            self.out_dim = 1024
            self.in_channels = 2
            self.out_channels = 4
            self.scale = 2
            self.skip_connection = True
            self.num_classes = 10
            self.layer_num = 2
            self.use_operator_attention = True
            self.operator_attention = {
                'embed_dim': 32,
                'hidden_dim': 64,
                'temperature': 1.0,
                'sparse_regularization': 0.01,
                'enabled_operators': ['FFT', 'HT', 'WF', 'I']
            }

    # Create mock modules
    def create_mock_modules():
        class MockFeatureExtractor:
            def __init__(self):
                pass
            def __len__(self):
                return 4

        return MockFeatureExtractor(), MockFeatureExtractor()

    # Test data
    batch_size, seq_len, channels = 4, 1024, 2
    x = torch.randn(batch_size, seq_len, channels)

    # Create model
    args = MockArgs()
    signal_processing_modules, feature_extractor = create_mock_modules()

    try:
        model = TSPNWithOperatorAttention(signal_processing_modules, feature_extractor, args)
        model.to(args.device)

        # Forward pass
        output, attention_info = model(x)

        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Attention loss: {attention_info['total_attention_loss'].item():.6f}")

        # Test attention summary
        summary = model.get_attention_summary()
        print(f"✅ Number of attention layers: {summary['num_layers']}")
        print(f"✅ Enabled operators: {summary['enabled_operators']}")
        print(f"✅ Operator importance: {summary['operator_importance']}")

        print("\n🎉 TSPN with Operator Attention test passed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()