"""
Simple Operator Attention Test

This script provides a basic test of the Operator Attention module
without complex dependencies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from model.operator_attention import SimpleOperatorAttention, OperatorLibrary


def create_mock_args():
    """Create mock arguments for model initialization."""
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

    return MockArgs()


def test_operator_attention():
    """Test the Operator Attention module."""
    print("🧪 Testing Operator Attention Module")
    print("=" * 50)

    try:
        # Create mock args
        args = create_mock_args()

        # Create operator attention module
        operator_attention = SimpleOperatorAttention(
            in_channels=args.in_channels,
            embed_dim=32,
            hidden_dim=64,
            temperature=1.0,
            sparse_regularization=0.01,
            device=args.device
        )

        print(f"✅ Created SimpleOperatorAttention")
        print(f"  - Input channels: {args.in_channels}")
        print(f"  - Embedding dimension: 32")
        print(f"  - Hidden dimension: 64")
        print(f"  - Number of operators: {operator_attention.num_operators}")
        print(f"  - Operator names: {operator_attention.operator_names}")

        # Create operator library
        operator_lib = OperatorLibrary(args)
        print(f"✅ Created OperatorLibrary with {len(operator_lib.operators)} operators")

        # Create test data
        batch_size, seq_len, channels = 4, 1024, 2
        x = torch.randn(batch_size, seq_len, channels)

        print(f"✅ Input tensor shape: {x.shape}")

        # Forward pass
        output, attention_weights, sparsity_loss = operator_attention(x, operator_lib.operators)

        print(f"✅ Output tensor shape: {output.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape}")
        print(f"✅ Sparsity loss: {sparsity_loss.item():.6f}")

        # Verify attention weight properties
        attention_sum = torch.sum(attention_weights, dim=1)
        print(f"✅ Attention weights sum (should be ~1.0): {attention_sum}")

        # Extract operator importance
        importance = operator_attention.get_operator_importance()
        if importance:
            print("✅ Operator importance scores:")
            for op_name, imp in importance.items():
                print(f"  • {op_name}: {imp:.4f}")

        # Test with different temperatures
        print(f"\n🔬 Testing different temperature parameters...")
        temperatures = [0.5, 1.0, 2.0]

        for temp in temperatures:
            operator_attention.temperature_param.data = torch.tensor(temp)
            _, attn_weights_temp, _ = operator_attention(x, operator_lib.operators)
            avg_entropy = -torch.mean(torch.sum(attn_weights_temp * torch.log(attn_weights_temp + 1e-10), dim=1))
            print(f"  Temperature {temp:.1f}: Avg Entropy = {avg_entropy:.3f}")

        print("✅ All tests PASSED!")
        return True, attention_weights.detach().numpy(), operator_attention.operator_names

    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def visualize_attention_weights(attention_weights, operator_names, save_path=None):
    """Visualize attention weights."""
    print("📊 Creating visualization...")

    if len(attention_weights.shape) != 2:
        print(f"❌ Cannot visualize: expected 2D array, got {attention_weights.shape}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Heatmap
    im1 = axes[0, 0].imshow(attention_weights.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Attention Weights Heatmap')
    axes[0, 0].set_xlabel('Batch Sample')
    axes[0, 0].set_ylabel('Operators')
    axes[0, 0].set_yticks(range(len(operator_names)))
    axes[0, 0].set_yticklabels(operator_names)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Average weights bar chart
    avg_weights = np.mean(attention_weights, axis=0)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = axes[0, 1].bar(operator_names, avg_weights, color=colors[:len(operator_names)])
    axes[0, 1].set_title('Average Attention Weights')
    axes[0, 1].set_ylabel('Average Weight')
    axes[0, 1].set_ylim(0, 1)

    # Add value labels on bars
    for bar, weight in zip(bars, avg_weights):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')

    # 3. Weights distribution
    axes[1, 0].hist(attention_weights.flatten(), bins=20, alpha=0.7, density=True,
                    color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Attention Weights Distribution')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Statistics table
    stats_text = "Attention Statistics:\n\n"
    stats_text += f"Mean: {np.mean(attention_weights):.4f}\n"
    stats_text += f"Std: {np.std(attention_weights):.4f}\n"
    stats_text += f"Min: {np.min(attention_weights):.4f}\n"
    stats_text += f"Max: {np.max(attention_weights):.4f}\n"

    # Compute entropy
    entropies = []
    for weights in attention_weights:
        weights_norm = weights + 1e-10  # Avoid log(0)
        weights_norm = weights_norm / np.sum(weights_norm)
        entropy = -np.sum(weights_norm * np.log2(weights_norm))
        entropies.append(entropy)

    stats_text += f"Avg Entropy: {np.mean(entropies):.4f} bits\n"
    stats_text += f"Sparsity: {np.mean(attention_weights < 0.25):.4f}"

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].set_title('Statistics')
    axes[1, 1].axis('off')

    plt.suptitle('Operator Attention Weights Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")

    plt.show()


def test_attention_vs_self_attention():
    """Compare operator attention with theoretical self-attention complexity."""
    print("\n🏁 Complexity Comparison Analysis")
    print("=" * 50)

    # Define parameters
    batch_size = 4
    seq_length = 1024
    channels = 2
    num_operators = 4

    # Operator Attention complexity: O(K * L * C)
    op_attention_flops = num_operators * seq_length * channels
    op_attention_memory = num_operators * seq_length * channels

    # Standard Self-Attention complexity: O(L^2 * C)
    self_attention_flops = seq_length * seq_length * channels
    self_attention_memory = seq_length * seq_length

    print(f"📐 Theoretical Complexity Analysis:")
    print(f"  Sequence Length (L): {seq_length}")
    print(f"  Channels (C): {channels}")
    print(f"  Number of Operators (K): {num_operators}")
    print()

    print(f"  Operator Attention:")
    print(f"    - FLOPs: {op_attention_flops:,}")
    print(f"    - Memory: {op_attention_memory:,}")
    print()

    print(f"  Self-Attention:")
    print(f"    - FLOPs: {self_attention_flops:,}")
    print(f"    - Memory: {self_attention_memory:,}")
    print()

    # Compute ratios
    flops_ratio = op_attention_flops / self_attention_flops
    memory_ratio = op_attention_memory / self_attention_memory

    print(f"📊 Complexity Ratios (OA / SA):")
    print(f"    - FLOPs Ratio: {flops_ratio:.6f} ({1/flops_ratio:.1f}x reduction)")
    print(f"    - Memory Ratio: {memory_ratio:.6f} ({1/memory_ratio:.1f}x reduction)")
    print()

    # For typical values where L >> K
    print(f"🎯 Key Insight:")
    print(f"  When L ({seq_length}) >> K ({num_operators}):")
    print(f"  • Operator Attention is ~{1/flops_ratio:.0f}x more efficient in FLOPs")
    print(f"  • Operator Attention is ~{1/memory_ratio:.0f}x more efficient in memory")


def main():
    """Run all tests."""
    print("🚀 Operator Attention Module Test Suite")
    print("=" * 60)

    # Test the core module
    success, attention_weights, operator_names = test_operator_attention()

    if success and attention_weights is not None:
        # Create visualization
        print("\n📈 Creating attention weight visualizations...")
        visualize_attention_weights(
            attention_weights,
            operator_names,
            'operator_attention_test.png'
        )

        # Complexity analysis
        test_attention_vs_self_attention()

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Visualization saved to: operator_attention_test.png")
    else:
        print(f"\n❌ Tests failed. Please check the implementation.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)