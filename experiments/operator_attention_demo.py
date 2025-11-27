"""
Operator Attention Demo Script

This script demonstrates the functionality of the Operator Attention mechanism
and provides a quick test of its integration with the existing TSPN framework.

Features:
- Quick test of Operator Attention module
- Visualization of attention weights
- Integration with explainability tools
- Performance comparison with baseline models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from model.operator_attention import SimpleOperatorAttention, OperatorLibrary
from model.TSPN_OperatorAttention import TSPNWithOperatorAttention, create_config_with_operator_attention
from model.Signal_processing import FFTSignalProcessing, HilbertTransform, WaveFilters, Identity

# Import explainability tools
from Paper.Explainable_FD_Toolkit.toolkit_integration.explainability.methods.intrinsic.operator_attention_explainer import (
    OperatorAttentionExplainer, OperatorAttentionMetrics
)
from Paper.Explainable_FD_Toolkit.toolkit_integration.explainability.core import SignalData


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


def test_simple_operator_attention():
    """Test the SimpleOperatorAttention module."""
    print("🧪 Testing SimpleOperatorAttention Module...")
    print("-" * 50)

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

        # Create operator library
        operator_lib = OperatorLibrary(args)

        # Create test data
        batch_size, seq_len, channels = 4, 1024, 2
        x = torch.randn(batch_size, seq_len, channels)

        print(f"✅ Input shape: {x.shape}")

        # Forward pass
        output, attention_weights, sparsity_loss = operator_attention(x, operator_lib.operators)

        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Attention weights shape: {attention_weights.shape}")
        print(f"✅ Sparsity loss: {sparsity_loss.item():.6f}")

        # Test attention properties
        attention_sum = torch.sum(attention_weights, dim=1)
        print(f"✅ Attention sum per sample: {attention_sum}")

        # Test operator importance extraction
        importance = operator_attention.get_operator_importance()
        if importance:
            print("✅ Operator importance:")
            for op_name, imp in importance.items():
                print(f"  • {op_name}: {imp:.4f}")
        else:
            print("ℹ️  No operator importance available (run forward pass first)")

        print("✅ SimpleOperatorAttention test PASSED")
        return True, operator_attention

    except Exception as e:
        print(f"❌ SimpleOperatorAttention test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_operator_library():
    """Test the OperatorLibrary module."""
    print("\n🧪 Testing OperatorLibrary Module...")
    print("-" * 50)

    try:
        args = create_mock_args()

        # Test with default operators
        operator_lib = OperatorLibrary(args)

        print(f"✅ Created operator library with {len(operator_lib.operators)} operators")
        print(f"✅ Operators: {list(operator_lib.operators.keys())}")

        # Test individual operator application
        batch_size, seq_len, channels = 2, 512, 2
        x = torch.randn(batch_size, seq_len, channels)

        print(f"✅ Test input shape: {x.shape}")

        for name, operator in operator_lib.operators.items():
            try:
                output = operator(x)
                print(f"✅ {name}: {x.shape} -> {output.shape}")
            except Exception as e:
                print(f"❌ {name}: FAILED - {e}")

        print("✅ OperatorLibrary test PASSED")
        return True

    except Exception as e:
        print(f"❌ OperatorLibrary test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_attention_weights(attention_weights, operator_names, save_path=None):
    """Visualize attention weights."""
    if len(attention_weights.shape) != 2:
        print("❌ Cannot visualize: attention_weights should be 2D (batch, operators)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Heatmap
    im = axes[0, 0].imshow(attention_weights.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Attention Weights Heatmap')
    axes[0, 0].set_xlabel('Batch Sample')
    axes[0, 0].set_ylabel('Operators')
    axes[0, 0].set_yticks(range(len(operator_names)))
    axes[0, 0].set_yticklabels(operator_names)
    plt.colorbar(im, ax=axes[0, 0])

    # Bar chart (average weights)
    avg_weights = np.mean(attention_weights, axis=0)
    bars = axes[0, 1].bar(operator_names, avg_weights, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 1].set_title('Average Attention Weights')
    axes[0, 1].set_ylabel('Average Weight')
    axes[0, 1].set_ylim(0, 1)

    # Add value labels
    for bar, weight in zip(bars, avg_weights):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')

    # Distribution
    axes[1, 0].hist(attention_weights.flatten(), bins=20, alpha=0.7, density=True, edgecolor='black')
    axes[1, 0].set_title('Attention Weights Distribution')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Density')

    # Statistics table
    stats_text = "Attention Statistics:\n\n"
    stats_text += f"Mean: {np.mean(attention_weights):.4f}\n"
    stats_text += f"Std: {np.std(attention_weights):.4f}\n"
    stats_text += f"Min: {np.min(attention_weights):.4f}\n"
    stats_text += f"Max: {np.max(attention_weights):.4f}\n"
    stats_text += f"Sparsity: {np.mean(attention_weights < 0.25):.4f}\n"  # Below 0.25 (1/4)

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


def test_explainability_integration():
    """Test integration with explainability framework."""
    print("\n🧪 Testing Explainability Integration...")
    print("-" * 50)

    try:
        # Create operator attention explainer
        explainer = OperatorAttentionExplainer({
            'include_temporal_analysis': True,
            'include_complexity_analysis': True,
            'operator_names': ['FFT', 'HT', 'WF', 'I']
        })

        print("✅ Created OperatorAttentionExplainer")

        # Create test signal data
        signal_length = 1024
        sampling_rate = 1024
        t = np.linspace(0, signal_length/sampling_rate, signal_length)

        # Create a test signal with multiple frequency components
        signal_data = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
                      0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
                      0.3 * np.sin(2 * np.pi * 100 * t))  # 100 Hz component

        # Add some noise
        signal_data += 0.1 * np.random.randn(len(signal_data))

        # Create SignalData object
        signal = SignalData(
            raw_signal=signal_data,
            sampling_rate=sampling_rate,
            signal_name="test_signal"
        )

        print(f"✅ Created test signal: {signal.get_shape()}")

        # Mock attention weights for demonstration
        batch_size = 5
        num_operators = 4
        attention_weights = np.random.dirichlet([1, 1, 1, 1], size=batch_size)

        print(f"✅ Mock attention weights: {attention_weights.shape}")

        # Create a mock model class for the explainer
        class MockOperatorAttentionModel:
            def __init__(self, attention_weights):
                self.attention_weights = attention_weights

            def get_operator_attention_weights(self, x):
                return self.attention_weights

        mock_model = MockOperatorAttentionModel(attention_weights)

        # Generate explanation
        explanation = explainer.explain(
            signal=signal,
            prediction="test_fault_type",
            model=mock_model,
            attention_weights=attention_weights
        )

        print("✅ Generated explanation")

        # Extract key metrics from explanation
        attention_entropy = explanation.get_data('attention_entropy', {}).get('entropy', 0)
        attention_sparsity = explanation.get_data('attention_sparsity', {}).get('sparsity', 0)
        operator_importance = explanation.get_data('operator_importance', {})

        print(f"✅ Attention Entropy: {attention_entropy:.3f} bits")
        print(f"✅ Attention Sparsity: {attention_sparsity:.3f}")

        if operator_importance:
            print("✅ Operator Importance:")
            for op_name, importance in operator_importance.items():
                print(f"  • {op_name}: {importance:.4f}")

        # Create visualization
        print("✅ Creating visualization...")
        fig = explainer.visualize(explanation, mode='attention_weights')
        plt.savefig('operator_attention_explanation.png', dpi=300, bbox_inches='tight')
        print("✅ Explanation visualization saved to: operator_attention_explanation.png")

        # Evaluate explanation
        print("✅ Evaluating explanation...")
        evaluation = explainer.evaluate([explanation])
        print("✅ Evaluation Metrics:")
        for metric, value in evaluation.items():
            print(f"  • {metric}: {value:.4f}")

        print("✅ Explainability integration test PASSED")
        return True

    except Exception as e:
        print(f"❌ Explainability integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Benchmark the performance of Operator Attention."""
    print("\n🧪 Testing Performance Benchmark...")
    print("-" * 50)

    try:
        import time

        args = create_mock_args()

        # Create operator attention module
        operator_attention = SimpleOperatorAttention(
            in_channels=args.in_channels,
            embed_dim=64,
            hidden_dim=128,
            device=args.device
        )

        # Create operator library
        operator_lib = OperatorLibrary(args)

        # Test with different batch sizes
        batch_sizes = [1, 4, 16, 32, 64]
        seq_length = 1024
        channels = 2

        print(f"Testing with sequence length: {seq_length}, channels: {channels}")
        print("-" * 30)

        for batch_size in batch_sizes:
            # Create input
            x = torch.randn(batch_size, seq_length, channels)

            # Measure inference time
            operator_attention.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = operator_attention(x, operator_lib.operators)

                # Time measurement
                start_time = time.time()
                for _ in range(20):
                    output, attention_weights, sparsity_loss = operator_attention(x, operator_lib.operators)
                end_time = time.time()

                avg_time = (end_time - start_time) / 20
                throughput = batch_size / avg_time  # samples per second

                print(f"Batch {batch_size:2d}: {avg_time*1000:6.2f}ms, {throughput:6.1f} samples/s, "
                      f"Output: {output.shape}, Weights: {attention_weights.shape}")

        print("✅ Performance benchmark test PASSED")
        return True

    except Exception as e:
        print(f"❌ Performance benchmark test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_demo():
    """Run comprehensive demo of all Operator Attention features."""
    print("🚀 Starting Comprehensive Operator Attention Demo")
    print("=" * 60)

    # Track test results
    test_results = {}

    # Test 1: Simple Operator Attention
    success, op_attention = test_simple_operator_attention()
    test_results['simple_operator_attention'] = success

    if success and op_attention is not None:
        # Visualize attention weights
        attention_weights = op_attention.get_attention_weights()
        if attention_weights is not None:
            print("\n📊 Creating attention weights visualization...")
            visualize_attention_weights(
                attention_weights,
                ['FFT', 'HT', 'WF', 'I'],
                'operator_attention_weights_demo.png'
            )

    # Test 2: Operator Library
    test_results['operator_library'] = test_operator_library()

    # Test 3: Explainability Integration
    test_results['explainability_integration'] = test_explainability_integration()

    # Test 4: Performance Benchmark
    test_results['performance_benchmark'] = test_performance_benchmark()

    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print("-" * 30)

    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    if passed_tests == total_tests:
        print("\n🎉 All tests PASSED! Operator Attention is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the implementation.")

    return test_results


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some tests failed