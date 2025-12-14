"""
Noise robustness testing for Fuzzy-XFD
Tests model performance under various noise conditions and SNR levels
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from trainer.trainer_basic import Basic_plmodel
from utils.enhanced_metrics import EnhancedMetricsCollector


class NoiseRobustnessTester:
    """
    Comprehensive noise robustness testing framework
    """

    def __init__(self, model_path: str, config_path: str, device: str = 'cuda:0'):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize model and load weights
        self.model = None
        self.load_model()

        # Noise types
        self.noise_types = {
            'gaussian': self.add_gaussian_noise,
            'uniform': self.add_uniform_noise,
            'impulse': self.add_impulse_noise,
            'colored': self.add_colored_noise
        }

    def load_model(self):
        """Load trained model from checkpoint"""
        # Create model instance
        self.model = Basic_plmodel(self.config)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)

    def add_gaussian_noise(self, data: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian white noise to data"""
        # Calculate signal power
        signal_power = torch.mean(data ** 2, dim=-1, keepdim=True)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
        return data + noise

    def add_uniform_noise(self, data: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add uniform noise to data"""
        signal_power = torch.mean(data ** 2, dim=-1, keepdim=True)

        # Calculate noise power for uniform distribution
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Uniform distribution has variance = (b-a)^2/12
        # Assuming symmetric around 0: variance = a^2/3
        noise_std = torch.sqrt(3 * noise_power)
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_std

        return data + noise

    def add_impulse_noise(self, data: torch.Tensor, snr_db: float, probability: float = 0.05) -> torch.Tensor:
        """Add impulse (salt-and-pepper) noise"""
        # Create mask for impulse noise
        mask = torch.rand_like(data) < probability

        # Calculate impulse magnitude based on SNR
        signal_power = torch.mean(data ** 2)
        snr_linear = 10 ** (snr_db / 10)
        impulse_power = signal_power / snr_linear * probability
        impulse_magnitude = torch.sqrt(impulse_power)

        # Add impulse noise
        noise = torch.zeros_like(data)
        noise[mask] = impulse_magnitude * torch.sign(torch.randn_like(data[mask]))

        return data + noise

    def add_colored_noise(self, data: torch.Tensor, snr_db: float, color: str = 'pink') -> torch.Tensor:
        """Add colored noise (pink, brown, etc.)"""
        batch_size, seq_len = data.shape

        # Generate white noise
        white_noise = torch.randn(batch_size, seq_len, device=self.device)

        # Apply frequency domain coloring
        fft = torch.fft.fft(white_noise)
        freqs = torch.fft.fftfreq(seq_len, device=self.device)

        # Create frequency filter based on color
        if color == 'pink':
            # 1/f frequency response
            freq_filter = torch.sqrt(1.0 / torch.abs(freqs + 1e-10))
        elif color == 'brown':
            # 1/f^2 frequency response
            freq_filter = 1.0 / (torch.abs(freqs) + 1e-10)
        elif color == 'blue':
            # f frequency response
            freq_filter = torch.sqrt(torch.abs(freqs))
        else:
            freq_filter = torch.ones_like(freqs)

        # Apply filter
        freq_filter[0] = 0  # Remove DC component
        colored_noise_fft = fft * freq_filter
        colored_noise = torch.real(torch.fft.ifft(colored_noise_fft))

        # Scale to desired SNR
        signal_power = torch.mean(data ** 2, dim=-1, keepdim=True)
        noise_power = torch.mean(colored_noise ** 2, dim=-1, keepdim=True)

        snr_linear = 10 ** (snr_db / 10)
        colored_noise = colored_noise * torch.sqrt(signal_power / (noise_power * snr_linear))

        return data + colored_noise

    def evaluate_with_noise(self, test_loader, noise_type: str, snr_db: float) -> Dict[str, float]:
        """Evaluate model performance with specified noise"""
        metrics_collector = EnhancedMetricsCollector(
            num_classes=self.config['args']['num_classes']
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Testing {noise_type} @ {snr_db}dB'):
                # Get data and labels
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                    y = batch['label'].to(self.device)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                # Add noise
                if noise_type in self.noise_types:
                    if noise_type == 'impulse':
                        x_noisy = self.noise_types[noise_type](x, snr_db)
                    else:
                        x_noisy = self.noise_types[noise_type](x, snr_db)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                # Get model predictions
                outputs = self.model(x_noisy)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('predictions'))
                else:
                    logits = outputs

                # Get probabilities and predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Update metrics
                correct += (preds == y).sum().item()
                total += y.size(0)

                metrics_collector.update(preds.cpu(), y.cpu(), probs.cpu())

        # Calculate metrics
        accuracy = correct / total
        per_class_metrics = metrics_collector.compute_per_class_metrics()
        safety_metrics = metrics_collector.compute_safety_metrics()

        results = {
            'accuracy': accuracy,
            'noise_type': noise_type,
            'snr_db': snr_db,
            'safety_metrics': safety_metrics
        }

        # Add per-class metrics
        for _, row in per_class_metrics.iterrows():
            class_name = row['class']
            results[f'accuracy_{class_name}'] = row['f1-score']

        return results

    def run_comprehensive_test(self, test_loader, snr_range: List[float] = None,
                             noise_types: List[str] = None) -> pd.DataFrame:
        """
        Run comprehensive noise robustness test

        Args:
            test_loader: Test data loader
            snr_range: List of SNR values to test (default: -10 to 30 dB)
            noise_types: List of noise types to test (default: all)

        Returns:
            DataFrame with all results
        """
        if snr_range is None:
            snr_range = [-10, -5, 0, 5, 10, 15, 20, 25, 30]

        if noise_types is None:
            noise_types = list(self.noise_types.keys())

        results = []

        print("Starting comprehensive noise robustness test...")
        print(f"SNR range: {snr_range} dB")
        print(f"Noise types: {noise_types}")

        # Test clean performance first
        print("\nTesting clean performance...")
        clean_results = self.evaluate_clean(test_loader)
        clean_results['noise_type'] = 'clean'
        clean_results['snr_db'] = float('inf')
        results.append(clean_results)

        # Test with noise
        for noise_type in noise_types:
            print(f"\nTesting {noise_type} noise...")
            for snr in snr_range:
                result = self.evaluate_with_noise(test_loader, noise_type, snr)
                results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save results
        output_dir = Path(self.config['args']['save_dir']) / 'noise_robustness'
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / 'noise_robustness_results.csv', index=False)

        # Save detailed report
        self.generate_noise_report(df, output_dir)

        print(f"\nResults saved to {output_dir}")
        return df

    def evaluate_clean(self, test_loader) -> Dict[str, float]:
        """Evaluate model on clean test data"""
        metrics_collector = EnhancedMetricsCollector(
            num_classes=self.config['args']['num_classes']
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                    y = batch['label'].to(self.device)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                # Get predictions
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('predictions'))
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                metrics_collector.update(preds.cpu(), y.cpu(), probs.cpu())

        accuracy = correct / total
        per_class_metrics = metrics_collector.compute_per_class_metrics()
        safety_metrics = metrics_collector.compute_safety_metrics()

        results = {
            'accuracy': accuracy,
            'safety_metrics': safety_metrics
        }

        for _, row in per_class_metrics.iterrows():
            class_name = row['class']
            results[f'accuracy_{class_name}'] = row['f1-score']

        return results

    def generate_noise_report(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive noise robustness report"""
        report = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'test_date': pd.Timestamp.now().isoformat(),
            'summary': {},
            'detailed_results': results_df.to_dict('records')
        }

        # Calculate summary statistics
        clean_acc = results_df[results_df['noise_type'] == 'clean']['accuracy'].iloc[0]

        for noise_type in results_df['noise_type'].unique():
            if noise_type == 'clean':
                continue

            noise_df = results_df[results_df['noise_type'] == noise_type]

            # Find SNR threshold for 10% performance drop
            threshold_acc = clean_acc * 0.9
            threshold_snr = None

            for _, row in noise_df.iterrows():
                if row['accuracy'] < threshold_acc:
                    threshold_snr = row['snr_db']
                    break

            report['summary'][noise_type] = {
                'clean_accuracy': clean_acc,
                'worst_case_accuracy': noise_df['accuracy'].min(),
                'best_case_accuracy': noise_df['accuracy'].max(),
                'accuracy_at_0dB': noise_df[noise_df['snr_db'] == 0]['accuracy'].iloc[0]
                if 0 in noise_df['snr_db'].values else None,
                'snr_threshold_10pct_drop': threshold_snr
            }

        # Save report
        with open(output_dir / 'noise_robustness_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Create visualizations
        self.create_noise_plots(results_df, output_dir)

    def create_noise_plots(self, results_df: pd.DataFrame, output_dir: Path):
        """Create noise robustness visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Accuracy vs SNR for all noise types
        clean_acc = results_df[results_df['noise_type'] == 'clean']['accuracy'].iloc[0]

        for noise_type in results_df['noise_type'].unique():
            if noise_type == 'clean':
                continue

            noise_df = results_df[results_df['noise_type'] == noise_type]
            axes[0, 0].plot(noise_df['snr_db'], noise_df['accuracy'],
                          marker='o', label=noise_type)

        axes[0, 0].axhline(y=clean_acc, color='black', linestyle='--',
                          label='Clean Performance')
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs SNR for Different Noise Types')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Performance degradation at 0dB
        zero_db_data = []
        noise_types = []

        for noise_type in results_df['noise_type'].unique():
            if noise_type == 'clean':
                continue

            noise_df = results_df[results_df['noise_type'] == noise_type]
            if 0 in noise_df['snr_db'].values:
                zero_db_data.append(noise_df[noise_df['snr_db'] == 0]['accuracy'].iloc[0])
                noise_types.append(noise_type)

        if zero_db_data:
            bars = axes[0, 1].bar(noise_types, zero_db_data)
            axes[0, 1].axhline(y=clean_acc, color='black', linestyle='--')
            axes[0, 1].set_ylabel('Accuracy at 0dB SNR')
            axes[0, 1].set_title('Performance Comparison at 0dB SNR')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Color bars based on performance
            for bar, acc in zip(bars, zero_db_data):
                if acc >= clean_acc * 0.9:
                    bar.set_color('green')
                elif acc >= clean_acc * 0.7:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')

        # Plot 3: SNR threshold for 10% drop
        thresholds = []
        noise_names = []

        for noise_type in results_df['noise_type'].unique():
            if noise_type == 'clean':
                continue

            noise_df = results_df[results_df['noise_type'] == noise_type]
            threshold_acc = clean_acc * 0.9
            threshold_snr = None

            # Find the SNR where accuracy drops below threshold
            sorted_df = noise_df.sort_values('snr_db', ascending=False)
            for _, row in sorted_df.iterrows():
                if row['accuracy'] < threshold_acc:
                    threshold_snr = row['snr_db']
                    break

            if threshold_snr is not None:
                thresholds.append(threshold_snr)
                noise_names.append(noise_type)

        if thresholds:
            axes[1, 0].barh(noise_names, thresholds)
            axes[1, 0].set_xlabel('SNR Threshold (dB)')
            axes[1, 0].set_title('SNR Threshold for 10% Performance Drop')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Safety metrics under noise
        # Plot false negative rates for critical class
        for noise_type in ['gaussian', 'uniform', 'impulse']:
            if noise_type in results_df['noise_type'].unique():
                noise_df = results_df[results_df['noise_type'] == noise_type]
                fn_rates = []

                for _, row in noise_df.iterrows():
                    if 'safety_metrics' in row and isinstance(row['safety_metrics'], dict):
                        fn_rate = row['safety_metrics'].get('critical_fn_rate', 0)
                        fn_rates.append(fn_rate)
                    else:
                        fn_rates.append(0)

                if fn_rates and noise_df['snr_db'].notna().any():
                    axes[1, 1].plot(noise_df['snr_db'], fn_rates,
                                  marker='o', label=f'{noise_type} FN Rate')

        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('False Negative Rate')
        axes[1, 1].set_title('Safety-Critical Error Rate vs SNR')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'noise_robustness_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for standalone testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Test noise robustness of Fuzzy-XFD model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='./noise_test_results',
                       help='Output directory')
    parser.add_argument('--snr_range', nargs='+', type=float,
                       default=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
                       help='SNR values to test')

    args = parser.parse_args()

    # Initialize tester
    tester = NoiseRobustnessTester(args.model, args.config)

    # Note: You would need to load the test data loader here
    # test_loader = load_test_data(args.config)

    print("Noise robustness testing ready!")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"SNR range: {args.snr_range}")


if __name__ == "__main__":
    import sys
    main()