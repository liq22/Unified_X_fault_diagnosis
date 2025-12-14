"""
Visualization suite for Fuzzy-XFD P0 validation metrics
Creates comprehensive plots for multi-seed validation, noise robustness, and safety analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class FuzzyXFDVisualizer:
    """
    Comprehensive visualization toolkit for Fuzzy-XFD metrics
    """

    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_multi_seed_performance(self, results_df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization for multi-seed validation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Accuracy distribution across seeds
        ax = axes[0, 0]
        sns.histplot(data=results_df, x='test_accuracy', bins=10, kde=True, ax=ax)
        ax.axvline(x=results_df['test_accuracy'].mean(), color='red', linestyle='--',
                  label=f'Mean: {results_df["test_accuracy"].mean():.4f}')
        ax.axvline(x=0.707, color='green', linestyle='-',
                  label=f'Target: 70.7%')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Count')
        ax.set_title('Accuracy Distribution Across Seeds')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Seed-wise performance with confidence intervals
        ax = axes[0, 1]
        seeds = results_df['seed']
        accuracies = results_df['test_accuracy']
        mean_acc = accuracies.mean()
        std_acc = accuracies.std()

        ax.plot(seeds, accuracies, 'o-', markersize=8, linewidth=2)
        ax.axhline(y=mean_acc, color='red', linestyle='--',
                  label=f'Mean: {mean_acc:.4f} ± {std_acc:.4f}')
        ax.fill_between(seeds, mean_acc - std_acc, mean_acc + std_acc,
                       alpha=0.2, color='red', label='±1 Std Dev')
        ax.set_xlabel('Random Seed')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Performance Per Seed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Execution time vs accuracy
        ax = axes[1, 0]
        scatter = ax.scatter(results_df['execution_time'], results_df['test_accuracy'],
                            s=100, alpha=0.7, c=results_df['seed'], cmap='viridis')
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Execution Time vs Accuracy')
        plt.colorbar(scatter, ax=ax, label='Seed')
        ax.grid(True, alpha=0.3)

        # Plot 4: Performance statistics summary
        ax = axes[1, 1]
        stats_text = f"""
        Multi-Seed Validation Summary
        ============================
        Number of Seeds: {len(results_df)}
        Mean Accuracy: {results_df['test_accuracy'].mean():.4f}
        Std Deviation: {results_df['test_accuracy'].std():.4f}
        Min Accuracy: {results_df['test_accuracy'].min():.4f}
        Max Accuracy: {results_df['test_accuracy'].max():.4f}
        95% Confidence Interval: ±{1.96 * results_df['test_accuracy'].std() / np.sqrt(len(results_df)):.4f}

        Breakthrough Status: {'✓ CONFIRMED' if results_df['test_accuracy'].mean() >= 0.70 else '✗ NOT REPRODUCED'}
        """
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.axis('off')

        plt.suptitle('Fuzzy-XFD Multi-Seed Validation Results', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrices(self, cm_data: Dict[str, np.ndarray],
                              class_names: List[str],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple confusion matrices with consistency analysis
        """
        num_seeds = len(cm_data)
        cols = min(3, num_seeds)
        rows = (num_seeds + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot individual confusion matrices
        for idx, (seed, cm) in enumerate(cm_data.items()):
            row = idx // cols
            col = idx % cols

            ax = axes[row, col] if rows > 1 else axes[col]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar_kws={'label': 'Count'})

            accuracy = np.trace(cm) / np.sum(cm)
            ax.set_title(f'Seed {seed}\nAccuracy: {accuracy:.4f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        # Hide unused subplots
        for idx in range(num_seeds, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.suptitle('Confusion Matrices Across Seeds', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_consistency_heatmap(self, cm_data: Dict[str, np.ndarray],
                               class_names: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap showing prediction consistency across seeds
        """
        # Calculate confusion matrix averages
        cms = list(cm_data.values())
        avg_cm = np.mean(cms, axis=0)
        std_cm = np.std(cms, axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Average confusion matrix
        ax = axes[0]
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Average Count'})
        ax.set_title('Average Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Standard deviation
        ax = axes[1]
        sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Std Dev'})
        ax.set_title('Prediction Variability (Std Dev)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Coefficient of variation
        ax = axes[2]
        cv_cm = np.where(avg_cm > 0, std_cm / avg_cm, 0)
        sns.heatmap(cv_cm, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'CV'})
        ax.set_title('Coefficient of Variation')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.suptitle('Prediction Consistency Analysis Across Seeds', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curves_with_confidence(self, roc_data: Dict,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves with confidence intervals from multiple seeds
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Colors for different classes
        colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data) - 1))

        # Plot each class
        for idx, (class_name, curves) in enumerate(roc_data.items()):
            if class_name == 'macro':
                continue

            # If we have multiple curves (from different seeds)
            if isinstance(curves, list):
                # Calculate mean and confidence intervals
                all_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(all_fpr)
                tprs = []

                for curve in curves:
                    # Interpolate to common FPR points
                    tpr_interp = np.interp(all_fpr, curve['fpr'], curve['tpr'])
                    tpr_interp[0] = 0
                    tprs.append(tpr_interp)

                tprs = np.array(tprs)
                mean_tpr = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)

                # Plot mean curve
                ax.plot(all_fpr, mean_tpr, color=colors[idx],
                       label=f'{class_name} (AUC = {np.mean([c["roc_auc"] for c in curves]):.3f})')

                # Plot confidence interval
                tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
                tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
                ax.fill_between(all_fpr, tpr_lower, tpr_upper,
                              color=colors[idx], alpha=0.2)
            else:
                # Single curve
                ax.plot(curves['fpr'], curves['tpr'], color=colors[idx],
                       label=f'{class_name} (AUC = {curves["roc_auc"]:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves with 95% Confidence Intervals', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_noise_robustness_summary(self, noise_results: pd.DataFrame,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive noise robustness visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Get clean accuracy for reference
        clean_acc = noise_results[noise_results['noise_type'] == 'clean']['accuracy'].iloc[0]

        # Plot 1: Accuracy vs SNR for all noise types
        ax = axes[0, 0]
        for noise_type in noise_results['noise_type'].unique():
            if noise_type == 'clean':
                continue
            noise_df = noise_results[noise_results['noise_type'] == noise_type]
            ax.plot(noise_df['snr_db'], noise_df['accuracy'],
                   marker='o', label=noise_type, linewidth=2)

        ax.axhline(y=clean_acc, color='black', linestyle='--',
                  label=f'Clean: {clean_acc:.3f}')
        ax.axhline(y=clean_acc * 0.9, color='red', linestyle=':',
                  label='90% Performance')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs SNR by Noise Type')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Performance degradation heatmap
        ax = axes[0, 1]
        pivot_data = noise_results.pivot_table(values='accuracy',
                                              index='noise_type',
                                              columns='snr_db')
        if 'clean' in pivot_data.index:
            pivot_data = pivot_data.drop('clean')

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_title('Accuracy Heatmap')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Noise Type')

        # Plot 3: Robustness score (area under curve)
        ax = axes[0, 2]
        robustness_scores = []
        noise_types = []

        for noise_type in noise_results['noise_type'].unique():
            if noise_type == 'clean':
                continue
            noise_df = noise_results[noise_results['noise_type'] == noise_type]
            # Simple robustness metric: average accuracy across SNR levels
            score = noise_df['accuracy'].mean()
            robustness_scores.append(score)
            noise_types.append(noise_type)

        bars = ax.bar(noise_types, robustness_scores)
        ax.axhline(y=clean_acc, color='black', linestyle='--', label='Clean Performance')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Noise Robustness Score')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Color bars based on performance
        for bar, score in zip(bars, robustness_scores):
            if score >= clean_acc * 0.9:
                bar.set_color('green')
            elif score >= clean_acc * 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Plot 4: Safety metrics under noise
        ax = axes[1, 0]
        for noise_type in noise_results['noise_type'].unique():
            if noise_type == 'clean':
                continue
            noise_df = noise_results[noise_results['noise_type'] == noise_type]
            if 'critical_fn_rate' in noise_df.columns:
                ax.plot(noise_df['snr_db'], noise_df['critical_fn_rate'],
                       marker='o', label=f'{noise_type} FN Rate')

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('False Negative Rate')
        ax.set_title('Safety-Critical Error Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Plot 5: Performance degradation curve
        ax = axes[1, 1]
        for noise_type in noise_results['noise_type'].unique():
            if noise_type == 'clean':
                continue
            noise_df = noise_results[noise_results['noise_type'] == noise_type]
            relative_acc = noise_df['accuracy'] / clean_acc
            ax.plot(noise_df['snr_db'], relative_acc * 100,
                   marker='o', label=noise_type)

        ax.axhline(y=90, color='red', linestyle=':', label='90% Threshold')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Relative Accuracy (%)')
        ax.set_title('Relative Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Summary statistics
        ax = axes[1, 2]
        summary_text = f"""
        Noise Robustness Summary
        ========================
        Clean Accuracy: {clean_acc:.3f}

        Performance at 0dB SNR:
        - Gaussian: {noise_results[noise_results['noise_type'] == 'gaussian']['accuracy'].min():.3f}
        - Uniform: {noise_results[noise_results['noise_type'] == 'uniform']['accuracy'].min():.3f}
        - Impulse: {noise_results[noise_results['noise_type'] == 'impulse']['accuracy'].min():.3f}

        Best Noise Type: {noise_types[np.argmax(robustness_scores)]}
        Robustness Score: {max(robustness_scores):.3f}
        """
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.axis('off')

        plt.suptitle('Fuzzy-XFD Noise Robustness Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_dashboard(self, multi_seed_results: pd.DataFrame,
                                   noise_results: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """
        Create interactive dashboard using Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Multi-Seed Accuracy Distribution',
                           'Noise Robustness Curves',
                           'Confusion Matrix Consistency',
                           'Safety Metrics'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )

        # Multi-seed accuracy histogram
        fig.add_trace(
            go.Histogram(x=multi_seed_results['test_accuracy'],
                        name='Accuracy Distribution',
                        nbinsx=10),
            row=1, col=1
        )

        # Noise robustness curves
        for noise_type in noise_results['noise_type'].unique():
            if noise_type != 'clean':
                noise_df = noise_results[noise_results['noise_type'] == noise_type]
                fig.add_trace(
                    go.Scatter(x=noise_df['snr_db'],
                              y=noise_df['accuracy'],
                              mode='lines+markers',
                              name=noise_type),
                    row=1, col=2
                )

        # Note: Add confusion matrix heatmap and safety metrics
        # (Would need to load confusion matrix data separately)

        fig.update_layout(
            title_text="Fuzzy-XFD Interactive Validation Dashboard",
            showlegend=True,
            height=800
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def generate_comprehensive_report(self, results_dir: str):
        """
        Generate all visualizations for a validation run
        """
        results_path = Path(results_dir)

        # Load multi-seed results
        multi_seed_file = results_path / 'results_table.csv'
        if multi_seed_file.exists():
            multi_seed_df = pd.read_csv(multi_seed_file)

            # Plot multi-seed performance
            fig = self.plot_multi_seed_performance(
                multi_seed_df,
                save_path=self.output_dir / 'multi_seed_performance.png'
            )

            # Load and plot confusion matrices
            cm_dir = results_path / 'confusion_matrices'
            if cm_dir.exists():
                cm_data = {}
                for cm_file in cm_dir.glob('*.npy'):
                    seed = int(cm_file.stem.split('_')[-1])
                    cm_data[seed] = np.load(cm_file)

                if cm_data:
                    # Get class names from config or use defaults
                    class_names = ['Healthy', 'Inner_Fault', 'Outer_Fault', 'Ball_Fault', 'Cage_Fault']

                    self.plot_confusion_matrices(
                        cm_data,
                        class_names,
                        save_path=self.output_dir / 'confusion_matrices.png'
                    )

                    self.plot_consistency_heatmap(
                        cm_data,
                        class_names,
                        save_path=self.output_dir / 'consistency_heatmap.png'
                    )

        # Load noise robustness results
        noise_file = results_path / 'noise_robustness_results.csv'
        if noise_file.exists():
            noise_df = pd.read_csv(noise_file)

            self.plot_noise_robustness_summary(
                noise_df,
                save_path=self.output_dir / 'noise_robustness_summary.png'
            )

        print(f"All visualizations saved to {self.output_dir}")


def main():
    """
    Example usage of the visualizer
    """
    import argparse

    parser = argparse.ArgumentParser(description='Generate visualizations for Fuzzy-XFD validation')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing validation results')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = FuzzyXFDVisualizer(args.output_dir)

    # Generate all plots
    visualizer.generate_comprehensive_report(args.results_dir)


if __name__ == "__main__":
    main()