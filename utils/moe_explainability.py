"""
MoE Explainability Analysis Tools

This module provides comprehensive explainability analysis tools for the
Mixture of Experts model, including visualization of expert routing,
path analysis, and feature contribution analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import json


class MoEExplainabilityAnalyzer:
    """
    Comprehensive explainability analyzer for MoE models.

    Provides tools for:
    - Expert routing visualization
    - Path signature analysis
    - Expert contribution analysis
    - Feature importance analysis
    - Decision boundary exploration
    """

    def __init__(self, model, save_dir: str = "./moe_analysis"):
        """
        Initialize the analyzer.

        Args:
            model: MoE model instance
            save_dir: Directory to save analysis results
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme for different experts
        self.expert_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        # Store analysis results
        self.analysis_results = {}

    def analyze_expert_routing(self, data_loader: torch.utils.data.DataLoader,
                              num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze expert routing patterns.

        Args:
            data_loader: DataLoader containing test data
            num_samples: Number of samples to analyze

        Returns:
            Dictionary containing routing analysis results
        """
        self.model.eval()
        routing_weights_all = []
        routing_entropy_all = []
        dominant_experts = []
        sample_indices = []

        with torch.no_grad():
            for i, (signals, labels) in enumerate(data_loader):
                if i * len(signals) >= num_samples:
                    break

                signals = signals.float()
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)  # Add channel dimension if needed

                # Get model outputs with explanations
                logits, metadata = self.model(signals, return_explanations=True)
                routing_weights = metadata['routing_weights']
                routing_info = metadata['routing_info']

                routing_weights_all.append(routing_weights.cpu().numpy())
                routing_entropy_all.append(routing_info['entropy'].cpu().numpy())
                dominant_experts.append(routing_info['dominant_expert'].cpu().numpy())
                sample_indices.extend(range(i * len(signals), (i + 1) * len(signals)))

        # Combine all results
        routing_weights_all = np.vstack(routing_weights_all)
        routing_entropy_all = np.concatenate(routing_entropy_all)
        dominant_experts = np.concatenate(dominant_experts)

        # Analyze routing patterns
        expert_usage_stats = {}
        for expert_id in range(self.model.num_experts):
            expert_usage = np.mean(routing_weights_all[:, expert_id])
            expert_std = np.std(routing_weights_all[:, expert_id])
            dominant_count = np.sum(dominant_experts == expert_id)
            dominant_ratio = dominant_count / len(dominant_experts)

            expert_usage_stats[f'expert_{expert_id}'] = {
                'mean_weight': float(expert_usage),
                'std_weight': float(expert_std),
                'dominant_count': int(dominant_count),
                'dominant_ratio': float(dominant_ratio)
            }

        # Routing balance analysis
        routing_balance = 1.0 - np.std([stats['mean_weight'] for stats in expert_usage_stats.values()])
        avg_entropy = float(np.mean(routing_entropy_all))

        analysis = {
            'expert_usage_stats': expert_usage_stats,
            'routing_balance': routing_balance,
            'avg_routing_entropy': avg_entropy,
            'routing_weights_distribution': routing_weights_all.tolist(),
            'dominant_experts': dominant_experts.tolist(),
            'sample_indices': sample_indices
        }

        self.analysis_results['routing_analysis'] = analysis
        return analysis

    def visualize_expert_routing(self, routing_analysis: Dict[str, Any],
                               save_plot: bool = True) -> plt.Figure:
        """
        Visualize expert routing patterns.

        Args:
            routing_analysis: Results from analyze_expert_routing
            save_plot: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MoE Expert Routing Analysis', fontsize=16, fontweight='bold')

        # 1. Expert usage bar plot
        expert_ids = list(routing_analysis['expert_usage_stats'].keys())
        mean_weights = [routing_analysis['expert_usage_stats'][eid]['mean_weight'] for eid in expert_ids]
        std_weights = [routing_analysis['expert_usage_stats'][eid]['std_weight'] for eid in expert_ids]

        bars = axes[0, 0].bar(range(len(expert_ids)), mean_weights,
                             yerr=std_weights, capsize=5, color=self.expert_colors[:len(expert_ids)])
        axes[0, 0].set_xlabel('Expert ID')
        axes[0, 0].set_ylabel('Average Routing Weight')
        axes[0, 0].set_title('Expert Usage Distribution')
        axes[0, 0].set_xticks(range(len(expert_ids)))
        axes[0, 0].set_xticklabels([f'E{i+1}' for i in range(len(expert_ids))])

        # Add value labels on bars
        for bar, weight in zip(bars, mean_weights):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{weight:.3f}', ha='center', va='bottom')

        # 2. Dominant expert pie chart
        dominant_counts = [routing_analysis['expert_usage_stats'][eid]['dominant_count']
                          for eid in expert_ids]
        labels = [f'E{i+1} ({count})' for i, count in enumerate(dominant_counts)]

        axes[0, 1].pie(dominant_counts, labels=labels, colors=self.expert_colors[:len(expert_ids)],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Dominant Expert Distribution')

        # 3. Routing weights heatmap
        routing_weights = np.array(routing_analysis['routing_weights_distribution'])
        sample_subset = min(200, len(routing_weights))  # Limit for visualization
        routing_subset = routing_weights[:sample_subset]

        im = axes[1, 0].imshow(routing_subset.T, aspect='auto', cmap='YlOrRd',
                              interpolation='nearest')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Expert ID')
        axes[1, 0].set_title(f'Routing Weights Heatmap (First {sample_subset} samples)')
        axes[1, 0].set_yticks(range(len(expert_ids)))
        axes[1, 0].set_yticklabels([f'E{i+1}' for i in range(len(expert_ids))])
        plt.colorbar(im, ax=axes[1, 0])

        # 4. Routing entropy distribution
        routing_entropy_all = []
        for i in range(len(routing_weights)):
            weights = routing_weights[i]
            entropy = -np.sum(weights * np.log(weights + 1e-8))
            routing_entropy_all.append(entropy)

        axes[1, 1].hist(routing_entropy_all, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(routing_entropy_all), color='red', linestyle='--',
                          label=f'Mean: {np.mean(routing_entropy_all):.3f}')
        axes[1, 1].set_xlabel('Routing Entropy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Routing Entropy Distribution')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'expert_routing_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / 'expert_routing_analysis.pdf', bbox_inches='tight')

        return fig

    def analyze_path_signatures(self, data_loader: torch.utils.data.DataLoader,
                               num_samples: int = 500) -> Dict[str, Any]:
        """
        Analyze individual sample path signatures.

        Args:
            data_loader: DataLoader containing test data
            num_samples: Number of samples to analyze

        Returns:
            Dictionary containing path signature analysis
        """
        self.model.eval()
        path_signatures = []
        sample_labels = []

        with torch.no_grad():
            for i, (signals, labels) in enumerate(data_loader):
                if i * len(signals) >= num_samples:
                    break

                signals = signals.float()
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)

                # Get explanations for each sample
                logits, metadata = self.model(signals, return_explanations=True)
                explanations = metadata['explanations']

                for j in range(len(signals)):
                    signature = explanations['path_signatures'][j]
                    path_signatures.append(signature)
                    sample_labels.append(labels[j].item())

        # Analyze patterns
        dominant_experts = [sig['dominant_expert'] for sig in path_signatures]
        confidences = [sig['expert_confidence'] for sig in path_signatures]
        entropies = [sig['routing_entropy'] for sig in path_signatures]

        # Group by true label
        label_groups = {}
        for i, label in enumerate(sample_labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(path_signatures[i])

        # Analyze per-class patterns
        class_analysis = {}
        for label, signatures in label_groups.items():
            dominant_experts_class = [sig['dominant_expert'] for sig in signatures]
            confidences_class = [sig['expert_confidence'] for sig in signatures]

            # Most common expert for this class
            expert_counts = np.bincount(dominant_experts_class, minlength=self.model.num_experts)
            most_common_expert = np.argmax(expert_counts)

            class_analysis[label] = {
                'sample_count': len(signatures),
                'most_common_expert': int(most_common_expert),
                'expert_distribution': expert_counts.tolist(),
                'avg_confidence': float(np.mean(confidences_class)),
                'confidence_std': float(np.std(confidences_class))
            }

        analysis = {
            'path_signatures': path_signatures,
            'sample_labels': sample_labels,
            'dominant_experts': dominant_experts,
            'avg_confidence': float(np.mean(confidences)),
            'avg_entropy': float(np.mean(entropies)),
            'class_analysis': class_analysis,
            'signature_stats': {
                'num_samples': len(path_signatures),
                'num_classes': len(set(sample_labels))
            }
        }

        self.analysis_results['path_analysis'] = analysis
        return analysis

    def visualize_path_signatures(self, path_analysis: Dict[str, Any],
                                save_plot: bool = True) -> plt.Figure:
        """
        Visualize path signature analysis.

        Args:
            path_analysis: Results from analyze_path_signatures
            save_plot: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Path Signature Analysis', fontsize=16, fontweight='bold')

        # 1. Confidence distribution
        confidences = [sig['expert_confidence'] for sig in path_analysis['path_signatures']]
        axes[0, 0].hist(confidences, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--',
                          label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 0].set_xlabel('Expert Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Expert Confidences')
        axes[0, 0].legend()

        # 2. Class vs dominant expert heatmap
        class_analysis = path_analysis['class_analysis']
        classes = sorted(class_analysis.keys())
        experts = range(self.model.num_experts)

        # Create matrix of expert usage per class
        usage_matrix = np.zeros((len(classes), self.model.num_experts))
        for i, cls in enumerate(classes):
            usage_matrix[i] = class_analysis[cls]['expert_distribution']

        # Normalize per class
        usage_matrix_normalized = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-8)

        im = axes[0, 1].imshow(usage_matrix_normalized, aspect='auto', cmap='Blues',
                              interpolation='nearest')
        axes[0, 1].set_xlabel('Expert ID')
        axes[0, 1].set_ylabel('True Class')
        axes[0, 1].set_title('Expert Usage by True Class')
        axes[0, 1].set_xticks(range(self.model.num_experts))
        axes[0, 1].set_xticklabels([f'E{i+1}' for i in range(self.model.num_experts)])
        axes[0, 1].set_yticks(range(len(classes)))
        axes[0, 1].set_yticklabels([f'Class {c}' for c in classes])
        plt.colorbar(im, ax=axes[0, 1])

        # Add text annotations
        for i in range(len(classes)):
            for j in range(self.model.num_experts):
                text = axes[0, 1].text(j, i, f'{usage_matrix_normalized[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

        # 3. Confidence vs Entropy scatter
        entropies = [sig['routing_entropy'] for sig in path_analysis['path_signatures']]
        dominant_experts = [sig['dominant_expert'] for sig in path_analysis['path_signatures']]

        scatter = axes[1, 0].scatter(confidences, entropies, c=dominant_experts,
                                   cmap='viridis', alpha=0.6, s=30)
        axes[1, 0].set_xlabel('Expert Confidence')
        axes[1, 0].set_ylabel('Routing Entropy')
        axes[1, 0].set_title('Confidence vs Entropy by Expert')
        plt.colorbar(scatter, ax=axes[1, 0], label='Dominant Expert')

        # 4. Class-wise expert distribution
        expert_names = [f'E{i+1}' for i in range(self.model.num_experts)]
        x = np.arange(len(classes))
        width = 0.8 / self.model.num_experts

        for i, expert in enumerate(experts):
            expert_usage = [class_analysis[cls]['expert_distribution'][expert] /
                           class_analysis[cls]['sample_count'] for cls in classes]
            axes[1, 1].bar(x + i * width, expert_usage, width,
                          label=f'Expert {i+1}', color=self.expert_colors[i], alpha=0.7)

        axes[1, 1].set_xlabel('True Class')
        axes[1, 1].set_ylabel('Expert Usage Ratio')
        axes[1, 1].set_title('Class-wise Expert Usage')
        axes[1, 1].set_xticks(x + width * (self.model.num_experts - 1) / 2)
        axes[1, 1].set_xticklabels([f'Class {c}' for c in classes])
        axes[1, 1].legend()

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'path_signature_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / 'path_signature_analysis.pdf', bbox_inches='tight')

        return fig

    def analyze_feature_importance(self, data_loader: torch.utils.data.DataLoader,
                                  num_samples: int = 500) -> Dict[str, Any]:
        """
        Analyze feature importance for routing decisions.

        Args:
            data_loader: DataLoader containing test data
            num_samples: Number of samples to analyze

        Returns:
            Dictionary containing feature importance analysis
        """
        self.model.eval()
        feature_importance_scores = []
        routing_weights_all = []

        with torch.no_grad():
            for i, (signals, labels) in enumerate(data_loader):
                if i * len(signals) >= num_samples:
                    break

                signals = signals.float()
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)

                # Get routing information
                routing_weights, routing_features, routing_info = self.model.router(signals)
                routing_weights_all.append(routing_weights.cpu().numpy())

                # Compute feature importance using gradient-free method
                # Analyze correlation between features and routing decisions
                features_np = routing_features.cpu().numpy()

                for expert_id in range(self.model.num_experts):
                    expert_weights = routing_weights[:, expert_id].cpu().numpy()

                    # Compute correlation between each feature and expert weights
                    feature_correlations = []
                    for feature_idx in range(features_np.shape[1]):
                        correlation = np.corrcoef(features_np[:, feature_idx], expert_weights)[0, 1]
                        if not np.isnan(correlation):
                            feature_correlations.append(abs(correlation))
                        else:
                            feature_correlations.append(0.0)

                    feature_importance_scores.append(feature_correlations)

        # Average importance across samples
        feature_importance_scores = np.array(feature_importance_scores)
        avg_importance = np.mean(feature_importance_scores, axis=1)  # Per expert

        # Feature names (assuming 15 statistical features)
        feature_names = [
            'Mean', 'Std', 'Var', 'RMS', 'Peak', 'Peak2Peak', 'AbsMean',
            'Skewness', 'Kurtosis', 'ImpulseFactor', 'ClearanceFactor',
            'ShapeFactor', 'CrestFactor', 'MarginFactor', 'Energy'
        ][:feature_importance_scores.shape[-1]]

        analysis = {
            'feature_importance_per_expert': avg_importance.tolist(),
            'feature_names': feature_names,
            'importance_matrix': feature_importance_scores.tolist(),
            'most_important_features': {}
        }

        # Find most important features for each expert
        for expert_id in range(self.model.num_experts):
            expert_importance = avg_importance[expert_id]
            top_features_idx = np.argsort(expert_importance)[-5:][::-1]
            analysis['most_important_features'][f'expert_{expert_id}'] = [
                {
                    'feature': feature_names[idx],
                    'importance': float(expert_importance[idx]),
                    'rank': rank + 1
                }
                for rank, idx in enumerate(top_features_idx)
            ]

        self.analysis_results['feature_importance'] = analysis
        return analysis

    def visualize_feature_importance(self, feature_analysis: Dict[str, Any],
                                   save_plot: bool = True) -> plt.Figure:
        """
        Visualize feature importance analysis.

        Args:
            feature_analysis: Results from analyze_feature_importance
            save_plot: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

        # 1. Feature importance per expert heatmap
        importance_matrix = np.array(feature_analysis['importance_matrix'])
        feature_names = feature_analysis['feature_names']

        im = axes[0, 0].imshow(importance_matrix, aspect='auto', cmap='YlOrRd',
                              interpolation='nearest')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Expert ID')
        axes[0, 0].set_title('Feature Importance per Expert')
        axes[0, 0].set_yticks(range(self.model.num_experts))
        axes[0, 0].set_yticklabels([f'E{i+1}' for i in range(self.model.num_experts)])
        axes[0, 0].set_xticks(range(len(feature_names)))
        axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        plt.colorbar(im, ax=axes[0, 0])

        # 2. Top features per expert
        expert_ids = list(feature_analysis['most_important_features'].keys())
        top_features = []

        for expert_id in expert_ids:
            expert_features = feature_analysis['most_important_features'][expert_id]
            top_features.append([f['feature'] for f in expert_features[:3]])

        # Create table
        axes[0, 1].axis('tight')
        axes[0, 1].axis('off')
        table_data = []
        for i, expert_id in enumerate(expert_ids):
            row = [f'Expert {i+1}'] + top_features[i]
            table_data.append(row)

        table = axes[0, 1].table(cellText=table_data,
                                colLabels=['Expert', 'Top Feature', '2nd Feature', '3rd Feature'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[0, 1].set_title('Most Important Features per Expert', pad=20)

        # 3. Feature importance distribution
        all_importance = importance_matrix.flatten()
        axes[1, 0].hist(all_importance, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Feature Importance Scores')

        # 4. Average importance across experts
        avg_importance_per_feature = np.mean(importance_matrix, axis=0)
        sorted_idx = np.argsort(avg_importance_per_feature)

        axes[1, 1].barh(range(len(feature_names)),
                       avg_importance_per_feature[sorted_idx],
                       color='lightblue', edgecolor='black')
        axes[1, 1].set_xlabel('Average Importance')
        axes[1, 1].set_ylabel('Features')
        axes[1, 1].set_title('Average Feature Importance Across All Experts')
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels([feature_names[i] for i in sorted_idx])

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.save_dir / 'feature_importance_analysis.pdf', bbox_inches='tight')

        return fig

    def generate_comprehensive_report(self, data_loader: torch.utils.data.DataLoader,
                                    num_samples: int = 1000) -> str:
        """
        Generate a comprehensive explainability report.

        Args:
            data_loader: DataLoader containing test data
            num_samples: Number of samples to analyze

        Returns:
            Path to the generated report
        """
        print("Generating MoE Explainability Report...")

        # Run all analyses
        print("1. Analyzing expert routing patterns...")
        routing_analysis = self.analyze_expert_routing(data_loader, num_samples)

        print("2. Analyzing path signatures...")
        path_analysis = self.analyze_path_signatures(data_loader, num_samples)

        print("3. Analyzing feature importance...")
        feature_analysis = self.analyze_feature_importance(data_loader, num_samples)

        print("4. Generating visualizations...")
        self.visualize_expert_routing(routing_analysis)
        self.visualize_path_signatures(path_analysis)
        self.visualize_feature_importance(feature_analysis)

        # Generate text report
        report = {
            'model_info': self.model.get_model_info(),
            'analysis_summary': {
                'routing_analysis': {
                    'routing_balance': routing_analysis['routing_balance'],
                    'avg_routing_entropy': routing_analysis['avg_routing_entropy'],
                    'most_used_expert': max(routing_analysis['expert_usage_stats'].items(),
                                          key=lambda x: x[1]['dominant_ratio'])[0]
                },
                'path_analysis': {
                    'num_samples_analyzed': path_analysis['signature_stats']['num_samples'],
                    'num_classes': path_analysis['signature_stats']['num_classes'],
                    'avg_confidence': path_analysis['avg_confidence'],
                    'avg_entropy': path_analysis['avg_entropy']
                },
                'feature_analysis': {
                    'num_features': len(feature_analysis['feature_names']),
                    'most_important_overall': max([
                        max(features, key=lambda x: x['importance'])
                        for features in feature_analysis['most_important_features'].values()
                    ], key=lambda x: x['importance'])['feature']
                }
            },
            'detailed_results': {
                'expert_usage': routing_analysis['expert_usage_stats'],
                'class_patterns': path_analysis['class_analysis'],
                'feature_importance': feature_analysis['most_important_features']
            }
        }

        # Save report
        report_path = self.save_dir / 'comprehensive_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate text summary
        summary_path = self.save_dir / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write("=== MoE Model Explainability Report ===\n\n")

            f.write(f"Model: {report['model_info']['model_name']} v{report['model_info']['version']}\n")
            f.write(f"Number of experts: {report['model_info']['num_experts']}\n")
            f.write(f"Number of classes: {report['model_info']['num_classes']}\n\n")

            f.write("--- Routing Analysis ---\n")
            f.write(f"Routing balance score: {report['analysis_summary']['routing_analysis']['routing_balance']:.3f}\n")
            f.write(f"Average routing entropy: {report['analysis_summary']['routing_analysis']['avg_routing_entropy']:.3f}\n")
            f.write(f"Most used expert: {report['analysis_summary']['routing_analysis']['most_used_expert']}\n\n")

            f.write("--- Path Analysis ---\n")
            f.write(f"Samples analyzed: {report['analysis_summary']['path_analysis']['num_samples_analyzed']}\n")
            f.write(f"Classes analyzed: {report['analysis_summary']['path_analysis']['num_classes']}\n")
            f.write(f"Average confidence: {report['analysis_summary']['path_analysis']['avg_confidence']:.3f}\n\n")

            f.write("--- Feature Importance ---\n")
            f.write(f"Total features: {report['analysis_summary']['feature_analysis']['num_features']}\n")
            f.write(f"Most important feature: {report['analysis_summary']['feature_analysis']['most_important_overall']}\n\n")

            f.write("--- Expert Details ---\n")
            for expert_info in report['model_info']['experts']:
                f.write(f"Expert {expert_info['expert_name']}:\n")
                f.write(f"  - Target faults: {', '.join(expert_info['target_faults'])}\n")
                f.write(f"  - Physical mechanism: {expert_info['physical_mechanism']}\n")
                f.write(f"  - Strengths: {', '.join(expert_info['strengths'])}\n\n")

        print(f"Report generated successfully!")
        print(f"Files saved to: {self.save_dir}")
        print(f"- Comprehensive JSON report: {report_path}")
        print(f"- Text summary: {summary_path}")
        print(f"- Visualizations: *.png and *.pdf files")

        return str(summary_path)

    def compare_with_baseline(self, baseline_model, data_loader: torch.utils.data.DataLoader,
                            num_samples: int = 500) -> Dict[str, Any]:
        """
        Compare MoE model explainability with a baseline model.

        Args:
            baseline_model: Baseline model to compare with
            data_loader: DataLoader containing test data
            num_samples: Number of samples to analyze

        Returns:
            Dictionary containing comparison results
        """
        # Analyze MoE model
        moe_analysis = self.analyze_expert_routing(data_loader, num_samples)

        # Analyze baseline model (assuming it has explainability methods)
        baseline_explanations = []
        moe_explanations = []

        self.model.eval()
        baseline_model.eval()

        with torch.no_grad():
            for i, (signals, labels) in enumerate(data_loader):
                if i * len(signals) >= num_samples:
                    break

                signals = signals.float()
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)

                # MoE explanations
                _, moe_metadata = self.model(signals, return_explanations=True)
                moe_explanations.append(moe_metadata['explanations'])

                # Baseline explanations (if available)
                if hasattr(baseline_model, 'get_explanations'):
                    baseline_explanations.append(baseline_model.get_explanations(signals))

        # Calculate comparison metrics
        comparison = {
            'moe_routing_entropy': moe_analysis['avg_routing_entropy'],
            'moe_routing_balance': moe_analysis['routing_balance'],
            'num_experts': self.model.num_experts,
            'explainable_features': self.model.get_model_info()['explainable_features'],
            'moe_advantages': [
                'Expert specialization provides clear decision paths',
                'Routing weights show model confidence distribution',
                'Multiple expert perspectives enable robust decisions',
                'Physical mechanisms provide domain understanding'
            ],
            'comparison_summary': f"MoE model provides {self.model.num_experts} specialized experts "
                                 f"with routing balance score of {moe_analysis['routing_balance']:.3f} "
                                 f"and average entropy of {moe_analysis['avg_routing_entropy']:.3f}."
        }

        return comparison