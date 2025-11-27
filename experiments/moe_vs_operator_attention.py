"""
MoE vs Operator Attention Comparison Experiments

This module implements comprehensive comparison experiments between
the Mixture of Experts model and Operator Attention approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from model.MoE import MoEModel
from model.operator_attention import TSPN_OperatorAttention
from model.TSPN import TSPN
from model.TSPN_explainable import TSPNExplainable

# Import explainability tools
from utils.moe_explainability import MoEExplainabilityAnalyzer


class ModelComparisonFramework:
    """
    Framework for comparing different explainable fault diagnosis models.

    Supports comparison of:
    - MoE (Mixture of Experts)
    - Operator Attention
    - Traditional TSPN
    - Explainable TSPN
    """

    def __init__(self, config_path: str, save_dir: str = "./comparison_results"):
        """
        Initialize the comparison framework.

        Args:
            config_path: Path to configuration file
            save_dir: Directory to save comparison results
        """
        self.config = self._load_config(config_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.models = {}
        self.model_results = {}

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def initialize_models(self) -> Dict[str, nn.Module]:
        """
        Initialize all models for comparison.

        Returns:
            Dictionary of initialized models
        """
        args = self.config.get('args', {})
        num_classes = args.get('num_classes', 10)
        feature_dim = args.get('feature_dim', 64)

        models = {}

        # 1. MoE Model
        print("Initializing MoE model...")
        models['moe'] = MoEModel(
            num_classes=num_classes,
            feature_dim=feature_dim,
            num_experts=3,
            routing_temperature=1.0,
            use_load_balance=True
        ).to(self.device)

        # 2. Operator Attention Model
        print("Initializing Operator Attention model...")
        models['operator_attention'] = TSPN_OperatorAttention(
            in_dim=args.get('in_dim', 4096),
            out_dim=args.get('out_dim', 64),
            num_classes=num_classes,
            layer1=['I', 'WF'],  # Signal processing layers
            layer2=['HT', 'I'],
            layer3=['FFT', 'I'],
            layer4=['I', 'WF']
        ).to(self.device)

        # 3. Traditional TSPN
        print("Initializing TSPN model...")
        models['tspn'] = TSPN(
            in_dim=args.get('in_dim', 4096),
            out_dim=args.get('out_dim', 64),
            num_classes=num_classes,
            layer1=['I', 'WF'],
            layer2=['HT', 'I'],
            layer3=['FFT', 'I'],
            layer4=['I', 'WF']
        ).to(self.device)

        # 4. Explainable TSPN
        print("Initializing Explainable TSPN model...")
        models['tspn_exp'] = TSPNExplainable(
            in_dim=args.get('in_dim', 4096),
            out_dim=args.get('out_dim', 64),
            num_classes=num_classes,
            layer1=['I', 'WF'],
            layer2=['HT', 'I'],
            layer3=['FFT', 'I'],
            layer4=['I', 'WF']
        ).to(self.device)

        self.models = models
        return models

    def train_model(self, model_name: str, train_loader: DataLoader,
                   val_loader: DataLoader, epochs: int = 50) -> Dict[str, Any]:
        """
        Train a single model and record metrics.

        Args:
            model_name: Name of the model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs

        Returns:
            Dictionary containing training results
        """
        print(f"\n=== Training {model_name.upper()} Model ===")

        model = self.models[model_name]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training metrics
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epoch_times = []

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (signals, labels) in enumerate(train_loader):
                signals, labels = signals.float().to(self.device), labels.to(self.device)

                # Ensure correct input shape
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if signals.shape[-1] == 1:
                    signals = signals.squeeze(-1)

                optimizer.zero_grad()

                # Forward pass
                if hasattr(model, 'forward') and 'return_explanations' in model.forward.__code__.co_varnames:
                    outputs, _ = model(signals, return_explanations=False)
                else:
                    outputs = model(signals)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.float().to(self.device), labels.to(self.device)

                    if len(signals.shape) == 2:
                        signals = signals.unsqueeze(-1)
                    if signals.shape[-1] == 1:
                        signals = signals.squeeze(-1)

                    if hasattr(model, 'forward') and 'return_explanations' in model.forward.__code__.co_varnames:
                        outputs, _ = model(signals, return_explanations=False)
                    else:
                        outputs = model(signals)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            epoch_time = time.time() - epoch_start_time

            # Record metrics
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            epoch_times.append(epoch_time)

            # Learning rate scheduling
            scheduler.step(val_loss / len(val_loader))

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), self.save_dir / f'{model_name}_best.pth')

            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Time: {epoch_time:.2f}s')

        # Compile results
        results = {
            'model_name': model_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'epoch_times': epoch_times,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_training_time': sum(epoch_times),
            'avg_epoch_time': sum(epoch_times) / len(epoch_times)
        }

        self.model_results[model_name] = results
        return results

    def evaluate_model(self, model_name: str, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model_name: Name of the model to evaluate
            test_loader: Test data loader

        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n=== Evaluating {model_name.upper()} Model ===")

        model = self.models[model_name]
        model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        all_predictions = []
        all_labels = []
        all_explanations = []

        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.float().to(self.device), labels.to(self.device)

                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if signals.shape[-1] == 1:
                    signals = signals.squeeze(-1)

                # Forward pass with explanations for explainable models
                if hasattr(model, 'forward') and 'return_explanations' in model.forward.__code__.co_varnames:
                    outputs, metadata = model(signals, return_explanations=True)
                    if 'explanations' in metadata:
                        all_explanations.append(metadata['explanations'])
                else:
                    outputs = model(signals)

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_acc = 100.0 * test_correct / test_total
        test_loss_avg = test_loss / len(test_loader)

        # Calculate per-class accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss_avg,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'num_explanations': len(all_explanations),
            'has_explanations': len(all_explanations) > 0
        }

        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Test Loss: {test_loss_avg:.4f}')

        return results

    def compare_explainability(self, test_loader: DataLoader, num_samples: int = 200) -> Dict[str, Any]:
        """
        Compare explainability features of different models.

        Args:
            test_loader: Test data loader
            num_samples: Number of samples to analyze

        Returns:
            Dictionary containing explainability comparison
        """
        print("\n=== Comparing Explainability Features ===")

        explainability_comparison = {}

        # 1. MoE Explainability Analysis
        print("Analyzing MoE explainability...")
        moe_analyzer = MoEExplainabilityAnalyzer(self.models['moe'],
                                                save_dir=str(self.save_dir / 'moe_analysis'))
        moe_exp_analysis = moe_analyzer.analyze_expert_routing(test_loader, num_samples)
        moe_path_analysis = moe_analyzer.analyze_path_signatures(test_loader, num_samples)
        moe_feature_analysis = moe_analyzer.analyze_feature_importance(test_loader, num_samples)

        explainability_comparison['moe'] = {
            'expert_routing_balance': moe_exp_analysis['routing_balance'],
            'avg_routing_entropy': moe_exp_analysis['avg_routing_entropy'],
            'num_experts': self.models['moe'].num_experts,
            'explanation_types': ['expert_routing', 'path_signatures', 'feature_importance'],
            'physical_interpretability': True,
            'decision_transparency': 'expert_selection_weights'
        }

        # 2. Operator Attention Explainability
        print("Analyzing Operator Attention explainability...")
        opatt_model = self.models['operator_attention']
        explainability_comparison['operator_attention'] = {
            'attention_mechanism': True,
            'signal_transparency': True,
            'explanation_types': ['attention_weights', 'signal_processing_path'],
            'physical_interpretability': True,
            'decision_transparency': 'operator_attention_weights'
        }

        # Analyze attention weights (if available)
        with torch.no_grad():
            for i, (signals, labels) in enumerate(test_loader):
                if i >= num_samples // len(signals):
                    break

                signals = signals.float().to(self.device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if signals.shape[-1] == 1:
                    signals = signals.squeeze(-1)

                if hasattr(opatt_model, 'get_attention_maps'):
                    attention_maps = opatt_model.get_attention_maps(signals)
                    explainability_comparison['operator_attention']['attention_analyzed'] = True
                    break

        # 3. Traditional TSPN (baseline)
        explainability_comparison['tspn'] = {
            'attention_mechanism': False,
            'signal_transparency': True,
            'explanation_types': ['signal_processing_layers'],
            'physical_interpretability': False,
            'decision_transparency': 'limited'
        }

        # 4. Explainable TSPN
        explainability_comparison['tspn_exp'] = {
            'attention_mechanism': False,
            'signal_transparency': True,
            'explanation_types': ['signal_processing_path', 'feature_importance'],
            'physical_interpretability': True,
            'decision_transparency': 'enhanced'
        }

        return explainability_comparison

    def run_comprehensive_comparison(self, train_loader: DataLoader, val_loader: DataLoader,
                                   test_loader: DataLoader, epochs: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all models.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of training epochs

        Returns:
            Dictionary containing all comparison results
        """
        print("Starting Comprehensive Model Comparison...")
        print(f"Device: {self.device}")
        print(f"Models to compare: {list(self.models.keys())}")

        comparison_results = {
            'training_results': {},
            'evaluation_results': {},
            'explainability_comparison': {},
            'summary_metrics': {}
        }

        # Train all models
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"TRAINING {model_name.upper()}")
            print(f"{'='*50}")

            training_results = self.train_model(model_name, train_loader, val_loader, epochs)
            comparison_results['training_results'][model_name] = training_results

            # Evaluate trained model
            evaluation_results = self.evaluate_model(model_name, test_loader)
            comparison_results['evaluation_results'][model_name] = evaluation_results

        # Compare explainability
        explainability_results = self.compare_explainability(test_loader)
        comparison_results['explainability_comparison'] = explainability_results

        # Generate summary metrics
        comparison_results['summary_metrics'] = self._generate_summary_metrics(
            comparison_results['training_results'],
            comparison_results['evaluation_results'],
            comparison_results['explainability_comparison']
        )

        # Save results
        self._save_comparison_results(comparison_results)

        # Generate comparison plots
        self._generate_comparison_plots(comparison_results)

        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")

        for model_name in comparison_results['summary_metrics']['ranking']:
            metrics = comparison_results['summary_metrics']['model_rankings'][model_name]
            print(f"\n{model_name.upper()}:")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%")
            print(f"  Training Time: {metrics['total_training_time']:.1f}s")
            print(f"  Explainability Score: {metrics['explainability_score']:.2f}")
            print(f"  Overall Score: {metrics['overall_score']:.3f}")

        return comparison_results

    def _generate_summary_metrics(self, training_results: Dict, evaluation_results: Dict,
                                 explainability_comparison: Dict) -> Dict[str, Any]:
        """Generate summary metrics for model comparison."""
        model_rankings = {}

        for model_name in training_results.keys():
            train_metrics = training_results[model_name]
            eval_metrics = evaluation_results[model_name]
            exp_metrics = explainability_comparison[model_name]

            # Calculate explainability score (0-1 scale)
            exp_score = 0.0
            if exp_metrics.get('physical_interpretability', False):
                exp_score += 0.3
            if exp_metrics.get('attention_mechanism', False):
                exp_score += 0.2
            if len(exp_metrics.get('explanation_types', [])) > 1:
                exp_score += 0.3
            if exp_metrics.get('decision_transparency') in ['expert_selection_weights', 'operator_attention_weights']:
                exp_score += 0.2

            # Calculate overall score (weighted combination)
            # Normalize metrics to 0-1 scale
            acc_normalized = eval_metrics['test_accuracy'] / 100.0
            time_normalized = 1.0 - (train_metrics['total_training_time'] / max([r['total_training_time'] for r in training_results.values()]))

            overall_score = 0.5 * acc_normalized + 0.3 * exp_score + 0.2 * time_normalized

            model_rankings[model_name] = {
                'test_accuracy': eval_metrics['test_accuracy'],
                'total_training_time': train_metrics['total_training_time'],
                'explainability_score': exp_score,
                'overall_score': overall_score,
                'best_val_acc': train_metrics['best_val_acc'],
                'training_efficiency': train_metrics['avg_epoch_time']
            }

        # Rank models by overall score
        ranking = sorted(model_rankings.keys(), key=lambda x: model_rankings[x]['overall_score'], reverse=True)

        return {
            'model_rankings': model_rankings,
            'ranking': ranking,
            'best_accuracy_model': max(model_rankings.keys(), key=lambda x: model_rankings[x]['test_accuracy']),
            'best_explainability_model': max(model_rankings.keys(), key=lambda x: model_rankings[x]['explainability_score']),
            'fastest_training_model': min(model_rankings.keys(), key=lambda x: model_rankings[x]['total_training_time'])
        }

    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results to files."""
        # Save as JSON
        results_path = self.save_dir / 'comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary as text
        summary_path = self.save_dir / 'comparison_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== Model Comparison Summary ===\n\n")

            for model_name, metrics in results['summary_metrics']['model_rankings'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%\n")
                f.write(f"  Training Time: {metrics['total_training_time']:.1f}s\n")
                f.write(f"  Explainability Score: {metrics['explainability_score']:.2f}\n")
                f.write(f"  Overall Score: {metrics['overall_score']:.3f}\n\n")

            f.write(f" Rankings:\n")
            for i, model in enumerate(results['summary_metrics']['ranking'], 1):
                f.write(f"  {i}. {model.upper()}\n")

        print(f"Comparison results saved to {self.save_dir}")

    def _generate_comparison_plots(self, results: Dict[str, Any]):
        """Generate comparison visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')

        model_names = list(results['training_results'].keys())
        metrics = results['summary_metrics']['model_rankings']

        # 1. Test Accuracy Comparison
        accuracies = [metrics[name]['test_accuracy'] for name in model_names]
        bars = axes[0, 0].bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_ylabel('Test Accuracy (%)')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylim(0, 100)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{acc:.1f}%', ha='center', va='bottom')

        # 2. Training Time Comparison
        training_times = [metrics[name]['total_training_time'] for name in model_names]
        bars = axes[0, 1].bar(model_names, training_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time Comparison')

        # Add value labels
        for bar, time in zip(bars, training_times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                           f'{time:.1f}s', ha='center', va='bottom')

        # 3. Explainability Score Comparison
        exp_scores = [metrics[name]['explainability_score'] for name in model_names]
        bars = axes[1, 0].bar(model_names, exp_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 0].set_ylabel('Explainability Score')
        axes[1, 0].set_title('Explainability Score Comparison')
        axes[1, 0].set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars, exp_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{score:.2f}', ha='center', va='bottom')

        # 4. Overall Score Radar Chart
        categories = ['Accuracy', 'Explainability', 'Efficiency']

        # Normalize each metric to 0-1 scale for radar chart
        acc_normalized = np.array([metrics[name]['test_accuracy'] for name in model_names]) / 100.0
        exp_normalized = np.array([metrics[name]['explainability_score'] for name in model_names])
        eff_normalized = 1.0 - (np.array([metrics[name]['training_efficiency'] for name in model_names]) /
                                max([metrics[name]['training_efficiency'] for name in model_names]))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for i, model_name in enumerate(model_names):
            values = [acc_normalized[i], exp_normalized[i], eff_normalized[i]]
            values += values[:1]  # Complete the circle

            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name.upper(), color=colors[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Overall Performance Comparison')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(self.save_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'comparison_plots.pdf', bbox_inches='tight')
        plt.close()

        print(f"Comparison plots saved to {self.save_dir}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            results: Comparison results

        Returns:
            Path to the generated report
        """
        report_path = self.save_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MIXTURE OF EXPERTS VS OPERATOR ATTENTION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            summary = results['summary_metrics']
            f.write(f"Best Overall Model: {summary['ranking'][0].upper()}\n")
            f.write(f"Best Accuracy: {summary['best_accuracy_model'].upper()}\n")
            f.write(f"Most Explainable: {summary['best_explainability_model'].upper()}\n")
            f.write(f"Fastest Training: {summary['fastest_training_model'].upper()}\n\n")

            # Detailed Results
            f.write("DETAILED RESULTS\n")
            f.write("-"*40 + "\n")

            for model_name in summary['ranking']:
                metrics = summary['model_rankings'][model_name]
                f.write(f"\n{model_name.upper()} MODEL:\n")
                f.write(f"  Test Accuracy: {metrics['test_accuracy']:.2f}%\n")
                f.write(f"  Best Validation Accuracy: {metrics['best_val_acc']:.2f}%\n")
                f.write(f"  Total Training Time: {metrics['total_training_time']:.1f} seconds\n")
                f.write(f"  Average Epoch Time: {metrics['training_efficiency']:.1f} seconds\n")
                f.write(f"  Explainability Score: {metrics['explainability_score']:.2f}/1.0\n")
                f.write(f"  Overall Score: {metrics['overall_score']:.3f}\n")

                # Add explainability details
                exp_info = results['explainability_comparison'][model_name]
                f.write(f"  Explainability Features:\n")
                for feature, value in exp_info.items():
                    if feature != 'explanation_types':
                        f.write(f"    {feature}: {value}\n")
                f.write(f"    Explanation Types: {', '.join(exp_info.get('explanation_types', []))}\n")

            # Conclusions
            f.write(f"\n\nCONCLUSIONS\n")
            f.write("-"*40 + "\n")
            f.write("1. Model Performance:\n")
            best_acc_model = summary['best_accuracy_model']
            best_acc = summary['model_rankings'][best_acc_model]['test_accuracy']
            f.write(f"   - {best_acc_model.upper()} achieved highest accuracy: {best_acc:.2f}%\n")

            f.write("\n2. Explainability:\n")
            best_exp_model = summary['best_explainability_model']
            best_exp_score = summary['model_rankings'][best_exp_model]['explainability_score']
            f.write(f"   - {best_exp_model.upper()} has best explainability: {best_exp_score:.2f}/1.0\n")

            f.write("\n3. Efficiency:\n")
            fastest_model = summary['fastest_training_model']
            fastest_time = summary['model_rankings'][fastest_model]['total_training_time']
            f.write(f"   - {fastest_model.upper()} trains fastest: {fastest_time:.1f}s\n")

            f.write("\n4. Recommendations:\n")
            if summary['ranking'][0] == 'moe':
                f.write("   - MoE model provides the best balance of accuracy and explainability\n")
                f.write("   - Recommended for applications requiring transparent decision making\n")
            elif summary['ranking'][0] == 'operator_attention':
                f.write("   - Operator Attention excels in performance with good interpretability\n")
                f.write("   - Recommended for high-performance applications\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")

        print(f"Comprehensive report generated: {report_path}")
        return str(report_path)