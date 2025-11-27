#!/usr/bin/env python3
"""
MoE Experiments Runner

This script runs comprehensive experiments with the MoE model,
including training, evaluation, and explainability analysis.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.MoE import MoEModel
from model.MoE_OperatorAttention import MoEOperatorAttentionFusion
from utils.moe_explainability import MoEExplainabilityAnalyzer
from experiments.moe_vs_operator_attention import ModelComparisonFramework

# Data loading utilities (adapt these to your actual data loading)
def load_dataset(data_dir: str, dataset_task: str, batch_size: int = 32):
    """
    Load dataset for training/evaluation.
    This is a placeholder - implement according to your data format.
    """
    # TODO: Implement actual data loading
    print(f"Loading dataset from {data_dir}, task: {dataset_task}")

    # Dummy implementation - replace with actual data loading
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, signal_length=4096, num_classes=10):
            self.num_samples = num_samples
            self.signal_length = signal_length
            self.num_classes = num_classes

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate dummy signal data
            signal = torch.randn(self.signal_length)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return signal, label

    # Create dummy datasets
    train_dataset = DummyDataset(800, 4096, 10)
    val_dataset = DummyDataset(100, 4096, 10)
    test_dataset = DummyDataset(100, 4096, 10)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_moe_model(config: dict, save_dir: str = "./moe_results"):
    """
    Train the MoE model.
    """
    print("="*60)
    print("TRAINING MOE MODEL")
    print("="*60)

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    args = config['args']
    model = MoEModel(
        num_classes=args['num_classes'],
        feature_dim=args['feature_dim'],
        num_experts=args['num_experts'],
        routing_temperature=args['routing_temperature'],
        use_load_balance=args['use_load_balance'],
        dropout_rate=args['dropout_rate']
    ).to(device)

    print(f"Model initialized with {model.num_experts} experts")
    print(f"Expert types: {[expert.get_expert_info()['expert_name'] for expert in model.experts]}")

    # Load data
    train_loader, val_loader, test_loader = load_dataset(
        args['data_dir'], args['dataset_task'], args['batch_size']
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],
                                weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )

    # Training loop
    epochs = args['epochs']
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.float().to(device), labels.to(device)

            # Ensure correct input shape
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            if signals.shape[-1] == 1:
                signals = signals.squeeze(-1)

            optimizer.zero_grad()

            # Forward pass
            outputs, metadata = model(signals, return_explanations=True)
            loss = criterion(outputs, labels)

            # Add regularization losses if available
            if 'regularization_losses' in metadata:
                for reg_loss_name, reg_loss_value in metadata['regularization_losses'].items():
                    loss += reg_loss_value

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
                signals, labels = signals.float().to(device), labels.to(device)

                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if signals.shape[-1] == 1:
                    signals = signals.squeeze(-1)

                outputs, metadata = model(signals, return_explanations=True)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss / len(val_loader))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, save_dir / 'best_moe_model.pth')

        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

            # Log expert usage statistics
            model.eval()
            with torch.no_grad():
                for signals, _ in train_loader:
                    signals = signals.float().to(device)
                    if len(signals.shape) == 2:
                        signals = signals.unsqueeze(-1)
                    if signals.shape[-1] == 1:
                        signals = signals.squeeze(-1)

                    _, metadata = model(signals, return_explanations=True)
                    routing_weights = metadata['routing_weights']
                    avg_usage = torch.mean(routing_weights, dim=0)
                    print(f'  Expert usage: {avg_usage.cpu().numpy().tolist()}')
                    break

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.float().to(device), labels.to(device)

            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            if signals.shape[-1] == 1:
                signals = signals.squeeze(-1)

            outputs, metadata = model(signals, return_explanations=True)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }


def analyze_explainability(model, test_loader, save_dir: str = "./moe_analysis"):
    """
    Perform comprehensive explainability analysis.
    """
    print("\n" + "="*60)
    print("EXPLAINABILITY ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = MoEExplainabilityAnalyzer(model, save_dir)

    # Generate comprehensive report
    report_path = analyzer.generate_comprehensive_report(test_loader, num_samples=200)

    print(f"Explainability analysis completed!")
    print(f"Report saved to: {report_path}")

    return analyzer, report_path


def run_fusion_experiments(config: dict, save_dir: str = "./fusion_results"):
    """
    Run experiments with the MoE + Operator Attention fusion model.
    """
    print("\n" + "="*60)
    print("TRAINING MOE + OPERATOR ATTENTION FUSION MODEL")
    print("="*60)

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize fusion model
    args = config['args']
    model = MoEOperatorAttentionFusion(
        num_classes=args['num_classes'],
        feature_dim=args['feature_dim'],
        num_experts=args['num_experts'],
        routing_temperature=args['routing_temperature'],
        use_operator_attention=args['use_operator_attention'],
        dropout_rate=args['dropout_rate']
    ).to(device)

    print(f"Fusion model initialized")
    print(f"Number of experts: {model.num_experts}")
    print(f"Operator attention enabled: {model.use_operator_attention}")

    # Load data
    train_loader, val_loader, test_loader = load_dataset(
        args['data_dir'], args['dataset_task'], args['batch_size']
    )

    # Training setup (similar to MoE training)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],
                                weight_decay=args['weight_decay'])

    # Simplified training loop for fusion model
    epochs = min(args['epochs'], 50)  # Reduced for demo
    best_val_acc = 0.0

    print(f"Starting fusion model training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.float().to(device), labels.to(device)

            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            if signals.shape[-1] == 1:
                signals = signals.squeeze(-1)

            optimizer.zero_grad()

            outputs, metadata = model(signals, return_explanations=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.float().to(device), labels.to(device)

                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if signals.shape[-1] == 1:
                    signals = signals.squeeze(-1)

                outputs, _ = model(signals, return_explanations=True)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, save_dir / 'best_fusion_model.pth')

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    print(f"\nFusion model training completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.float().to(device), labels.to(device)

            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            if signals.shape[-1] == 1:
                signals = signals.squeeze(-1)

            outputs, _ = model(signals, return_explanations=True)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    print(f'Fusion Model Test Accuracy: {test_acc:.2f}%')

    return model, test_acc


def run_comparison_experiments(config: dict):
    """
    Run comprehensive comparison experiments.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)

    # Initialize comparison framework
    framework = ModelComparisonFramework(
        config_path="",  # Using passed config directly
        save_dir="./comparison_results"
    )

    # Override config
    framework.config = config

    # Initialize models
    models = framework.initialize_models()

    # Load data
    args = config['args']
    train_loader, val_loader, test_loader = load_dataset(
        args['data_dir'], args['dataset_task'], args['batch_size']
    )

    # Run comparison
    results = framework.run_comprehensive_comparison(
        train_loader, val_loader, test_loader, epochs=30  # Reduced for demo
    )

    # Generate report
    report_path = framework.generate_report(results)

    print(f"Comparison experiments completed!")
    print(f"Report saved to: {report_path}")

    return results, report_path


def main():
    parser = argparse.ArgumentParser(description='MoE Experiments Runner')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, choices=['moe', 'fusion', 'comparison', 'all'],
                       default='all', help='Type of experiment to run')
    parser.add_argument('--save_dir', type=str, default='./moe_experiments',
                       help='Directory to save results')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running experiments with config: {args.config}")
    print(f"Experiment type: {args.experiment}")
    print(f"Save directory: {args.save_dir}")

    # Create main save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project_name'],
            config=config,
            name=f"moe_experiments_{args.experiment}_{int(time.time())}"
        )

    try:
        if args.experiment in ['moe', 'all']:
            print("\n" + "="*80)
            print("RUNNING MOE EXPERIMENTS")
            print("="*80)

            moe_results = train_moe_model(config, save_dir / "moe")

            if config['explainability']['enabled']:
                _, _ = load_dataset(config['args']['data_dir'],
                                   config['args']['dataset_task'],
                                   config['args']['batch_size'])
                analyze_explainability(moe_results['model'], _,
                                      save_dir / "moe_analysis")

        if args.experiment in ['fusion', 'all']:
            print("\n" + "="*80)
            print("RUNNING FUSION EXPERIMENTS")
            print("="*80)

            fusion_model, fusion_acc = run_fusion_experiments(config, save_dir / "fusion")

        if args.experiment in ['comparison', 'all']:
            print("\n" + "="*80)
            print("RUNNING COMPARISON EXPERIMENTS")
            print("="*80)

            comparison_results, comparison_report = run_comparison_experiments(config)

        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {save_dir}")

    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if config['logging']['use_wandb']:
            wandb.finish()

    return 0


if __name__ == "__main__":
    exit(main())