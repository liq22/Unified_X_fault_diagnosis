#!/usr/bin/env python3
"""
Main script for training 1D-2D Fusion models

This script provides a unified interface for training 1D-2D fusion models
with the fault diagnosis framework, supporting both early and aligned fusion
strategies.
"""

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer.fusion_trainer import create_fusion_trainer
from data.data_provider import get_data
from explainability.core.unified_explainer import UnifiedExplainer
from explainability.visualization.visualizer import create_visualization_report


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train 1D-2D Fusion Model')

    # Configuration
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--config_override', type=str, default=None,
                       help='YAML string to override config values')

    # Training settings
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run in fast development mode (1 epoch)')
    parser.add_argument('--overfit_batches', type=int, default=0,
                       help='Overfit on this many batches (for debugging)')

    # Explainability
    parser.add_argument('--enable_explainability', action='store_true',
                       help='Enable explainability analysis')
    parser.add_argument('--explanation_samples', type=int, default=10,
                       help='Number of samples to generate explanations for')
    parser.add_argument('--explanation_method', type=str, default='auto',
                       choices=['auto', 'signal_path', 'integrated_gradients', 'grad_cam'],
                       help='Explanation method to use')

    # Comparison
    parser.add_argument('--run_baseline_comparison', action='store_true',
                       help='Run comparison with baseline models')
    parser.add_argument('--baseline_models', nargs='+',
                       default=['TSPN', 'ResNet', 'SincNet'],
                       help='Baseline models to compare against')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='results/fusion_experiments/',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name')

    return parser.parse_args()


def load_config(config_file: str, config_override: str = None) -> dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides if provided
    if config_override:
        override_dict = yaml.safe_load(config_override)
        config.update(override_dict)

    # Convert nested args to flat namespace-like structure
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, key, Args(**value))
                else:
                    setattr(self, key, value)

        def to_dict(self):
            result = {}
            for key, value in self.__dict__.items():
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                elif isinstance(value, Args):
                    result[key] = vars(value)
                else:
                    result[key] = value
            return result

    # Create args object
    args = Args(**config)

    return args


def setup_experiment(args, output_dir: str):
    """Setup experiment logging and output directories"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set experiment name
    if not hasattr(args, 'experiment_name') or args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"Fusion1D2D_{timestamp}"

    # Setup wandb logger
    wandb_config = getattr(args, 'wandb', {})
    wandb_enabled = wandb_config.get('enabled', False)

    if wandb_enabled:
        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'Fusion1D2D_Experiments'),
            name=args.experiment_name,
            entity=wandb_config.get('entity', None),
            tags=wandb_config.get('tags', []),
            config=args.to_dict()
        )
    else:
        wandb_logger = None

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_config = getattr(args, 'checkpoint', {})
    callbacks.append(ModelCheckpoint(
        dirpath=output_dir + 'checkpoints/',
        filename=f"{args.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True)
    ))

    # Early stopping
    early_stop_config = getattr(args, 'early_stopping', {})
    if early_stop_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_loss'),
            patience=early_stop_config.get('patience', 10),
            mode=early_stop_config.get('mode', 'min'),
            verbose=True
        ))

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    return wandb_logger, callbacks


def train_fusion_model(args, output_dir: str, resume_from_checkpoint: str = None,
                      fast_dev_run: bool = False, overfit_batches: int = 0):
    """Train the 1D-2D fusion model"""

    print("Setting up experiment...")
    wandb_logger, callbacks = setup_experiment(args, output_dir)

    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = get_data(args)
        print(f"Data loaded successfully:")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print("Creating model...")
    trainer_module = create_fusion_trainer(args)

    print("Model information:")
    model_info = trainer_module.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("Setting up trainer...")
    # Configure trainer
    trainer_config = {
        'max_epochs': getattr(args, 'num_epochs', 30),
        'accelerator': 'auto',
        'devices': 1 if torch.cuda.is_available() else None,
        'callbacks': callbacks,
        'logger': wandb_logger,
        'deterministic': True,
        'enable_progress_bar': True,
    }

    if fast_dev_run:
        trainer_config['fast_dev_run'] = True
        trainer_config['overfit_batches'] = overfit_batches

    trainer = pl.Trainer(**trainer_config)

    print("Starting training...")
    try:
        trainer.fit(
            trainer_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_from_checkpoint
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    print("Evaluating on test set...")
    try:
        test_results = trainer.test(trainer_module, test_loader)
        print("Test results:", test_results)
    except Exception as e:
        print(f"Error during testing: {e}")
        test_results = None

    # Save model information
    import json
    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    return trainer_module, test_results


def run_explainability_analysis(trainer_module, test_loader, args, output_dir: str,
                               explanation_samples: int = 10, explanation_method: str = 'auto'):
    """Run explainability analysis on the trained model"""
    print("Running explainability analysis...")

    # Initialize explainer
    trainer_module.initialize_explainer(explanation_method)

    # Get a batch of test samples
    test_batch = next(iter(test_loader))
    x_test, y_test = test_batch

    # Limit number of samples
    if explanation_samples < len(x_test):
        x_test = x_test[:explanation_samples]
        y_test = y_test[:explanation_samples]

    # Generate explanations
    explanations = trainer_module.explain_batch(
        (x_test, y_test),
        method=explanation_method
    )

    # Analyze branch contributions
    branch_analysis = trainer_module.analyze_branch_contributions(test_batch)

    # Save explanations
    explainability_dir = os.path.join(output_dir, 'explainability')
    os.makedirs(explainability_dir, exist_ok=True)

    # Save branch analysis
    import json
    with open(os.path.join(explainability_dir, 'branch_analysis.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        branch_analysis_serializable = {}
        for key, value in branch_analysis.items():
            if isinstance(value, np.ndarray):
                branch_analysis_serializable[key] = value.tolist()
            elif isinstance(value, np.float32):
                branch_analysis_serializable[key] = float(value)
            else:
                branch_analysis_serializable[key] = value

        json.dump(branch_analysis_serializable, f, indent=2)

    print(f"Explainability analysis saved to {explainability_dir}")

    return explanations, branch_analysis


def main():
    """Main function"""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config_file}...")
    config_args = load_config(args.config_file, args.config_override)

    # Setup output directory
    output_dir = args.output_dir
    experiment_name = args.experiment_name or config_args.experiment_name
    if experiment_name:
        output_dir = os.path.join(output_dir, experiment_name)

    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_args.to_dict(), f, default_flow_style=False)

    # Train model
    trainer_module, test_results = train_fusion_model(
        config_args,
        output_dir,
        args.resume_from_checkpoint,
        args.fast_dev_run,
        args.overfit_batches
    )

    if trainer_module is None:
        print("Training failed. Exiting.")
        return

    # Run explainability analysis if requested
    if args.enable_explainability and trainer_module is not None:
        try:
            # Load test data
            _, _, test_loader = get_data(config_args)

            explanations, branch_analysis = run_explainability_analysis(
                trainer_module, test_loader, config_args, output_dir,
                args.explanation_samples, args.explanation_method
            )

            # Create visualization report
            if hasattr(config_args, 'explainability') and \
               getattr(config_args.explainability, 'visualization', {}).get('save_plots', True):
                create_visualization_report(
                    explanations,
                    branch_analysis,
                    output_dir=os.path.join(output_dir, 'visualizations'),
                    config=getattr(config_args.explainability, 'visualization', {})
                )

        except Exception as e:
            print(f"Error during explainability analysis: {e}")

    # Run baseline comparison if requested
    if args.run_baseline_comparison:
        print("Baseline comparison not yet implemented. This would run the same experiment with baseline models.")
        # TODO: Implement baseline comparison
        pass

    print(f"Experiment completed. Results saved to {output_dir}")

    # Close wandb
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()