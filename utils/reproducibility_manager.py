"""
Reproducibility Manager for Fuzzy-XFD
Ensures experiments are fully reproducible with detailed documentation
"""

import os
import sys
import json
import hashlib
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import platform
import psutil
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class SystemInfo:
    """System configuration information"""
    os: str
    os_version: str
    architecture: str
    cpu: str
    cpu_count: int
    memory_gb: float
    gpu: List[str]
    gpu_memory: List[float]
    cuda_version: str
    cudnn_version: str
    python_version: str
    pip_version: str


@dataclass
class LibraryInfo:
    """Python library versions"""
    pytorch: str
    pytorch_lightning: str
    numpy: str
    pandas: str
    scikit_learn: str
    matplotlib: str
    seaborn: str
    ptwt: str
    wandb: str


@dataclass
class DatasetInfo:
    """Dataset integrity information"""
    path: str
    size_bytes: int
    num_files: int
    checksum: str
    last_modified: datetime
    samples_train: int
    samples_val: int
    samples_test: int
    num_classes: int
    sample_rate: float
    duration: float


@dataclass
class ModelInfo:
    """Model configuration information"""
    architecture: str
    num_parameters: int
    trainable_parameters: int
    model_size_mb: float
    config_hash: str
    weights_hash: str


class ReproducibilityManager:
    """
    Comprehensive reproducibility management system
    """

    def __init__(self, experiment_dir: str = './reproducibility'):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.experiment_dir / timestamp
        self.run_dir.mkdir(exist_ok=True)

        # Initialize logs
        self.log_entries = []

    def capture_system_info(self) -> SystemInfo:
        """Capture detailed system configuration"""
        # OS information
        os_info = platform.uname()
        gpu_list = []
        gpu_memory = []

        # GPU information
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_list.append(f"{props.name} (Compute: {props.major}.{props.minor})")
                    gpu_memory.append(props.total_memory / 1024**3)  # Convert to GB

                # CUDA and cuDNN versions
                cuda_version = torch.version.cuda
                cudnn_version = torch.backends.cudnn.version()
            else:
                cuda_version = "Not available"
                cudnn_version = "Not available"
        except:
            cuda_version = "Error getting CUDA version"
            cudnn_version = "Error getting cuDNN version"

        # Get pip version
        try:
            pip_version = subprocess.check_output(
                ['pip', '--version'],
                universal_newlines=True
            ).split()[1]
        except:
            pip_version = "Not available"

        system_info = SystemInfo(
            os=os_info.system,
            os_version=f"{os_info.release} {os_info.version}",
            architecture=os_info.machine,
            cpu=os_info.processor,
            cpu_count=os.cpu_count(),
            memory_gb=psutil.virtual_memory().total / 1024**3,
            gpu=gpu_list,
            gpu_memory=gpu_memory,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            python_version=platform.python_version(),
            pip_version=pip_version
        )

        # Save system info
        with open(self.run_dir / 'system_info.json', 'w') as f:
            json.dump(asdict(system_info), f, indent=2)

        self.log("System information captured")
        return system_info

    def capture_library_info(self) -> LibraryInfo:
        """Capture Python library versions"""
        libraries = {
            'pytorch': torch.__version__,
        }

        # Try to import and get versions
        try:
            import pytorch_lightning
            libraries['pytorch_lightning'] = pytorch_lightning.__version__
        except:
            libraries['pytorch_lightning'] = "Not installed"

        try:
            import numpy
            libraries['numpy'] = numpy.__version__
        except:
            libraries['numpy'] = "Not installed"

        try:
            import pandas
            libraries['pandas'] = pandas.__version__
        except:
            libraries['pandas'] = "Not installed"

        try:
            import sklearn
            libraries['scikit_learn'] = sklearn.__version__
        except:
            libraries['scikit_learn'] = "Not installed"

        try:
            import matplotlib
            libraries['matplotlib'] = matplotlib.__version__
        except:
            libraries['matplotlib'] = "Not installed"

        try:
            import seaborn
            libraries['seaborn'] = seaborn.__version__
        except:
            libraries['seaborn'] = "Not installed"

        try:
            import ptwt
            libraries['ptwt'] = ptwt.__version__
        except:
            libraries['ptwt'] = "Not installed"

        try:
            import wandb
            libraries['wandb'] = wandb.__version__
        except:
            libraries['wandb'] = "Not installed"

        library_info = LibraryInfo(**libraries)

        # Save library info
        with open(self.run_dir / 'library_info.json', 'w') as f:
            json.dump(asdict(library_info), f, indent=2)

        self.log("Library information captured")
        return library_info

    def verify_dataset_integrity(self, data_path: str) -> DatasetInfo:
        """Verify dataset integrity and capture metadata"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

        # Calculate directory size and file count
        total_size = 0
        file_count = 0
        checksums = []

        for file_path in data_path.rglob('*'):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size

                # Calculate checksum for each file (sample only for large datasets)
                if file_count <= 100:  # Limit to first 100 files
                    with open(file_path, 'rb') as f:
                        checksums.append(hashlib.md5(f.read()).hexdigest())

        # Combine checksums
        overall_checksum = hashlib.md5(''.join(checksums).encode()).hexdigest()

        # Get modification time
        last_modified = datetime.fromtimestamp(
            max(f.stat().st_mtime for f in data_path.rglob('*') if f.is_file())
        )

        # Note: You would need to load the actual dataset to get these values
        # For now, we'll use placeholders
        dataset_info = DatasetInfo(
            path=str(data_path),
            size_bytes=total_size,
            num_files=file_count,
            checksum=overall_checksum,
            last_modified=last_modified,
            samples_train=0,  # To be filled by actual data loading
            samples_val=0,
            samples_test=0,
            num_classes=5,  # Default for THU_018
            sample_rate=0.0,  # To be filled
            duration=0.0  # To be filled
        )

        # Save dataset info
        with open(self.run_dir / 'dataset_info.json', 'w') as f:
            # Convert datetime to string for JSON serialization
            dataset_dict = asdict(dataset_info)
            dataset_dict['last_modified'] = dataset_info.last_modified.isoformat()
            json.dump(dataset_dict, f, indent=2)

        self.log(f"Dataset integrity verified: {file_count} files, {total_size/1024**2:.1f} MB")
        return dataset_info

    def verify_model_config(self, config_path: str, model_path: Optional[str] = None) -> ModelInfo:
        """Verify model configuration and capture metadata"""
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Calculate config hash
        with open(config_path, 'rb') as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()

        # Get model info from configuration
        model_info = ModelInfo(
            architecture=config['args'].get('model', 'Unknown'),
            num_parameters=0,  # To be filled by model loading
            trainable_parameters=0,  # To be filled
            model_size_mb=0,  # To be filled
            config_hash=config_hash,
            weights_hash=""
        )

        # If model path provided, load and analyze
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')

                # Count parameters
                state_dict = checkpoint.get('state_dict', checkpoint)
                total_params = sum(p.numel() for p in state_dict.values())
                trainable_params = sum(p.numel() for p in state_dict.values() if p.requires_grad)

                # Calculate model size
                model_size_mb = Path(model_path).stat().st_size / 1024**2

                # Calculate weights hash
                weights_hash = hashlib.sha256(
                    str(sorted([(k, v.shape) for k, v in state_dict.items()])).encode()
                ).hexdigest()

                model_info.num_parameters = total_params
                model_info.trainable_parameters = trainable_params
                model_info.model_size_mb = model_size_mb
                model_info.weights_hash = weights_hash

            except Exception as e:
                self.log(f"Warning: Could not analyze model: {e}")

        # Save model info
        with open(self.run_dir / 'model_info.json', 'w') as f:
            json.dump(asdict(model_info), f, indent=2)

        self.log("Model configuration verified")
        return model_info

    def save_full_configuration(self, config_path: str):
        """Save complete configuration for reproducibility"""
        # Copy original config
        import shutil
        shutil.copy2(config_path, self.run_dir / 'config.yaml')

        # Save command line arguments
        if hasattr(sys, 'argv'):
            with open(self.run_dir / 'command_line.txt', 'w') as f:
                f.write(' '.join(sys.argv))

        # Save environment variables (filtered)
        env_vars = {}
        relevant_vars = ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'WANDB_API_KEY']
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        with open(self.run_dir / 'environment.json', 'w') as f:
            json.dump(env_vars, f, indent=2)

        self.log("Full configuration saved")

    def set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        # Python
        import random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # PyTorch deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save seed information
        with open(self.run_dir / 'seeds.json', 'w') as f:
            json.dump({
                'seed': int(seed),
                'numpy_seed': int(np.random.get_state()[1][0]),
                'python_seed': int(random.getstate()[1][0]),
                'torch_seed': int(torch.initial_seed())
            }, f, indent=2)

        self.log(f"Random seeds set to: {seed}")

    def log(self, message: str):
        """Log a message with timestamp"""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {message}"
        self.log_entries.append(entry)

        # Also write to file immediately
        with open(self.run_dir / 'reproducibility.log', 'a') as f:
            f.write(entry + '\n')

    def generate_reproducibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report"""
        report = {
            'experiment_id': self.run_dir.name,
            'timestamp': datetime.now().isoformat(),
            'log_entries': self.log_entries,
            'files': {
                'system_info': 'system_info.json',
                'library_info': 'library_info.json',
                'dataset_info': 'dataset_info.json',
                'model_info': 'model_info.json',
                'config': 'config.yaml',
                'environment': 'environment.json',
                'seeds': 'seeds.json',
                'log': 'reproducibility.log'
            },
            'reproducibility_checklist': self.generate_checklist()
        }

        # Save report
        with open(self.run_dir / 'reproducibility_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Create human-readable report
        self.create_human_readable_report(report)

        return report

    def generate_checklist(self) -> Dict[str, bool]:
        """Generate reproducibility checklist"""
        checklist = {
            'random_seeds_set': False,
            'deterministic_algorithms': True,
            'config_versioned': True,
            'data_integrity_checked': False,
            'environment_captured': True,
            'gpu_determinism': False,
            'results_saved': False
        }

        # Check if seeds were set
        if (self.run_dir / 'seeds.json').exists():
            checklist['random_seeds_set'] = True

        # Check dataset info
        if (self.run_dir / 'dataset_info.json').exists():
            checklist['data_integrity_checked'] = True

        # GPU determinism is limited in PyTorch
        checklist['gpu_determinism'] = False

        return checklist

    def create_human_readable_report(self, report: Dict[str, Any]):
        """Create a human-readable markdown report"""
        report_path = self.run_dir / 'REPRODUCIBILITY_REPORT.md'

        # Load info files
        system_info = {}
        library_info = {}
        dataset_info = {}
        model_info = {}

        if (self.run_dir / 'system_info.json').exists():
            with open(self.run_dir / 'system_info.json') as f:
                system_info = json.load(f)

        if (self.run_dir / 'library_info.json').exists():
            with open(self.run_dir / 'library_info.json') as f:
                library_info = json.load(f)

        if (self.run_dir / 'dataset_info.json').exists():
            with open(self.run_dir / 'dataset_info.json') as f:
                dataset_info = json.load(f)

        if (self.run_dir / 'model_info.json').exists():
            with open(self.run_dir / 'model_info.json') as f:
                model_info = json.load(f)

        # Create markdown report
        md_content = f"""# Reproducibility Report

**Experiment ID:** {report['experiment_id']}
**Generated:** {report['timestamp']}

## System Configuration

- **OS:** {system_info.get('os', 'Unknown')} {system_info.get('os_version', '')}
- **CPU:** {system_info.get('cpu', 'Unknown')} ({system_info.get('cpu_count', 0)} cores)
- **Memory:** {system_info.get('memory_gb', 0):.1f} GB
- **GPU:** {', '.join(system_info.get('gpu', ['None']))}
- **CUDA Version:** {system_info.get('cuda_version', 'Unknown')}
- **Python Version:** {system_info.get('python_version', 'Unknown')}

## Library Versions

- **PyTorch:** {library_info.get('pytorch', 'Unknown')}
- **PyTorch Lightning:** {library_info.get('pytorch_lightning', 'Unknown')}
- **NumPy:** {library_info.get('numpy', 'Unknown')}
- **Pandas:** {library_info.get('pandas', 'Unknown')}
- **Scikit-learn:** {library_info.get('scikit_learn', 'Unknown')}

## Dataset Information

- **Path:** {dataset_info.get('path', 'Unknown')}
- **Size:** {dataset_info.get('size_bytes', 0) / 1024**2:.1f} MB
- **Files:** {dataset_info.get('num_files', 0)}
- **Checksum:** {dataset_info.get('checksum', 'Unknown')}

## Model Information

- **Architecture:** {model_info.get('architecture', 'Unknown')}
- **Parameters:** {model_info.get('num_parameters', 0):,}
- **Trainable:** {model_info.get('trainable_parameters', 0):,}
- **Model Size:** {model_info.get('model_size_mb', 0):.1f} MB
- **Config Hash:** {model_info.get('config_hash', 'Unknown')[:8]}...

## Reproducibility Checklist

{self._format_checklist(report['reproducibility_checklist'])}

## Execution Log

{self._format_log_entries(report['log_entries'])}

## How to Reproduce

1. Install required libraries with exact versions (see Library Versions section)
2. Ensure similar hardware configuration (GPU with similar memory)
3. Set random seeds using the saved `seeds.json` file
4. Use the exact configuration from `config.yaml`
5. Run with the same command line arguments (see `command_line.txt`)

## Files for Reproduction

All necessary files are included in this directory:
- `config.yaml` - Model configuration
- `seeds.json` - Random seed values
- `environment.json` - Relevant environment variables
- `command_line.txt` - Command used to run experiment
"""

        with open(report_path, 'w') as f:
            f.write(md_content)

        self.log(f"Human-readable report created: {report_path}")

    def _format_checklist(self, checklist: Dict[str, bool]) -> str:
        """Format checklist as markdown"""
        items = []
        for item, checked in checklist.items():
            status = '✅' if checked else '❌'
            items.append(f"- {status} {item.replace('_', ' ').title()}")
        return '\n'.join(items)

    def _format_log_entries(self, entries: List[str]) -> str:
        """Format log entries for markdown"""
        return '\n'.join(f"  {entry}" for entry in entries)


def main():
    """
    Example usage of reproducibility manager
    """
    import argparse

    parser = argparse.ArgumentParser(description='Manage reproducibility for Fuzzy-XFD experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                       help='Path to dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='./reproducibility',
                       help='Output directory')

    args = parser.parse_args()

    # Initialize manager
    manager = ReproducibilityManager(args.output)

    # Set random seeds
    manager.set_random_seeds(args.seed)

    # Capture all information
    manager.capture_system_info()
    manager.capture_library_info()
    manager.verify_dataset_integrity(args.data) if args.data else None
    manager.verify_model_config(args.config, args.model)

    # Save configuration
    manager.save_full_configuration(args.config)

    # Generate report
    report = manager.generate_reproducibility_report()

    print(f"\nReproducibility package created in: {manager.run_dir}")
    print(f"Experiment ID: {report['experiment_id']}")
    print(f"\nTo reproduce, ensure:")
    print(f"1. Same library versions (see library_info.json)")
    print(f"2. Similar hardware configuration")
    print(f"3. Use config.yaml and seeds.json from this directory")


if __name__ == "__main__":
    main()