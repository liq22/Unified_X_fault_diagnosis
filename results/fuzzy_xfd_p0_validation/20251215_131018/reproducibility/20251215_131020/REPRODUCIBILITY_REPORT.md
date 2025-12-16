# Reproducibility Report

**Experiment ID:** 20251215_131020
**Generated:** 2025-12-15T13:10:25.565144

## System Configuration

- **OS:** Linux 5.15.0-139-generic #149~20.04.1-Ubuntu SMP Wed Apr 16 08:29:56 UTC 2025
- **CPU:** x86_64 (144 cores)
- **Memory:** 188.5 GB
- **GPU:** NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9), NVIDIA GeForce RTX 4090 (Compute: 8.9)
- **CUDA Version:** 12.4
- **Python Version:** 3.10.0

## Library Versions

- **PyTorch:** 2.6.0+cu124
- **PyTorch Lightning:** 2.3.3
- **NumPy:** 1.23.5
- **Pandas:** 1.5.3
- **Scikit-learn:** 1.2.2

## Dataset Information

- **Path:** Unknown
- **Size:** 0.0 MB
- **Files:** 0
- **Checksum:** Unknown

## Model Information

- **Architecture:** FuzzyLogicV2
- **Parameters:** 0
- **Trainable:** 0
- **Model Size:** 0.0 MB
- **Config Hash:** 2c1d9345...

## Reproducibility Checklist

- ✅ Random Seeds Set
- ✅ Deterministic Algorithms
- ✅ Config Versioned
- ❌ Data Integrity Checked
- ✅ Environment Captured
- ❌ Gpu Determinism
- ❌ Results Saved

## Execution Log

  [2025-12-15T13:10:20.751272] Random seeds set to: 42
  [2025-12-15T13:10:21.578605] System information captured
  [2025-12-15T13:10:25.554987] Library information captured
  [2025-12-15T13:10:25.564775] Model configuration verified
  [2025-12-15T13:10:25.565099] Full configuration saved

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
