# GEMINI.md: Project Guide for AI Agents

This document provides a comprehensive guide for AI agents interacting with the Unified X-Fault Diagnosis (UXFD) repository. It summarizes the project's architecture, conventions, and operational procedures.

## 1. Project Overview

This repository is a **Unified, Explainable Fault Diagnosis (UXFD) framework** designed for advanced machine learning research and academic paper writing. It is built on PyTorch and PyTorch Lightning.

**Dual Purpose:**
1.  **ML Research Platform:** A framework for developing, training, and evaluating fault diagnosis models, with a focus on explainable AI (XAI) through Transparent Signal Processing Networks (TSPNs).
2.  **Academic Writing Assistant:** A sophisticated ecosystem with integrated AI agents to support the entire research lifecycle, from literature review to manuscript preparation.

**Core Technologies:**
- **ML/DL:** PyTorch, PyTorch Lightning, Scikit-learn
- **Experiment Tracking:** Weights & Biases (`wandb`)
- **Data Handling:** Pandas, NumPy
- **Signal Processing:** `ptwt` (PyTorch Wavelet Toolbox), `scipy`
- **Environment:** Python 3.9, managed with `conda`.

## 2. Environment Setup

To get started, create and activate the Conda environment:

```shell
# Create the environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate UXFD
```
Set the `WANDB_API_KEY` environment variable for experiment tracking.

## 3. Running Experiments

All experiments are launched via Python entry points, controlled by YAML configuration files.

**Key Entry Points:**
- `main.py`: For primary models (TSPN, TKAN, NNSPN, TFON, MoE, etc.).
- `main_com.py`: For baseline/comparison models (ResNet, WKN, MCN, etc.).
- `main_kshotexp.py`: For K-shot few-sample learning experiments.
- `main_ablation_exp.py`: For ablation studies.

**Example Commands:**

```shell
# Run a primary model (e.g., TSPN)
python main.py --config_file configs/a_018_THU/config_TSPN.yaml

# Run a comparison model (e.g., Resnet)
# Note: The argument is --config_dir in AGENTS.md, but --config_file in CLAUDE.md.
# The `main.py` argparse definition is `--config_dir`, but it is used as a file.
# Prefer the documented command that aligns with the scripts.
python main_com.py --config_file configs/a_018_THU/config_com.yaml

# Run a quick demo
./script/demo.sh
```

**Batch Scripts:**
The `script/` directory contains shell scripts for running multiple experiments, such as `run_TSPN_ablation.sh` and `run_kshot_exp.sh`.

## 4. Project Architecture

- `model/`: Contains core, self-developed models like `TSPN.py`, `MoE.py`, and `OperatorAttention.py`.
- `model_collection/`: Contains implementations of baseline models from prior work.
- `trainer/`: Manages the PyTorch Lightning training (`trainer_basic.py`), setup (`trainer_set.py`), and utilities.
- `configs/`: Contains all YAML configuration files, organized by dataset and model type. This is the primary way to control experiments.
- `data/`: Handles data loading (`data_provider.py`, `datasets.py`).
- `script/`: Bash scripts for running experiments.
- `save/` & `wandb/`: Default output directories for model checkpoints and W&B logs (ignored by Git).

## 5. Development Conventions

- **Language & Style:** Python 3.9, following PEP 8 with 4-space indents. Use absolute imports from the project root.
- **Naming:**
    - Files & Functions: `snake_case`
    - Classes: `PascalCase`
    - Constants: `UPPER_SNAKE_CASE`
- **Configuration:** Changes should be made through YAML files. The parsing logic is in `configs/config.py`.
- **Testing:** The project currently lacks a formal test suite. For new features, add tests in the `tests/` directory using `pytest`. Focus on model forward passes and data I/O.
- **Commits:** Follow a conventional commit style (e.g., `feat: add MoE router`).

## 6. Guidelines for AI Agents

- **Primary Language:** Please respond and generate code comments in **Chinese (用中文回答)**.
- **Respect Scope:** Avoid unrelated refactoring. Focus on the task at hand and create minimal, targeted diffs.
- **Configuration over Code:** Prefer modifying YAML files in `configs/` to alter behavior instead of hardcoding changes in Python scripts.
- **File Modifications:** Touch only the files necessary to complete the request. Maintain backward compatibility.
- **Review Agent-Specific Files:** For more detailed context, refer to `AGENTS.md` and `CLAUDE.md`.
