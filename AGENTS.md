# Repository Guidelines
用中文回答
## Project Structure & Modules
- Code: `model/` (core networks), `model_collection/` (baselines), `trainer/` (loops, utils), `configs/` (YAML), `utils/`, `script/` (bash runners).
- Entrypoints: `main.py` (TSPN/TKAN/NNSPN/TFON) and `main_com.py` (baselines).
- Artifacts: `save/` and `wandb/` are git-ignored; keep large data in `data/`.

## Build, Test, and Development
- Environment: `conda env create -f environment.yml && conda activate UXFD`
- Run model: `python main.py --config_dir configs/a_018_THU/config_TSPN.yaml`
- Run baselines: `python main_com.py --config_dir configs/a_018_THU/config_Resnet.yaml`
- Scripts: see `script/run.sh` for dataset/model matrices. GPU pinning: `CUDA_VISIBLE_DEVICES=0 ...`
- Logging: Weights & Biases; set `WANDB_API_KEY` and project derives from config `dataset_task`.

## Coding Style & Naming
- Python 3.9, PEP 8, 4-space indent. Prefer type hints and docstrings on public APIs.
- Filenames and functions: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE`.
- Imports absolute from repo root (e.g., `from trainer.trainer_basic import Basic_plmodel`).
- Config keys: keep existing naming; extend via YAML and `configs/config.py`.

## Testing Guidelines
- Current repo has no formal unit tests; add `tests/` with `test_*.py` using `pytest` for new code.
- Focus: `trainer/` loops, data paths in `configs/`, and model forward shapes. Example: minimal forward pass on dummy tensor.
- Run: `pytest -q` (please add as a dev dependency if used).

## Commit & Pull Requests
- Commits: present tense, scoped and small. Example: `trainer: fix checkpoint path`.
- Include why in body if not obvious; reference issues like `Fixes #123`.
- PRs: clear description, config used, dataset path redactions, sample command to reproduce, and screenshots/metrics (W&B link) when relevant.

## Security & Configuration Tips
- Never commit private data or credentials. Paths live in YAML under `args.data_dir`.
- Reproducibility: set seeds via config; Lightning uses `seed_everything`.
- GPU/CPU: control via `args.device` and `CUDA_VISIBLE_DEVICES`.

## Agent-Specific Notes
- Respect scopes and existing style; avoid unrelated refactors.
- Touch only necessary files; prefer minimal diffs and keep behavior backward compatible.

see CLAUDE.md for more details on project overview, environment setup, and common commands.