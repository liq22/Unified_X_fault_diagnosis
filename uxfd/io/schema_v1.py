from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SCHEMA_VERSION = "paper2_schema_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_info(repo_root: Path) -> Tuple[str, bool]:
    import subprocess

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "UNKNOWN"

    try:
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        dirty = True

    return commit, dirty


def _torch_version() -> str:
    try:
        import torch  # type: ignore

        return str(torch.__version__)
    except Exception:
        return "UNKNOWN"


def _infer_paper_id_from_model_id(model_id: str) -> str:
    model_id = (model_id or "").strip()
    mapping = {
        "Fusion1D2D": "paper1",
        "MoE": "paper4",
        "FuzzyLogic": "paper5",
        "FuzzyLogicV2": "paper5",
        "OperatorAttention": "paper7",
    }
    return mapping.get(model_id, "baseline")


def _read_test_result_csv(test_result_csv: Path) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(test_result_csv)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): v for k, v in row.items()}


def _extract_test_accuracy(test_row: Dict[str, Any]) -> Optional[float]:
    for key in ("test/accuracy", "test_acc", "test_accuracy", "acc", "accuracy"):
        if key in test_row and test_row[key] is not None:
            try:
                return float(test_row[key])
            except Exception:
                continue
    return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并字典：patch 覆盖 base 的同名字段；子 dict 做深合并。
    """
    out: Dict[str, Any] = dict(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    paper_id: str
    paper_dir: Path
    model_id: str
    seed: int
    dataset_id: str
    dataset_numeric_id: Optional[int]
    command: str
    config_path: str
    device: str
    task: str = "classification"
    notes: str = ""


def write_run_schema(
    ctx: RunContext,
    test_result_csv: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    run_meta_patch: Optional[Dict[str, Any]] = None,
    metrics_patch: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """
    在 `<RUN_DIR>` 下写入：
    - `run_meta.yaml`
    - `metrics.json`
    并确保 `artifacts/` 存在。
    """

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    _ensure_dir(ctx.run_dir)
    artifacts_dir = ctx.run_dir / "artifacts"
    _ensure_dir(artifacts_dir)
    _ensure_dir(artifacts_dir / "figures")
    _ensure_dir(artifacts_dir / "tables")
    _ensure_dir(artifacts_dir / "logs")

    test_row: Dict[str, Any] = {}
    if test_result_csv is not None and test_result_csv.exists():
        test_row = _read_test_result_csv(test_result_csv)

    test_acc = _extract_test_accuracy(test_row)
    if test_acc is None:
        test_acc = 0.0

    commit, dirty = _git_info(repo_root)

    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "paper": {
            "paper_id": ctx.paper_id,
            "paper_dir": str(ctx.paper_dir),
        },
        "run": {
            "run_id": ctx.run_dir.name,
            "seed": int(ctx.seed),
            "dataset_id": ctx.dataset_id,
            "dataset_numeric_id": ctx.dataset_numeric_id,
            "model_id": ctx.model_id,
            "explainer_id": "uxfd",
        },
        "repro": {
            "command": ctx.command,
            "config_path": ctx.config_path,
        },
        "git": {"commit": commit, "dirty": bool(dirty)},
        "env": {
            "python": platform.python_version(),
            "torch": _torch_version(),
            "device": ctx.device,
        },
        "timestamps": {
            "start_utc": _utc_now_iso(),
            "end_utc": _utc_now_iso(),
        },
        "outputs": {
            "run_dir": str(ctx.run_dir),
            "metrics_path": str(ctx.run_dir / "metrics.json"),
        },
    }
    if ctx.notes:
        run_meta["notes"] = ctx.notes

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": ctx.paper_id,
        "dataset_id": ctx.dataset_id,
        "model_id": ctx.model_id,
        "seed": int(ctx.seed),
        "task": ctx.task,
        "split_metrics": {"test": {"accuracy": float(test_acc)}},
        "explainability": {},
        "artifacts": {
            "logs": [],
            "figures": [],
            "tables": [],
        },
    }
    if test_row:
        metrics["raw_test_row"] = test_row

    if isinstance(run_meta_patch, dict) and run_meta_patch:
        run_meta = _deep_merge_dict(run_meta, run_meta_patch)
    if isinstance(metrics_patch, dict) and metrics_patch:
        metrics = _deep_merge_dict(metrics, metrics_patch)

    # 写入文件
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to write run_meta.yaml") from exc

    run_meta_path = ctx.run_dir / "run_meta.yaml"
    metrics_path = ctx.run_dir / "metrics.json"
    run_meta_path.write_text(yaml.safe_dump(run_meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return run_meta_path, metrics_path


def infer_paper_id(model_id: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env_pid = os.getenv("UXFD_PAPER_ID")
    if env_pid:
        return env_pid
    return _infer_paper_id_from_model_id(model_id)
