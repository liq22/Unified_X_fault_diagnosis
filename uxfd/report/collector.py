from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = "paper2_schema_v1"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read run_meta.yaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _iter_run_dirs(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("run_meta.yaml"):
            yield p.parent


def _collect_one_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    run_meta_path = run_dir / "run_meta.yaml"
    metrics_path = run_dir / "metrics.json"
    if not run_meta_path.exists() or not metrics_path.exists():
        return None

    run_meta = _read_yaml(run_meta_path)
    if run_meta.get("schema_version") != SCHEMA_VERSION:
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != SCHEMA_VERSION:
        return None

    row: Dict[str, Any] = {
        "paper_id": _safe_get(run_meta, "paper.paper_id"),
        "dataset_id": _safe_get(run_meta, "run.dataset_id"),
        "dataset_numeric_id": _safe_get(run_meta, "run.dataset_numeric_id"),
        "model_id": _safe_get(run_meta, "run.model_id"),
        "seed": _safe_get(run_meta, "run.seed"),
        "run_id": _safe_get(run_meta, "run.run_id"),
        "run_dir": str(run_dir),
        "git_commit": _safe_get(run_meta, "git.commit"),
        "git_dirty": _safe_get(run_meta, "git.dirty"),
        "test_accuracy": _safe_get(metrics, "split_metrics.test.accuracy"),
        "test_f1_macro": _safe_get(metrics, "split_metrics.test.f1_macro"),
        "faithfulness_del_k_auc": _safe_get(metrics, "explainability.faithfulness.del_k_auc"),
        "stability_spearman_mean": _safe_get(metrics, "explainability.stability.spearman_mean"),
        "eff_time_ms_per_sample": _safe_get(metrics, "explainability.efficiency.time_ms_per_sample"),
        "sparsity_rules_activated_mean": _safe_get(metrics, "explainability.sparsity.rules_activated_mean"),
    }
    return row


def collect_results_master(roots: Sequence[str | Path], out_csv: str | Path) -> Path:
    """
    扫描多个 roots（如 `save/` 与 `outputs/`）下符合 Paper2 schema v1 的 runs，
    汇总为一个 master CSV 表。
    """

    root_paths = [Path(p) for p in roots]
    run_dirs = list(_iter_run_dirs(root_paths))

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = _collect_one_run(run_dir)
        if row is not None:
            rows.append(row)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path

