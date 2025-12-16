from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from uxfd.io.schema_v1 import RunContext, write_run_schema
from uxfd.registry.papers import PAPER_REGISTRY
from uxfd.report.collector import collect_results_master


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_paper6_theory(
    *,
    master_csv: str,
    roots: List[str],
    out_dir: str,
    config_path: str,
    notes: str = "",
) -> Path:
    """
    Paper6（Theory）：
    - 消费真实 runs 的 master 表
    - 生成命题验证表/相关性分析/异常点列表（最小可用版本）
    - 写入 Paper2 schema（paper6 的运行证据）
    """

    start_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    t0 = time.perf_counter()

    master_csv_path = Path(master_csv)
    if not master_csv_path.exists():
        collect_results_master(roots, master_csv_path)

    df = pd.read_csv(master_csv_path)
    if df.empty:
        raise RuntimeError(f"No rows in master_csv: {master_csv_path}")

    # 只分析模型论文（避免把 paper2/3/6 的工具 run 混进统计）
    model_mask = df["paper_id"].isin(["paper1", "paper4", "paper5", "paper7"]) if "paper_id" in df.columns else None
    df_model = df[model_mask].copy() if model_mask is not None else df.copy()

    group_cols = [c for c in ["paper_id", "dataset_id", "model_id"] if c in df_model.columns]
    metric_cols = [c for c in ["test_accuracy", "faithfulness_del_k_auc", "stability_spearman_mean", "eff_time_ms_per_sample"] if c in df_model.columns]

    summary = df_model.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]

    # 相关性分析（跨 run 粒度）
    corr_targets = [c for c in ["test_accuracy", "faithfulness_del_k_auc", "stability_spearman_mean"] if c in df_model.columns]
    corr = df_model[corr_targets].corr(method="spearman") if len(corr_targets) >= 2 else pd.DataFrame()

    # 命题：解释稳定性与性能的正相关（示例，便于后续替换为 paper6 真命题）
    proposition_rows: List[Dict[str, Any]] = []
    if "test_accuracy" in df_model.columns and "stability_spearman_mean" in df_model.columns:
        v = float(df_model["test_accuracy"].corr(df_model["stability_spearman_mean"], method="spearman"))
        proposition_rows.append(
            {
                "proposition_id": "P6-P1",
                "statement": "Stability 与 Accuracy 正相关（Spearman > 0）",
                "metric": "spearman_corr(acc, stability)",
                "value": v,
                "pass": bool(v > 0),
            }
        )
    if "test_accuracy" in df_model.columns and "faithfulness_del_k_auc" in df_model.columns:
        v = float(df_model["test_accuracy"].corr(df_model["faithfulness_del_k_auc"], method="spearman"))
        proposition_rows.append(
            {
                "proposition_id": "P6-P2",
                "statement": "Faithfulness 与 Accuracy 正相关（Spearman > 0）",
                "metric": "spearman_corr(acc, faithfulness)",
                "value": v,
                "pass": bool(v > 0),
            }
        )

    propositions = pd.DataFrame(proposition_rows)

    run_dir = Path(out_dir) / f"run_{_utc_stamp()}"
    tables_dir = run_dir / "artifacts" / "tables"
    logs_dir = run_dir / "artifacts" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tables_dir / "summary_by_group.csv"
    summary.to_csv(summary_path, index=False)

    corr_path = tables_dir / "spearman_correlation.csv"
    if not corr.empty:
        corr.to_csv(corr_path)

    prop_path = tables_dir / "propositions.csv"
    propositions.to_csv(prop_path, index=False)

    report_path = logs_dir / "theory_eval_report.md"
    report_lines = [
        "# Paper6 Theory Eval (Minimal)",
        "",
        f"- master_csv: `{master_csv_path}`",
        f"- rows_total: {len(df)}",
        f"- rows_model_only: {len(df_model)}",
        f"- groups: {len(summary)}",
        "",
        "## Propositions",
        "",
        _to_markdown_table(propositions, max_rows=200) if not propositions.empty else "No propositions computed.",
        "",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    t1 = time.perf_counter()
    end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    cmd = "python -m uxfd " + " ".join([c for c in sys.argv[1:] if c])
    paper_dir = PAPER_REGISTRY["paper6"].paper_dir if "paper6" in PAPER_REGISTRY else Path("Paper/Neuralsymbolic_theory")

    ctx = RunContext(
        run_dir=run_dir,
        paper_id="paper6",
        paper_dir=paper_dir,
        model_id="TheoryEval",
        seed=0,
        dataset_id="MULTI",
        dataset_numeric_id=None,
        command=cmd,
        config_path=config_path or "",
        device="cpu",
        task="theory_eval",
        notes=notes,
    )

    metrics_patch: Dict[str, Any] = {
        "task": "theory_eval",
        "theory_eval": {
            "rows_total": int(len(df)),
            "rows_model_only": int(len(df_model)),
            "groups": int(len(summary)),
        },
        "artifacts": {
            "logs": [str(report_path)],
            "tables": [str(summary_path), str(prop_path)] + ([str(corr_path)] if corr_path.exists() else []),
            "figures": [],
        },
    }
    run_meta_patch: Dict[str, Any] = {
        "timestamps": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "duration_sec": float(t1 - t0),
        }
    }

    write_run_schema(ctx, run_meta_patch=run_meta_patch, metrics_patch=metrics_patch)
    return run_dir
