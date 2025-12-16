from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from uxfd.io.schema_v1 import RunContext, write_run_schema
from uxfd.llm.settings import LLMSettings, load_dotenv
from uxfd.registry.papers import PAPER_REGISTRY
from uxfd.report.collector import collect_results_master


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _render_template_explanation(evidence: Dict[str, Any], style: str) -> str:
    """
    可审计的模板解释（无网/无 key 时默认使用）。
    """
    style = (style or "standard").strip()
    header = f"# 诊断解释（Template / style={style}）"
    lines = [header, ""]

    lines.append("## Run 信息")
    for k in ["run_id", "paper_id", "model_id", "dataset_id", "seed", "test_accuracy"]:
        if k in evidence:
            lines.append(f"- {k}: {evidence[k]}")
    lines.append("")

    lines.append("## 可解释评估（Paper2 schema）")
    for k in [
        "faithfulness_del_k_auc",
        "stability_spearman_mean",
        "eff_time_ms_per_sample",
        "sparsity_rules_activated_mean",
    ]:
        if k in evidence and evidence[k] is not None and evidence[k] == evidence[k]:
            lines.append(f"- {k}: {evidence[k]}")
    lines.append("")

    lines.append("## 说明")
    lines.append("- 本文本为模板生成：不引入外部事实，不推断未记录的信息。")
    lines.append("- 若配置了真实 LLM provider，可在同一证据字段上替换为模型生成解释，并记录 provider/latency/失败率。")
    lines.append("")
    return "\n".join(lines)


def run_paper3_llm(
    *,
    master_csv: str,
    roots: List[str],
    out_dir: str,
    provider: Optional[str],
    style: str,
    max_items: int,
    target_paper: Optional[str],
    config_path: str,
    notes: str = "",
) -> Path:
    """
    Paper3（LLM Toolkit）：
    - 消费 Paper2 schema 的 master 表（或 roots 扫描结果）
    - 生成“可审计自然语言解释”（默认 template/mock，不依赖网络）
    - 写入 Paper2 schema（paper3 的运行证据）
    """

    master_csv_path = Path(master_csv)
    if not master_csv_path.exists():
        collect_results_master(roots, master_csv_path)

    df = pd.read_csv(master_csv_path)
    if target_paper and "paper_id" in df.columns:
        df = df[df["paper_id"] == target_paper]

    df = df.head(int(max_items))

    load_dotenv()
    settings = LLMSettings.from_env()
    provider_req = (provider or settings.provider or "mock").strip()
    provider_used = provider_req
    if not settings.is_configured():
        provider_used = "template"

    run_dir = Path(out_dir) / f"run_{_utc_stamp()}"
    paper_dir = PAPER_REGISTRY["paper3"].paper_dir if "paper3" in PAPER_REGISTRY else Path("Paper/LLM_Explainable_FD_Toolkit")

    artifacts_logs = []
    artifacts_tables = [str(master_csv_path)]
    successes = 0
    start_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    t0 = time.perf_counter()

    logs_dir = run_dir / "artifacts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        evidence = row.to_dict()
        evidence["paper_id"] = evidence.get("paper_id")
        evidence["run_id"] = evidence.get("run_id")

        # 当前阶段：只做 template（避免无网环境失败）；保留 provider 字段用于后续替换
        text = _render_template_explanation(evidence, style=style)

        run_id = str(evidence.get("run_id", "unknown"))
        out_path = logs_dir / f"{run_id}_explanation.md"
        out_path.write_text(text, encoding="utf-8")
        artifacts_logs.append(str(out_path))
        successes += 1

        # 同步保存一份结构化证据（便于审计/对照）
        evidence_path = logs_dir / f"{run_id}_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts_logs.append(str(evidence_path))

    t1 = time.perf_counter()
    end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    latency_ms = (t1 - t0) * 1000.0 / max(1, int(successes))

    cmd = "python -m uxfd " + " ".join([c for c in sys.argv[1:] if c])
    ctx = RunContext(
        run_dir=run_dir,
        paper_id="paper3",
        paper_dir=paper_dir,
        model_id="LLMToolkit",
        seed=0,
        dataset_id="MULTI",
        dataset_numeric_id=None,
        command=cmd,
        config_path=config_path or "",
        device="cpu",
        task="llm_explanation",
        notes=notes,
    )

    metrics_patch: Dict[str, Any] = {
        "task": "llm_explanation",
        "llm": {
            "provider_requested": provider_req,
            "provider_used": provider_used,
            "style": style,
            "max_items": int(max_items),
            "num_inputs": int(len(df)),
            "num_success": int(successes),
            "latency_ms_per_item": float(latency_ms),
        },
        "artifacts": {
            "logs": artifacts_logs,
            "tables": artifacts_tables,
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
