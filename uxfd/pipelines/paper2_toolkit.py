from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from uxfd.io.schema_v1 import RunContext, write_run_schema
from uxfd.registry.papers import PAPER_REGISTRY
from uxfd.report.collector import collect_results_master
from uxfd.report.simple_report import write_simple_markdown_report


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_paper2_toolkit(
    *,
    roots: List[str],
    out_csv: str,
    out_md: str,
    target_paper: Optional[str],
    output_root: str,
    config_path: str,
    do_collect: bool = True,
    do_report: bool = True,
    notes: str = "",
) -> Path:
    """
    Paper2（Toolkit）：
    - collect: 扫描 runs（schema v1）汇总 master csv
    - report: 生成最小 Markdown 报告
    - 同时写入 Paper2 schema（工具链运行证据）
    """

    start_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    t0 = time.perf_counter()

    out_csv_path = Path(out_csv)
    if do_collect:
        out_csv_path = collect_results_master(roots, out_csv_path)

    out_md_path: Optional[Path] = None
    if do_report:
        out_md_path = write_simple_markdown_report(out_csv_path, out_md, target_paper=target_paper)

    t1 = time.perf_counter()
    end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    run_dir = Path(output_root) / "uxfd" / "paper2_toolkit" / f"run_{_utc_stamp()}"
    paper_dir = PAPER_REGISTRY["paper2"].paper_dir if "paper2" in PAPER_REGISTRY else Path("Paper/Explainable_FD_Toolkit")

    # 将关键产物复制进本次 run 的 artifacts（形成自包含证据链）
    logs_dir = run_dir / "artifacts" / "logs"
    tables_dir = run_dir / "artifacts" / "tables"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    copied_master_csv: Optional[Path] = None
    if out_csv_path.exists():
        copied_master_csv = tables_dir / out_csv_path.name
        if copied_master_csv.resolve() != out_csv_path.resolve():
            shutil.copy2(out_csv_path, copied_master_csv)

    copied_report_md: Optional[Path] = None
    if out_md_path is not None and out_md_path.exists():
        copied_report_md = logs_dir / out_md_path.name
        if copied_report_md.resolve() != out_md_path.resolve():
            shutil.copy2(out_md_path, copied_report_md)

    cmd = "python -m uxfd " + " ".join([c for c in sys.argv[1:] if c])
    run_meta_patch: Dict[str, Any] = {
        "timestamps": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "duration_sec": float(t1 - t0),
        }
    }

    metrics_patch: Dict[str, Any] = {
        "task": "collect_report",
        "toolkit": {
            "master_csv": str(out_csv_path),
            "report_md": str(out_md_path) if out_md_path else None,
            "target_paper": target_paper,
            "master_csv_in_run": str(copied_master_csv) if copied_master_csv else None,
            "report_md_in_run": str(copied_report_md) if copied_report_md else None,
        },
        "artifacts": {
            "logs": [str(copied_report_md)] if copied_report_md else ([str(out_md_path)] if out_md_path else []),
            "tables": [str(copied_master_csv)] if copied_master_csv else [str(out_csv_path)],
            "figures": [],
        },
    }

    ctx = RunContext(
        run_dir=run_dir,
        paper_id="paper2",
        paper_dir=paper_dir,
        model_id="Toolkit",
        seed=0,
        dataset_id="MULTI",
        dataset_numeric_id=None,
        command=cmd,
        config_path=config_path or "",
        device="cpu",
        task="collect_report",
        notes=notes,
    )
    write_run_schema(ctx, run_meta_patch=run_meta_patch, metrics_patch=metrics_patch)

    return run_dir
