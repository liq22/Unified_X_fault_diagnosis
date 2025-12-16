from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from uxfd.config import load_run_config
from uxfd.llm.settings import LLMSettings, load_dotenv
from uxfd.pipelines import run_paper2_toolkit, run_paper3_llm, run_paper6_theory
from uxfd.registry.papers import PAPER_REGISTRY, PaperType
from uxfd.report.collector import collect_results_master
from uxfd.report.simple_report import write_simple_markdown_report


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_kv(key: str, value: str) -> None:
    print(f"{key}: {value}")


def cmd_doctor(_: argparse.Namespace) -> int:
    repo_root = _repo_root()
    _print_kv("repo_root", str(repo_root))
    _print_kv("python", sys.version.replace("\n", " "))

    try:
        import torch  # type: ignore

        _print_kv("torch", torch.__version__)
        _print_kv("cuda_available", str(torch.cuda.is_available()))
    except Exception:
        _print_kv("torch", "NOT_INSTALLED")

    gitignore = repo_root / ".gitignore"
    if gitignore.exists():
        gitignore_text = gitignore.read_text(encoding="utf-8")
        _print_kv(".env_gitignored", str(".env" in gitignore_text))
    else:
        _print_kv(".env_gitignored", "UNKNOWN (.gitignore missing)")

    dotenv_example = repo_root / ".env.example"
    _print_kv(".env.example_exists", str(dotenv_example.exists()))

    load_dotenv()
    llm = LLMSettings.from_env()
    _print_kv("LLM_PRIMARY_PROVIDER", llm.provider)
    _print_kv("LLM_configured", str(llm.is_configured()))

    _print_kv("papers", ", ".join(sorted(PAPER_REGISTRY.keys())))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    out_path = collect_results_master(args.roots, args.out)
    print(f"[OK] wrote: {out_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_path = write_simple_markdown_report(args.input, args.out, target_paper=args.paper)
    print(f"[OK] wrote: {out_path}")
    return 0


def _parse_maybe_paths(values: Optional[List[str]]) -> List[str]:
    return values or []


def _order_papers(paper_ids: List[str]) -> List[str]:
    """
    执行顺序：先模型论文产出 run，再由工具/LLM/理论消费 artifacts。
    """

    order = {
        PaperType.MODEL: 0,
        PaperType.TOOLKIT: 1,
        PaperType.LLM_TOOLKIT: 2,
        PaperType.THEORY: 3,
    }

    def _key(pid: str) -> int:
        spec = PAPER_REGISTRY.get(pid)
        if spec is None:
            return 99
        return int(order.get(spec.paper_type, 99))

    return sorted(paper_ids, key=_key)


def _run_one_paper(
    paper_id: str,
    modes: List[str],
    *,
    roots: Optional[List[str]] = None,
    out_csv: Optional[str] = None,
    report_out: Optional[str] = None,
    target_paper: Optional[str] = None,
    base_config: Optional[str] = None,
    datasets: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    output_root: str = "outputs",
    run_config_path: str = "",
    llm_provider: Optional[str] = None,
    llm_style: Optional[str] = None,
    llm_out_dir: Optional[str] = None,
    llm_max_items: Optional[int] = None,
    theory_out_dir: Optional[str] = None,
    dry_run: bool = False,
    notes: str = "",
) -> int:
    if paper_id not in PAPER_REGISTRY:
        raise SystemExit(f"Unknown paper_id: {paper_id} (available: {sorted(PAPER_REGISTRY.keys())})")
    spec = PAPER_REGISTRY[paper_id]

    modes_set = set([str(m).strip() for m in (modes or []) if str(m).strip()])
    if not modes_set:
        modes_set = {"collect", "report"} if spec.paper_type == PaperType.TOOLKIT else {"doctor"}

    if spec.paper_type == PaperType.TOOLKIT:
        roots = roots or ["save", "outputs"]
        out_csv = out_csv or "results/results_table_master.csv"
        do_collect = "collect" in modes_set
        do_report = "report" in modes_set
        if not (do_collect or do_report):
            print(f"[SKIP] {paper_id}: no collect/report in modes={sorted(modes_set)}")
            return 0
        if dry_run:
            if do_collect:
                print(f"[PLAN] collect_results_master roots={roots} out={out_csv}")
            if do_report:
                out_md = report_out or "results/uxfd_report.md"
                print(f"[PLAN] write_simple_markdown_report in={out_csv} out={out_md} target_paper={target_paper}")
            return 0
        out_md = report_out or "results/uxfd_report.md"
        run_dir = run_paper2_toolkit(
            roots=roots,
            out_csv=out_csv,
            out_md=out_md,
            target_paper=target_paper,
            output_root=output_root,
            config_path=run_config_path,
            do_collect=do_collect,
            do_report=do_report,
            notes=notes,
        )
        if do_collect:
            print(f"[OK] collect -> {out_csv}")
        if do_report:
            print(f"[OK] report -> {out_md}")
        print(f"[OK] paper2 run_dir -> {run_dir}")
        return 0

    if spec.paper_type == PaperType.LLM_TOOLKIT:
        if "llm" not in modes_set:
            print(f"[SKIP] {paper_id}: no llm in modes={sorted(modes_set)}")
            return 0
        if dry_run:
            master_csv = out_csv or "results/results_table_master.csv"
            out_dir = llm_out_dir or "outputs/uxfd/paper3_llm_explanations"
            print(f"[PLAN] paper3 llm_explain master_csv={master_csv} out_dir={out_dir} provider={llm_provider} style={llm_style}")
            return 0
        master_csv = out_csv or "results/results_table_master.csv"
        out_dir = llm_out_dir or "outputs/uxfd/paper3_llm_explanations"
        run_dir = run_paper3_llm(
            master_csv=master_csv,
            roots=roots or ["save", "outputs"],
            out_dir=out_dir,
            provider=llm_provider,
            style=llm_style or "standard",
            max_items=int(llm_max_items or 50),
            target_paper=target_paper,
            config_path=run_config_path,
            notes=notes,
        )
        print(f"[OK] paper3 run_dir -> {run_dir}")
        return 0

    if spec.paper_type == PaperType.THEORY:
        if "theory_eval" not in modes_set:
            print(f"[SKIP] {paper_id}: no theory_eval in modes={sorted(modes_set)}")
            return 0
        if dry_run:
            master_csv = out_csv or "results/results_table_master.csv"
            out_dir = theory_out_dir or "outputs/uxfd/paper6_theory_eval"
            print(f"[PLAN] paper6 theory_eval master_csv={master_csv} out_dir={out_dir}")
            return 0
        master_csv = out_csv or "results/results_table_master.csv"
        out_dir = theory_out_dir or "outputs/uxfd/paper6_theory_eval"
        run_dir = run_paper6_theory(
            master_csv=master_csv,
            roots=roots or ["save", "outputs"],
            out_dir=out_dir,
            config_path=run_config_path,
            notes=notes,
        )
        print(f"[OK] paper6 run_dir -> {run_dir}")
        return 0

    # Model papers: orchestrate legacy main.py runs (per-dataset, per-seed)
    if not ({"train", "eval", "explain"} & modes_set):
        print(f"[SKIP] {paper_id}: no train/eval/explain in modes={sorted(modes_set)}")
        return 0

    base_config = base_config or (str(spec.default_base_config) if spec.default_base_config else None)
    if base_config is None:
        raise SystemExit(f"base_config is required for {paper_id} (no default_base_config in registry)")

    datasets = datasets or []
    seeds = seeds or []
    if not seeds:
        seeds = [17]

    plans = []
    config_paths: List[str] = []

    if datasets:
        for ds in datasets:
            gen_path = _generate_single_vibench_config(
                base_config_path=Path(base_config),
                dataset_numeric_id=int(ds),
                out_dir=_repo_root() / "outputs" / "uxfd" / "generated_configs" / paper_id,
            )
            config_paths.append(str(gen_path))
    else:
        config_paths.append(str(base_config))

    for config_path in config_paths:
        for seed in seeds:
            cmd = [
                sys.executable,
                "main.py",
                "--config_file",
                str(config_path),
                "--iteration",
                "1",
                "--seed",
                str(seed),
            ]
            if notes:
                cmd.extend(["--notes", notes])
            plans.append(cmd)

    for cmd in plans:
        print(f"[PLAN] {' '.join(cmd)}")

    if dry_run:
        return 0

    import subprocess

    env = os.environ.copy()
    env["UXFD_PAPER_ID"] = paper_id

    for cmd in plans:
        subprocess.run(cmd, check=True, env=env, cwd=str(_repo_root()))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """
    统一执行入口：
    - 推荐：`python -m uxfd run --run_config configs/unified_papers.yaml`
    - 也支持单 paper：`python -m uxfd run --paper paper2 --modes collect report`
    """

    if args.run_config:
        run_cfg, _raw = load_run_config(args.run_config, repo_root=_repo_root())
        paper_ids = run_cfg.target_papers()
        if args.paper:
            paper_ids = [args.paper]
        if not paper_ids:
            paper_ids = sorted(PAPER_REGISTRY.keys())

        paper_ids = _order_papers(paper_ids)

        modes = args.modes or run_cfg.modes
        datasets = args.datasets if args.datasets is not None else run_cfg.datasets
        seeds = args.seeds if args.seeds is not None else run_cfg.seeds
        roots = args.roots or run_cfg.roots

        out_csv = args.out or run_cfg.report.out_csv
        report_out = args.report_out or run_cfg.report.out_md
        target_paper = args.target_paper or run_cfg.report.target_paper

        for pid in paper_ids:
            override = run_cfg.paper_overrides.get(pid)
            base_config = args.config or (override.base_config if override else None)
            _run_one_paper(
                paper_id=pid,
                modes=modes,
                roots=roots,
                out_csv=out_csv,
                report_out=report_out,
                target_paper=target_paper,
                base_config=base_config,
                datasets=datasets,
                seeds=seeds,
                output_root=run_cfg.output_root,
                run_config_path=str(args.run_config),
                llm_provider=run_cfg.llm.provider,
                llm_style=run_cfg.llm.style,
                llm_out_dir=run_cfg.llm.out_dir,
                llm_max_items=run_cfg.llm.max_items,
                theory_out_dir=run_cfg.theory.out_dir,
                dry_run=bool(args.dry_run),
                notes=str(args.notes or ""),
            )
        return 0

    if not args.paper:
        raise SystemExit("--paper is required when --run_config is not provided")

    return _run_one_paper(
        paper_id=args.paper,
        modes=args.modes or [],
        roots=args.roots,
        out_csv=args.out,
        report_out=args.report_out,
        target_paper=args.target_paper,
        base_config=args.config,
        datasets=args.datasets,
        seeds=args.seeds,
        output_root="outputs",
        run_config_path="",
        dry_run=bool(args.dry_run),
        notes=str(args.notes or ""),
    )


def _generate_single_vibench_config(base_config_path: Path, dataset_numeric_id: int, out_dir: Path) -> Path:
    """
    生成单数据集 vbench 配置（dataset_ids=[id]），用于 uxfd CLI 编排。
    注意：此阶段以 legacy main.py 为真源，因此只生成最小覆盖字段，避免“重写一套配置系统”。
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to generate configs") from exc

    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "args" not in cfg:
        raise ValueError(f"Invalid base config: {base_config_path}")

    # 允许 base config 在 args 或顶层提供 vbench_config：最终写回顶层，便于 legacy adapter 注入
    vb = {}
    if isinstance(cfg.get("vbench_config"), dict):
        vb.update(cfg["vbench_config"])
    if isinstance(cfg["args"], dict) and isinstance(cfg["args"].get("vbench_config"), dict):
        vb.update(cfg["args"]["vbench_config"])

    vb.setdefault("metadata_file", "metadata_6_11.xlsx")
    vb.setdefault("target_column", "Label")
    vb.setdefault("task_type", "fault_diagnosis")

    # data_dir 默认从环境变量读取（可在 .env 中配置）
    vb_data_dir = vb.get("data_dir") or os.getenv("PHM_VIBENCH_ROOT")
    if not vb_data_dir:
        vb_data_dir = "/home/user/data/PHMbenchdata/PHM-Vibench"
    vb["data_dir"] = vb_data_dir

    vb["dataset_ids"] = [int(dataset_numeric_id)]
    cfg["vbench_config"] = vb

    # 对齐到 VbenchDataset 入口（dataset_task 触发）
    cfg["args"]["dataset_task"] = "PHM_Vibench_basic"
    # 让 configs/config.py 自动推断 num_classes（若 metadata 可用）
    cfg["args"]["num_classes"] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_config_path.stem}_dataset_{dataset_numeric_id}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uxfd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="environment self-check")
    p_doctor.set_defaults(func=cmd_doctor)

    p_collect = sub.add_parser("collect", help="collect schema v1 runs into master csv")
    p_collect.add_argument("--roots", nargs="+", default=["save", "outputs"])
    p_collect.add_argument("--out", required=True)
    p_collect.set_defaults(func=cmd_collect)

    p_report = sub.add_parser("report", help="generate minimal markdown report from master csv")
    p_report.add_argument("--input", required=True, help="path to results_table_master.csv")
    p_report.add_argument("--out", required=True, help="output markdown path")
    p_report.add_argument("--paper", default=None, help="optional filter by paper_id")
    p_report.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="paper-aware run (supports a single run_config.yaml)")
    p_run.add_argument("--run_config", default=None, help="unified run config yaml (e.g., configs/unified_papers.yaml)")
    p_run.add_argument("--paper", default=None, help="paper_id: paper1..paper7 (overrides run_config)")
    p_run.add_argument("--modes", nargs="+", default=None, help="collect/report/llm/theory_eval/...")
    p_run.add_argument("--roots", nargs="+", default=None, help="roots to scan (for toolkit/theory)")
    p_run.add_argument("--out", default=None, help="output path for master csv (toolkit)")
    p_run.add_argument("--report_out", default=None, help="output markdown report path (toolkit)")
    p_run.add_argument("--target_paper", default=None, help="report filter paper_id (toolkit)")
    p_run.add_argument("--config", default=None, help="base config path for model papers (optional if registry has default)")
    p_run.add_argument("--datasets", nargs="+", type=int, default=None, help="vibench dataset_numeric_id list (e.g., 1 2)")
    p_run.add_argument("--seeds", nargs="+", type=int, default=None, help="seed list (e.g., 20 42 2024)")
    p_run.add_argument("--dry_run", action="store_true", help="print planned commands only")
    p_run.add_argument("--notes", default="", help="notes forwarded to legacy main.py")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
