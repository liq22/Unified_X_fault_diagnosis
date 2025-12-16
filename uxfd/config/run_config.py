from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uxfd.llm.settings import load_dotenv

RUN_CONFIG_SCHEMA_VERSION = "uxfd_run_v1"


def _expand_env_vars(obj: Any) -> Any:
    """
    递归展开配置中的环境变量（支持 ${VAR} 与 $VAR）。
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_int_list(value: Any, field_name: str) -> List[int]:
    values = _as_list(value)
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid int in {field_name}: {v!r}") from exc
    return out


def _as_str_list(value: Any) -> List[str]:
    values = _as_list(value)
    return [str(v) for v in values if str(v).strip()]


def _resolve_repo_path(repo_root: Path, path_like: Optional[str]) -> Optional[str]:
    if not path_like:
        return None
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


@dataclass(frozen=True)
class ReportConfig:
    out_csv: str = "results/results_table_master.csv"
    out_md: str = "results/uxfd_report.md"
    target_paper: Optional[str] = None


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    style: str = "standard"
    out_dir: str = "outputs/uxfd/paper3_llm_explanations"
    max_items: int = 50


@dataclass(frozen=True)
class TheoryConfig:
    out_dir: str = "outputs/uxfd/paper6_theory_eval"


@dataclass(frozen=True)
class PaperOverride:
    base_config: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    """
    统一运行配置（最小可用版本）：
    - 用一个 YAML 文件选择 paper/papers、datasets、seeds、modes
    - 保持与 legacy configs 解耦：model 训练仍以 legacy main.py 为真源
    """

    schema_version: str = RUN_CONFIG_SCHEMA_VERSION
    paper: Optional[str] = None
    papers: List[str] = field(default_factory=list)
    datasets: List[int] = field(default_factory=list)  # PHM-Vibench dataset_numeric_id
    seeds: List[int] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)

    roots: List[str] = field(default_factory=lambda: ["save", "outputs"])
    output_root: str = "outputs"

    report: ReportConfig = field(default_factory=ReportConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    theory: TheoryConfig = field(default_factory=TheoryConfig)
    paper_overrides: Dict[str, PaperOverride] = field(default_factory=dict)

    def target_papers(self) -> List[str]:
        merged = []
        if self.paper:
            merged.append(self.paper)
        merged.extend(self.papers)
        # 去重保持顺序
        seen = set()
        out = []
        for p in merged:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out


def _parse_report_config(raw: Dict[str, Any]) -> ReportConfig:
    return ReportConfig(
        out_csv=str(raw.get("out_csv", ReportConfig.out_csv)),
        out_md=str(raw.get("out_md", ReportConfig.out_md)),
        target_paper=raw.get("target_paper"),
    )


def _parse_llm_config(raw: Dict[str, Any]) -> LLMConfig:
    provider = str(raw.get("provider", LLMConfig.provider)).strip()
    # 若环境变量未展开（如 "${LLM_PRIMARY_PROVIDER}"），则回退到默认 mock
    if "$" in provider:
        provider = LLMConfig.provider
    return LLMConfig(
        provider=provider,
        style=str(raw.get("style", LLMConfig.style)),
        out_dir=str(raw.get("out_dir", LLMConfig.out_dir)),
        max_items=int(raw.get("max_items", LLMConfig.max_items)),
    )


def _parse_theory_config(raw: Dict[str, Any]) -> TheoryConfig:
    return TheoryConfig(out_dir=str(raw.get("out_dir", TheoryConfig.out_dir)))


def _parse_overrides(raw: Dict[str, Any], repo_root: Path) -> Dict[str, PaperOverride]:
    overrides: Dict[str, PaperOverride] = {}
    for paper_id, v in raw.items():
        if not isinstance(v, dict):
            continue
        base_config = _resolve_repo_path(repo_root, v.get("base_config"))
        overrides[str(paper_id)] = PaperOverride(base_config=base_config)
    return overrides


def load_run_config(path: str | Path, repo_root: Optional[Path] = None) -> Tuple[RunConfig, Dict[str, Any]]:
    """
    加载 `configs/unified_papers.yaml` 风格的统一运行配置。

    Returns:
        (RunConfig, raw_dict) 便于调用方做 debug/回写。
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load run config") from exc

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    # 允许在 YAML 中使用 ${VAR} 引用 `.env` 内变量
    load_dotenv(repo_root / ".env")
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid run config YAML (expect dict): {path}")

    raw = _expand_env_vars(raw)
    schema_version = str(raw.get("schema_version", RUN_CONFIG_SCHEMA_VERSION))
    if schema_version != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {schema_version} (expect {RUN_CONFIG_SCHEMA_VERSION})")

    paper = raw.get("paper")
    papers = _as_str_list(raw.get("papers"))
    datasets = _as_int_list(raw.get("datasets"), "datasets")
    seeds = _as_int_list(raw.get("seeds"), "seeds")
    modes = _as_str_list(raw.get("modes"))

    roots = _as_str_list(raw.get("roots")) or ["save", "outputs"]
    output_root = str(raw.get("output_root", "outputs"))

    report_raw = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    llm_raw = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    theory_raw = raw.get("theory") if isinstance(raw.get("theory"), dict) else {}

    overrides_raw = raw.get("paper_overrides") if isinstance(raw.get("paper_overrides"), dict) else {}

    cfg = RunConfig(
        schema_version=schema_version,
        paper=str(paper) if paper else None,
        papers=papers,
        datasets=datasets,
        seeds=seeds,
        modes=modes,
        roots=roots,
        output_root=output_root,
        report=_parse_report_config(report_raw),
        llm=_parse_llm_config(llm_raw),
        theory=_parse_theory_config(theory_raw),
        paper_overrides=_parse_overrides(overrides_raw, repo_root=repo_root),
    )
    return cfg, raw
