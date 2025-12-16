from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PaperType(str, Enum):
    MODEL = "model"
    TOOLKIT = "toolkit"
    LLM_TOOLKIT = "llm-toolkit"
    THEORY = "theory"


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    paper_dir: Path
    paper_type: PaperType
    default_model_id: Optional[str] = None
    default_base_config: Optional[Path] = None


def _p(path_str: str) -> Path:
    return Path(path_str)


PAPER_REGISTRY = {
    # A) 模型论文（Model Papers）
    "paper1": PaperSpec(
        paper_id="paper1",
        paper_dir=_p("Paper/1D-2D_fusion_explainable"),
        paper_type=PaperType.MODEL,
        default_model_id="Fusion1D2D",
        default_base_config=_p("configs/unified_baseline/config_Fusion1D2D.yaml"),
    ),
    "paper4": PaperSpec(
        paper_id="paper4",
        paper_dir=_p("Paper/MOE_explainable"),
        paper_type=PaperType.MODEL,
        default_model_id="MoE",
        default_base_config=_p("configs/unified_baseline/config_MoE.yaml"),
    ),
    "paper5": PaperSpec(
        paper_id="paper5",
        paper_dir=_p("Paper/Paper_fuzzy_XFD"),
        paper_type=PaperType.MODEL,
        default_model_id="FuzzyLogicV2",
        default_base_config=_p("configs/unified_baseline/config_FuzzyLogic_v2.yaml"),
    ),
    "paper7": PaperSpec(
        paper_id="paper7",
        paper_dir=_p("Paper/TII_operator_attention"),
        paper_type=PaperType.MODEL,
        default_model_id="OperatorAttention",
        default_base_config=_p("configs/unified_baseline/config_OperatorAttention_fixed.yaml"),
    ),
    # B) 工具论文（Toolkit Paper）
    "paper2": PaperSpec(
        paper_id="paper2",
        paper_dir=_p("Paper/Explainable_FD_Toolkit"),
        paper_type=PaperType.TOOLKIT,
    ),
    # C) LLM 工具论文（LLM-Toolkit）
    "paper3": PaperSpec(
        paper_id="paper3",
        paper_dir=_p("Paper/LLM_Explainable_FD_Toolkit"),
        paper_type=PaperType.LLM_TOOLKIT,
    ),
    # D) 理论论文（Theory Paper）
    "paper6": PaperSpec(
        paper_id="paper6",
        paper_dir=_p("Paper/Neuralsymbolic_theory"),
        paper_type=PaperType.THEORY,
    ),
}

