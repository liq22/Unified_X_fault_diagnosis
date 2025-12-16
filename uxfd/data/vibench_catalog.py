from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetCatalog:
    """
    Vibench Dataset_id -> Name 映射。

    优先从仓库内的 `data/vibench_dataset_catalog.md` 解析，避免依赖外部文件路径；
    若解析失败，则使用内置最小映射（覆盖常用数据集）。
    """

    id_to_name: Dict[int, str]

    def name_of(self, dataset_numeric_id: int) -> Optional[str]:
        return self.id_to_name.get(int(dataset_numeric_id))


def _parse_catalog_md(path: Path) -> Dict[int, str]:
    text = path.read_text(encoding="utf-8")
    mapping: Dict[int, str] = {}

    # 兼容两种：Markdown 表格 or 列表。只做稳健提取：抓取形如 "1 | RM_001_CWRU" 的行。
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        try:
            dataset_id = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        if name:
            mapping[dataset_id] = name
    return mapping


def get_default_catalog(repo_root: Path | None = None) -> DatasetCatalog:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    md_path = repo_root / "data" / "vibench_dataset_catalog.md"
    if md_path.exists():
        try:
            mapping = _parse_catalog_md(md_path)
            if mapping:
                return DatasetCatalog(mapping)
        except Exception:
            pass

    # 内置最小映射（常用）
    return DatasetCatalog(
        {
            1: "RM_001_CWRU",
            2: "RM_002_XJTU",
            3: "RM_003_FEMTO",
            4: "RM_004_IMS",
            5: "RM_005_Ottawa23",
            6: "RM_006_THU",
            7: "RM_007_MFPT",
            8: "RM_008_UNSW",
            9: "RM_009_SEU",
            14: "RM_014_THU24",
            15: "RM_015_SEU",
            16: "RM_016_DIRG",
            20: "RM_020_PU",
        }
    )

