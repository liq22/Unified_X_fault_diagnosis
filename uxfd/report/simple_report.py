from __future__ import annotations

from pathlib import Path
from typing import Optional


def _to_markdown_table(df, max_rows: int = 20) -> str:
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


def write_simple_markdown_report(
    master_csv: str | Path,
    out_md: str | Path,
    target_paper: Optional[str] = None,
) -> Path:
    """
    生成一个最小可用的 Markdown 报告：
    - 统计每个 paper/model/dataset 的 test_accuracy（若存在）
    - 作为 paper2/report 的最小落地产物（后续可替换为更严格的论文表格/图生成器）
    """

    import pandas as pd

    df = pd.read_csv(master_csv)
    if target_paper:
        df = df[df["paper_id"] == target_paper]

    lines = []
    lines.append("# UXFD Results Report (Minimal)")
    lines.append("")
    lines.append(f"- master_csv: `{Path(master_csv)}`")
    if target_paper:
        lines.append(f"- target_paper: `{target_paper}`")
    lines.append(f"- rows: {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("No rows found.")
    else:
        cols = [c for c in ["paper_id", "dataset_id", "model_id", "seed", "test_accuracy"] if c in df.columns]
        lines.append("## Rows (head)")
        lines.append("")
        lines.append(_to_markdown_table(df[cols], max_rows=20))
        lines.append("")

        if "test_accuracy" in df.columns:
            lines.append("## Summary (mean ± std)")
            lines.append("")
            grp_cols = [c for c in ["paper_id", "dataset_id", "model_id"] if c in df.columns]
            if grp_cols:
                summary = (
                    df.groupby(grp_cols)["test_accuracy"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("mean", ascending=False)
                )
                lines.append(_to_markdown_table(summary, max_rows=200))
                lines.append("")

    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
