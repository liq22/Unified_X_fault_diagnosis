# Step 04 — Paper2（Toolkit）流水线验证

## 目标
- 运行 paper2 的 `collect + report` 流水线，并写入 Paper2 schema v1（paper2 自己也要产出 run_meta/metrics）。
- 验证 paper2 的 run 目录内 `artifacts/` **自包含** master/report 副本（证据链）。

## 允许改动范围
- 允许写入 `outputs/uxfd/` 与 `results/`（若存在）；不允许修改代码文件。

## Actions
- 运行：
  - `python -m uxfd run --paper paper2 --modes collect report --roots outputs/uxfd --out outputs/uxfd/_p2_master.csv --report_out outputs/uxfd/_p2_report.md`

## Deliverables
- 给出生成的 `paper2 run_dir`（命令输出里会打印）
- 列出 `run_dir` 内关键文件：
  - `run_meta.yaml`, `metrics.json`
  - `artifacts/tables/*.csv`（应包含 master 副本）
  - `artifacts/logs/*.md`（应包含 report 副本）

## 验收标准
- run_dir 存在且 schema 文件齐全；artifacts 内包含 master/report 的副本。

