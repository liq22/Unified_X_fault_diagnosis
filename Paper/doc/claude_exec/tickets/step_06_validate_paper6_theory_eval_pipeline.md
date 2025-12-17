# Step 06 — Paper6（Theory Eval）流水线验证

## 目标
- 消费 master 表生成理论验证产物（最小版本）：按 group 汇总表 + 命题表 +（可算则）相关性表。
- 产出 paper6 自己的 run_meta/metrics，并把表/报告写入 artifacts。

## 允许改动范围
- 允许写入 `outputs/uxfd/`；不允许修改代码文件。

## Actions
- 运行：
  - `python -m uxfd run --paper paper6 --modes theory_eval --roots outputs/uxfd --out outputs/uxfd/_p2_master.csv`

## Deliverables
- 给出生成的 `paper6 run_dir`（命令输出里会打印）
- 列出 `run_dir` 内关键文件：
  - `run_meta.yaml`, `metrics.json`
  - `artifacts/tables/summary_by_group.csv`
  - `artifacts/tables/propositions.csv`
  - `artifacts/logs/theory_eval_report.md`

## 验收标准
- schema 文件齐全；上述 tables/log 至少存在 3 个（summary/propositions/report）。

