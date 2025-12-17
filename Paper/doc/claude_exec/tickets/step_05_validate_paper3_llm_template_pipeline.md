# Step 05 — Paper3（LLM Toolkit, template）流水线验证

## 目标
- 消费 master 表，生成可审计解释文本（template/mock，不依赖网络）。
- 产出 paper3 自己的 run_meta/metrics，并在 artifacts/logs 里保存 explanation + evidence。

## 允许改动范围
- 允许写入 `outputs/uxfd/`；不允许修改代码文件。

## Actions
- 运行：
  - `python -m uxfd run --paper paper3 --modes llm --roots outputs/uxfd --out outputs/uxfd/_p2_master.csv`

## Deliverables
- 给出生成的 `paper3 run_dir`（命令输出里会打印）
- 列出 `run_dir` 内关键文件：
  - `run_meta.yaml`, `metrics.json`
  - `artifacts/logs/*_explanation.md`
  - `artifacts/logs/*_evidence.json`

## 验收标准
- schema 文件齐全，且至少生成 1 组 explanation+evidence。

