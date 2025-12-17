# Step 02 — 统一配置 dry-run 计划生成

## 目标
- 验证 `configs/unified_papers.yaml` 可被 `uxfd` 解析。
- 验证 dry-run 输出包含：模型论文训练计划 + paper2/paper3/paper6 消费型计划。

## 允许改动范围
- **不允许修改任何文件**（只运行命令与只读检查）。

## Actions
- 运行：
  - `python -m uxfd run --run_config configs/unified_papers.yaml --dry_run`

## Deliverables
- 输出中必须能看到：
  - `main.py --config_file ...generated_configs/... --seed ...` 的多条 `[PLAN]`（datasets×seeds）
  - `collect_results_master ...`、`write_simple_markdown_report ...`
  - `paper3 llm_explain ...`
  - `paper6 theory_eval ...`

## 验收标准
- 命令成功退出（exit code 0），且上述 4 类计划均出现。

