# Step 03 — 最小 schema 自测（生成 + collect + report）

## 目标
- 生成一个最小合规的 Paper2 schema v1 run（`run_meta.yaml` + `metrics.json`）。
- 验证 `uxfd collect/report` 能消费该 run 并生成 master 表/报告。

## 允许改动范围
- 允许写入 `outputs/uxfd/`（该目录已 gitignore），不允许改动代码文件。

## Actions
1) 生成自测 run：
   - `python -c "from uxfd.io.schema_v1 import RunContext, write_run_schema; from pathlib import Path; import pandas as pd; rd=Path('outputs/uxfd/_selftest_run2'); rd.mkdir(parents=True, exist_ok=True); pd.DataFrame([{'test_acc':0.5}]).to_csv(rd/'test_result.csv', index=False); ctx=RunContext(run_dir=rd, paper_id='paper1', paper_dir=Path('Paper/1D-2D_fusion_explainable'), model_id='Fusion1D2D', seed=1, dataset_id='RM_001_CWRU', dataset_numeric_id=1, command='selftest', config_path='configs/unified_baseline/config_Fusion1D2D.yaml', device='cpu'); write_run_schema(ctx, test_result_csv=rd/'test_result.csv'); print('OK')"`
2) 汇总：
   - `python -m uxfd collect --roots outputs/uxfd --out outputs/uxfd/_selftest_master.csv`
3) 报告：
   - `python -m uxfd report --input outputs/uxfd/_selftest_master.csv --out outputs/uxfd/_selftest_report.md`

## Deliverables
- 说明以下文件是否存在：
  - `outputs/uxfd/_selftest_run2/run_meta.yaml`
  - `outputs/uxfd/_selftest_run2/metrics.json`
  - `outputs/uxfd/_selftest_master.csv`
  - `outputs/uxfd/_selftest_report.md`

## 验收标准
- 上述文件全部存在，且命令无报错。

