# Step 07 — Schema 扫描校验（全量 runs）

## 目标
- 扫描 `save/` 与 `outputs/` 中所有 `run_meta.yaml`，逐个运行 `uxfd.io.validate.validate_run_dir`。
- 输出通过数量与失败列表（若有），为后续修复准备明确工单。

## 允许改动范围
- **不允许修改任何文件**（只读扫描 + 运行 Python）。

## Actions
- 运行（单行脚本）：
  - `python -c "from pathlib import Path; from uxfd.io.validate import validate_run_dir; roots=[Path('save'),Path('outputs')]; bad=[]; ok=0; \nfor root in roots:\n  if not root.exists():\n    continue\n  for p in root.rglob('run_meta.yaml'):\n    passed, errs = validate_run_dir(p.parent)\n    ok += int(passed)\n    if not passed:\n      bad.append((str(p.parent), errs))\nprint('ok_runs', ok); print('bad_runs', len(bad)); \n[print('\\nRUN',d,'\\n', '\\n'.join(e)) for d,e in bad[:10]]"` 

## Deliverables
- 输出 `ok_runs` 与 `bad_runs`
- 若 `bad_runs>0`：列出前 10 个失败 run_dir 与错误内容

## 验收标准
- `bad_runs == 0`；否则进入“fix 工单”。

