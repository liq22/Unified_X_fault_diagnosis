# Step 01 — 基础自检（doctor + 语法）

## 目标
- 验证 `uxfd` CLI 的最小可用性：`python -m uxfd doctor` 正常运行。
- 验证关键模块无语法错误：`py_compile` 通过。

## 允许改动范围
- **本工单不允许修改任何文件**（只允许只读检查与运行安全命令）。

## 禁止
- 任何联网操作（curl/wget/pip/conda/git fetch 等）
- 读取/输出敏感文件（`.env`、`secrets/`、`~/.ssh/*` 等）

## Actions（Bash 白名单内）
1) 运行：
   - `python -m uxfd doctor`
2) 运行：
   - `python -m py_compile uxfd/cli.py uxfd/io/schema_v1.py uxfd/explain/protocol_v1.py uxfd/pipelines/paper2_toolkit.py uxfd/pipelines/paper3_llm.py uxfd/pipelines/paper6_theory.py`

## Deliverables
- 贴出命令输出（截取关键部分即可：doctor 输出键值 + py_compile 是否报错）。

## 验收标准
- 两条命令均无 Traceback/语法错误。

