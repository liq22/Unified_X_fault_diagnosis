# 2025-12-15：6个Executor Agent 一步步执行 Prompts（按 Paper2 schema 统一）

> 用途：把 6 篇 Paper 的执行官任务“标准化成可直接派发的 prompt”，用于跑实验→生成可复现证据链→最终产出可过最严格审稿的论文。  
> 统一约束：所有实验输出必须满足 Paper2 schema v1：`Paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md`。

---

## 全局统一约束（所有Agent必须遵守）

1) **入口参数（项目实际情况）**
- `main.py` 使用 `--config_dir <yaml>`（仓库代码真实存在该参数）。
- 注意：`main.py` 内部默认会循环 `iteration=5`（会产生 `it0..it4` 的 5 次运行目录）；这属于本项目现有行为。若只需要单次运行，请在 config/命令层面明确采用“快速/小epoch”策略，或后续由 Paper2 提供统一 wrapper（见文末“模板化命令清单”）。

2) **真实结果真源**
- `main.py` 每个 `it` 的输出目录会在 `save/task_<dataset_task>/model_<model>/..._it<it>/` 下生成，且写出 `test_result.csv`（这是生成 `metrics.json` 的一手真源）。

3) **Schema v1（必须）**
- 每个 “单数据集×单seed×单模型” 的 `<RUN_DIR>` 必须包含：
  - `<RUN_DIR>/run_meta.yaml`（`schema_version: paper2_schema_v1`）
  - `<RUN_DIR>/metrics.json`（`schema_version: paper2_schema_v1`）
- 校验命令（必须通过）：
  - `python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>`

4) **数据集命名（必须统一）**
- `run.dataset_id` 使用 Vibench Name（例如 `RM_001_CWRU`），并写 `run.dataset_numeric_id`（例如 `1`）；映射表见：`data/vibench_dataset_catalog.md`。

5) **W&B 与网络（实际约束）**
- 由于网络可能受限，建议所有Agent在运行前设置：
  - `export WANDB_MODE=offline`
  - `export WANDB_SILENT=true`

6) **重要现实约束（避免“demo当论文结果”）**
- `Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py` 与 `run_benchmark_standalone.py` 当前是**模拟/演示型评估**（含硬编码或随机生成指标），只能用于原型展示与报告格式，不可作为顶刊论文主结果真源。
- 顶刊主结果必须来自 `main.py` 的真实训练/测试输出（`test_result.csv`）以及真实解释评估流水线。

---

## Agent1 Prompt（Paper1：1D-2D Fusion）

你是 Agent1（Executor）。唯一目标：按 `Paper/1D-2D_fusion_explainable/CORE.md` 完成顶刊证据链闭环（多数据集、multi-seed、解释评估），并按 Paper2 schema v1 落盘可复现结果。

### 允许改动范围
- 只允许修改/新增 `Paper/1D-2D_fusion_explainable/` 内文件（含 `save/` 输出目录内补写 schema 文件）。

### P0（先拿到 L0 合规 run）
1) 阅读：
- `Paper/1D-2D_fusion_explainable/CORE.md`
- `Paper/1D-2D_fusion_explainable/plan/12_15/codex/AGENT_TASKS_P0.md`

2) 先跑最小数据集（CWRU，dataset_numeric_id=1）：
- 若已有本 paper 的 vbench 配置可用，优先用 vbench_config 方式；
- 如果暂时只能跑 `THU_018_basic`，也允许先跑通 schema L0（但必须在后续补 CWRU/XJTU）。

> 注意（实际修正）：`Paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml` 与 `config_XJTU.yaml` 必须做到“单数据集单run”：  
> - CWRU 配置的 `args.vbench_config.dataset_ids` 应为 `[1]`；  
> - XJTU 配置的 `args.vbench_config.dataset_ids` 应为 `[2]`；  
> 并且 `args.num_classes` 必须是整数（Fusion1D2D 模型不接受 None）。

3) 运行命令（示例）：
```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir Paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml
```

4) 在每个 `it` 的保存目录（`save/..._it<it>/`）写入：
- `run_meta.yaml`（dataset_id=RM_001_CWRU, dataset_numeric_id=1, model_id=Fusion1D2D, seed=base_seed+it；命令与config_path必须真实）
- `metrics.json`（从该目录 `test_result.csv` 提取 test accuracy 等）

5) 校验：
```bash
python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>
```

### P1（CWRU + XJTU，多seed主结果）
6) 对 XJTU（dataset_numeric_id=2, dataset_id=RM_002_XJTU）重复 P0 并得到 ≥3 seed（可用 `it0..it2` 作为 3 seed 子集，或显式控制 seed 版本）。

7) 汇总表与图（写入 paper 目录）：
- `Paper/1D-2D_fusion_explainable/manuscript/tables/table_main_results.csv`
- `Paper/1D-2D_fusion_explainable/manuscript/figures/fig_main_errorbar.png`

### P1（解释评估）
8) 解释评估必须按统一协议补齐（faithfulness/stability/efficiency），并写入 `metrics.json.explainability.*`，同时保存曲线图到 `<RUN_DIR>/artifacts/figures/`。

### P2（扩展数据集与跨域）
9) 扩展到 FEMTO(3)/IMS(4)/Ottawa23(5) 中至少1个数据集（见 `Paper/1D-2D_fusion_explainable/CORE.md` 数据集矩阵），并做跨域/失败案例解释。

### 交付验收
- 至少 CWRU+XJTU 各 3 个 schema 合规 `<RUN_DIR>`
- 主结果表+误差条图写入 manuscript
- 解释评估曲线与表可回链到 `<RUN_DIR>`

---

## Agent2 Prompt（Paper2：Explainable_FD_Toolkit / Schema Owner）

你是 Agent2（Executor + Schema Owner）。目标：确保 schema v1 在项目真实运行产物上可落地（从 `test_result.csv` 自动生成 `metrics.json`），并能汇总 6 篇 paper 的结果到 master 表，同时扩展 benchmark 覆盖更多 Vibench 数据集。

### 允许改动范围
- `Paper/Explainable_FD_Toolkit/`、`data/`（目录表更新时）

### P0（schema 真落地）
1) 阅读：
- `Paper/Explainable_FD_Toolkit/CORE.md`
- `Paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md`

2) 校验工具可用：
- `Paper/Explainable_FD_Toolkit/scripts/validate_schema.py`
- `Paper/Explainable_FD_Toolkit/scripts/collect_results_master.py`

3) 针对“项目实际输出在 save/ 下、且 main.py 默认 iteration=5”的事实：
- 明确规定：每个 `save/..._it<it>/` 目录就是一个 `<RUN_DIR>`；
- 从该 `<RUN_DIR>/test_result.csv` 生成 `metrics.json`；
- 将 dataset_id/name 统一对齐 `data/vibench_dataset_catalog.md`。

4) 说明“演示脚本≠论文真源”（必须写进 Paper2 核心文档与执行说明）：
- `run_unified_explain_eval.py` / `run_benchmark_standalone.py` 仅用于展示与格式；
- 论文主结果必须来自真实训练输出与真实解释评估流水线（并写入 schema）。

### P1（多数据集 benchmark 扩展）
4) 在 benchmark 中明确最小覆盖 + 扩展覆盖：
- 最小：CWRU(1)+XJTU(2)
- 扩展：FEMTO(3), IMS(4), Ottawa23(5), THU(6), MFPT(7), UNSW(8), SEU(9/15), DIRG(16), PU(20)…

5) 汇总 master 表：
```bash
python Paper/Explainable_FD_Toolkit/scripts/collect_results_master.py \
  --roots Paper/1D-2D_fusion_explainable/outputs Paper/MOE_explainable/outputs \
         Paper/Paper_fuzzy_XFD/outputs Paper/LLM_Explainable_FD_Toolkit/outputs \
         Paper/Neuralsymbolic_theory/outputs \
  --out Paper/Explainable_FD_Toolkit/results_table_master.csv
```
> 如果各 paper 采用 `save/` 作为 RUN_DIR，则把 roots 指向 `save/` 或在各 paper 下建立 `outputs/` 软链接到对应 `save/task_*` 子树。

### 交付验收
- schema v1 文档明确“save目录=RUN_DIR”的适配口径
- master 表可生成且包含6篇记录

---

## Agent3 Prompt（Paper3：LLM_Explainable_FD_Toolkit）

你是 Agent3（Executor）。目标：用“结构化解释→文本”的证据链对话系统，在多场景数据集上完成可复现评估（时间、正确率、主观评分、幻觉风险），并按 schema v1 落盘。

### 允许改动范围
- 只允许修改/新增 `Paper/LLM_Explainable_FD_Toolkit/`

### P0（demo + schema 合规）
1) 运行上游结构化解释（Paper2）：
```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
python Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py
```

2) 运行最小 LLM demo：
```bash
python Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/run_minimal_llm_demo.py
```

> 注意（实际情况）：`run_minimal_llm_demo.py` 当前使用 `MockDataAdapter` + `LocalTemplateLLM`，**不消费 Paper2 的真实结构化解释输出**。  
> 因此：P0 允许把它当作“离线演示/接口自检”，但**不得作为论文主结果证据链**。论文证据链必须在 P1 通过真实结构化解释（来自真实模型/解释器）来驱动对话与评测。

3) 为 demo 输出建立 `<RUN_DIR>` 并写入：
- `run_meta.yaml`（paper_id=paper3，dataset_id=RM_001_CWRU 或 RM_002_XJTU；dataset_numeric_id 对齐）
- `metrics.json`（minimum字段满足；用户研究指标通过 `artifacts` 回链到表格/原始记录）

4) 校验 schema：`validate_schema.py` 必须 OK。

### P1（用户研究最小闭环）
5) 落盘：任务集、问卷、统计计划（包含对照组：无解释/可视化/文本解释）
6) 执行最小评测并产出论文主表（不编造；不够数据就写 TODO-EXP + 复现实验命令）

### P2（多场景扩展）
7) 将案例库扩展到 Ottawa23(5)/IMS(4)/SEU(9/15) 中至少1个（更能暴露幻觉风险），形成失败案例与防护对照。

---

## Agent4 Prompt（Paper4：MOE_explainable）

你是 Agent4（Executor）。目标：完成 MoE 的多数据集（至少 CWRU+XJTU）+ 多seed稳定性 + 3/5/8专家消融 + 路由解释证据链，并按 schema v1 落盘。

### P0（3/5/8 跑通 + schema）
```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_5experts.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_8experts.yaml
```
对每个 `save/..._it<it>/` 目录补写 `run_meta.yaml/metrics.json`（从 `test_result.csv` 提取），并通过校验。

> 注意（实际缺口）：当前 `configs/PHM_Vibench/` 目录没有 `config_MoE.yaml`。  
> 要满足“CWRU+XJTU 多数据集”要求，需由 Agent4（或协调 Agent2）新增 vbench 版本配置：  
> - 基于 `configs/PHM_Vibench/config_TSPN.yaml` 复制一份到 `Paper/MOE_explainable/configs/PHM_Vibench/config_MoE.yaml`（建议放在 paper 目录内避免影响全局），  
> - 修改 `args.model: MoE`，并确保 `args.num_classes` 为整数且与 dataset 的 Label 唯一值数量一致。

### P1（稳定性与改进）
- 多seed（可用 it 子集或显式seed控制版本）统计 mean±std/CI/CV；
- 至少2种稳定性改进策略对照（初始化/路由正则/学习率调度）。

### P2（复杂数据集补强）
- 加入 IMS(4) 或 PU(20) 做“路由泛化/失败路由解释”。

---

## Agent5 Prompt（Paper5：Fuzzy-XFD）

你是 Agent5（Executor）。目标：规则可审计 + 安全关键兜底的顶刊证据链：多数据集、多seed、faithfulness/stability/sparsity/efficiency、以及2–3个高风险失败案例；按 schema v1 落盘。

### P0（先把当前口径跑实）
```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml
```
对每个 `save/..._it<it>/` 目录从 `test_result.csv` 生成 `metrics.json`，并补写 `run_meta.yaml`（dataset_id/name对齐），校验 OK。

### P1（多数据集与解释评估）
- CWRU(1)+XJTU(2) 3-seed；
- 解释评估四项必须写入 `metrics.json.explainability.*`（sparsity 必须有规则激活统计）。

> 注意（实际情况）：如要在 Vibench 上按“单数据集单run”统计，需要为 Fuzzy 的 vbench 配置拆分：
> - CWRU：`dataset_ids: [1]`
> - XJTU：`dataset_ids: [2]`
> 建议在 `Paper/Paper_fuzzy_XFD/configs/PHM_Vibench/` 内复制 `configs/PHM_Vibench/config_FuzzyLogic.yaml` 并分别改成单数据集版本。

### P1（安全关键失败案例）
- 自动导出 2–3 个高风险误判（触发规则、隶属度、证据字段、兜底建议），并可截图入论文。

### P2（扩展数据集增强兜底价值）
- 增加 THU(6)/MFPT(7)/SEU(9/15)/Ottawa23(5) 中至少1个做压力测试与失败案例补强。

---

## Agent6 Prompt（Paper6：Neuralsymbolic_theory）

你是 Agent6（Executor）。目标：命题可复现实验（重点命题2）+ 跨方法映射可运行 + 多数据集验证或边界条件反例；按 schema v1 落盘。

### P0（命题2最小版本 + schema）
```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
python Paper/Neuralsymbolic_theory/simple_validation_demo.py
```
建立 `<RUN_DIR>` 并写 `run_meta.yaml/metrics.json`；若为合成数据，dataset_id 可写 `Synthetic`，但必须在 notes 标注“P1补真实数据集”。

> 注意（schema现实约束）：schema v1 要求 `split_metrics.test.accuracy` 存在。  
> 对“命题验证”这种非分类任务，允许暂时将 `accuracy` 解释为“命题通过率/成功率”的数值，并在 `metrics.json.notes` 中明确其含义；P2 再补充更合适的指标字段与图表回链。

### P1（跨方法映射）
- 对 Paper1/4/5 各选代表机制，输出“映射验证脚本+报告”（可运行，不仅画图）。

### P2（多数据集命题泛化/反例）
- 至少 CWRU(1)+XJTU(2)，建议再加 Ottawa23(5)/IMS(4)/SEU(9/15) 作为反例或边界条件（失败要写成理论贡献）。

---

## 模板化命令清单（符合项目实际情况，给未来自动化使用）

你之前的诉求是把 6 个 prompt 再“模板化成可直接运行的命令清单”。为确保符合本项目现实约束，模板化应遵循：

1) **入口必须用 `main.py --config_dir <yaml>`**（本仓库当前不支持 `--config_file`）。  
2) **输出目录在 `save/task_<dataset_task>/model_<model>/..._it<it>/`**，且其中有 `test_result.csv` 可作为指标真源。  
3) **dataset_numeric_id 的控制方式**：通过 config 的 `vbench_config.dataset_ids: [<id>]`（例如 `[1]` 表示 CWRU），并将 `args.dataset_task` 设为 `PHM_Vibench_basic` 以触发 `VbenchDataset`。  
4) **schema 文件自动写入策略**：模板化脚本应在每个 `it` 生成的 `save/..._it<it>/` 目录内写入 `run_meta.yaml/metrics.json`，并调用 `validate_schema.py` 校验。  
5) **多 seed 的现实做法**：由于 `main.py` 默认 `iteration=5` 且 seed 会随 `it` 偏移（`seed+it`），模板化脚本应把 `it0..it2` 视作“最低3-seed子集”（或提供参数选取 it 列表）。  

当你决定启动模板化，我建议由 Paper2（Agent2）新增一个统一 wrapper（例如 `Paper/Explainable_FD_Toolkit/scripts/run_vbench_with_schema.py`）：  
- 输入：base_config.yaml、dataset_numeric_id、gpu_id、base_seed、选择的 it 列表；  
- 输出：自动生成 per-dataset config（仅改 `vbench_config.dataset_ids` 与少量标识字段）、运行训练、从 `test_result.csv` 写 schema、校验并汇总。  
这样每个 Agent 只需要改 GPU id 与 seed 列表即可批量跑通多数据集实验。
