# UXFD 主仓库整合方案（Draft, 12_15/codex）

> 目的：把 7 篇 paper 的共享方法/脚本/评估链路整合进主仓库，使仓库开箱即用，并支持“一个配置文件”切换并运行 7 篇 paper 的全流程（train/eval/explain/collect/report）。
> 说明：本文件为 **执行前方案草案**（你会在此基础上修改）；不代表已执行重构。

---

## 0) 现阶段限制与说明

- 本方案基于对仓库的**只读审计**生成（接口/参数以代码为准，严禁凭空假设）。
- 由于仓库当前存在历史技术债（尤其是 `--config_dir` / `--config_file` 混用、Vibench 配置注入不一致），整合方案将优先解决“可复现证据链”与“入口一致性”。

---

## 1) 现状审计（只读结论）

### 1.1 仓库顶层结构与关键入口

- 顶层核心：`main.py`, `main_com.py`, `configs/`, `data/`, `model/`, `trainer/`, `utils/`, `scripts/`, `script/`, `Paper/`
- 额外入口：`main_fusion.py`（1D-2D fusion 专用入口，参数风格与 `main.py` 不同）

### 1.2 真实入口参数现状（兼容性风险）

- `main.py`：只接受 `--config_dir`（不支持 `--config_file`）
- `main_com.py`：只接受 `--config_dir`（不支持 `--config_file`；且脚本里常见 `--model` 覆盖当前也不被支持）
- `main_fusion.py`：使用 `--config_file`

> 结果：文档/脚本中大量 `--config_file` 的写法与现有入口不一致，影响“开箱即用”。

### 1.3 训练输出真源与路径（可复现证据链基础）

- `configs/config.py`：固定输出到 `save/task_<dataset_task>/model_<model>/<name>/`
- `main.py`/`main_com.py`：固定写 `test_result.csv`

> 结论：短期最稳妥的统一证据链做法，是把 **`save/.../test_result.csv` 作为真源**，在同目录补齐 `run_meta.yaml`/`metrics.json`，再统一 collector 扫描汇总。

### 1.4 PHM-Vibench 配置“实际是否生效”的关键问题

- `VbenchDataset` 读取配置入口是：`args.vbench_config` 或 `args.config['vbench_config']`
- 但当前 `parse_arguments()` 只把 `config['args']` 转成 `args`，不会自动把顶层 `vbench_config:` 注入到 `args`

> 结论：大量 configs 虽然写了顶层 `vbench_config:`，但在 `main.py` 跑时**很可能被忽略**（表现为 dataset_ids/采样策略不按预期工作）。这会导致多数据集验证口径漂移，是顶刊级别的高风险点。

### 1.5 Paper2 的 “standalone benchmark” 目前为模拟数据

- `Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py`：`mock`/`np.random` + hardcode accuracy（不是真实模型输出）
- `Paper/Explainable_FD_Toolkit/scripts/run_benchmark_standalone.py`：随机生成“看起来真实”的指标（不是真实实验）

> 结论：可保留为 demo，但 **不可作为论文主结果证据链**。整合后必须提供“真实跑→真实解释→真实评估→落盘 schema→汇总报告”的链路。

---

## 2) 《重复清单》（按功能分组 + 建议唯一落点）

> 目标：把“共享能力”从 Paper 子目录/零散脚本上收到主仓库唯一实现（建议落到 `uxfd/`），Paper 目录只保留 manuscript + paper-spec + 极薄 wrapper。

| 功能组                  | 重复/分散实现（代表路径）                                                                                                                                               | 主要问题                                     | 建议唯一落点（归并后）                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Dataset adapter         | `data/vbench_dataset.py`；`Paper/1D-2D_fusion_explainable/code/utils/datasets.py`（dummy + sys.path hack）                                                          | 数据入口不统一；部分是 demo/占位实现         | `uxfd/data/`（统一 dataset_id 映射、vibench 入口、split/采样策略）                     |
| Vibench 配置注入        | 顶层 `vbench_config`（大量 configs） vs `args.vbench_config`（少量 Paper configs）                                                                                  | 配置位置不一致导致“看起来可配、实际不生效” | `uxfd/config/legacy_adapter.py`（加载时自动注入/合并）+ 后续逐步统一 YAML 写法         |
| Explainability 核心实现 | 主仓库 `explainability/*` vs `Paper/Explainable_FD_Toolkit/toolkit_integration/explainability/*`（整包重复）                                                        | 双实现必然漂移；修 bug/加协议需改两处        | `uxfd/explain/`（最终唯一实现）；短期可用 shim 兼容旧 import                           |
| LLM 接口                | `explainability/llm/*`、`Paper/Explainable_FD_Toolkit/.../llm_interface.py`、`Paper/LLM_Explainable_FD_Toolkit/code/llm_explainable_toolkit/*`                    | Provider/配置方式多套；`.env` 规范不统一   | `uxfd/llm/`（统一 settings+provider+mock；所有 key 仅 env/.env）                       |
| Explainability 评估     | `explainability/evaluation/*`；Paper2 scripts（多为模拟）                                                                                                             | 协议未与真实训练输出绑定；实现质量不一致     | `uxfd/explain/eval/` + `uxfd/metrics/`（faithfulness/stability/efficiency/sparsity） |
| 结果汇总/报告           | `configs/unified_baseline/collect_results.py`（含 simulate） vs `Paper/Explainable_FD_Toolkit/scripts/collect_results_master.py` vs 多个 `scripts/visualize_*.py` | 多套 collector/plot，口径难统一              | `uxfd/report/`（master 表 + per-paper tables/figures + repro commands）                |
| 输出 schema             | `Paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md` + `validate_schema.py`                                                                                           | 规范主要在 Paper2 目录；主仓库无唯一入口     | `uxfd/io/`（schema 读写/校验/适配 save/outputs）                                       |
| CLI/Run orchestration   | `main.py`/`main_com.py`/`main_fusion.py` + `script/*.sh` + `scripts/*.py`                                                                                     | 参数口径不一致；脚本包含入口不支持参数       | `uxfd/cli.py`（唯一新入口），旧入口保留为 legacy wrapper                               |
| Paper7 合成信号脚本     | `Paper/TII_operator_attention/code/synthetic_signals/*` 与 `Paper/TII_operator_attention/experiments/synthetic_signals/*`                                           | 同名脚本多份，难维护                         | `uxfd/experiments/synthetic/` 或 `Paper7/legacy/`（只留一套）                        |

---

## 3) 《整合方案》（执行前必须确认）

### A. 新的目录结构（目标树状图）

```text
Unified_X_fault_diagnosis/
  uxfd/                         # 主仓库唯一能力层（新增）
    config/                     # 统一配置加载/合并/校验（含 legacy 兼容）
    registry/                   # paper registry（paper_id -> spec/pipeline）
    pipelines/                  # 仅模型类：train/eval/explain 标准流水线（工具/理论类以消费 artifacts 为主）
    data/                       # 数据集适配层（Vibench/本地/映射）
    models/                     # 共享模型组件（逐步从 Paper/ 上收）
    explain/                    # 解释器封装 + explainability 评估协议
    metrics/                    # 指标计算与多seed汇总
    io/                         # schema 写入/读取 + save/outputs 适配
    report/                     # 表格/图/复现命令清单生成
    llm/                        # LLM client（env/.env；可 mock）
    theory_eval/                # 理论验证工具（消费真实 runs，产出验证表/反例集）
    cli.py                      # uxfd run/collect/report/doctor
  configs/
    uxfd_run.yaml               # 可选：极简 run 配置（非必须；推荐优先用 CLI 参数）
  Paper/
    <paperX>/
      manuscript/               # 论文正文与图表源（保留）
      paper_spec.yaml           # paper 最薄注册信息（新增/归并）
      README.md                 # 指向统一入口与证据链（更新）
      legacy/                   # 旧 scripts/code（逐步迁移/冻结）
```

### B. 7 篇 paper 分型（4类，必须区分“训练者 vs 消费者”）

> 约束：不要把所有 paper 都硬塞进 Train/Eval；工具类/理论类以“消费 run artifacts”为主。

- A. **模型论文（Model Papers）**：paper1、paper4、paper5、paper7
  - 贡献：新模型/结构/训练策略；必须产出真实 run（`save/.../test_result.csv`）并落盘 schema + artifacts。
- B. **解析/可视化工具论文（Toolkit Paper）**：paper2
  - 贡献：统一解释 API、可视化、评估协议、基准编排、collector/report；主要消费 A 类 runs。
- C. **LLM 解释工具论文（LLM-Toolkit）**：paper3
  - 贡献：把 paper2 产出的“结构化解释证据”转成可审计自然语言解释/对话；评估幻觉风险与工程指标；不以训练新模型为主。
  - LLM 可以调用 Toolkit 来进行解释
- D. **理论论文（Theory Paper）**：paper6
  - 贡献：命题/框架/统一表述；实验是验证/反例/边界条件；优先消费 A 类真实 runs 作为证据。

### C. 能力需求矩阵（分型后，边界更清晰）

| Paper / 目录                         | paper_id | Type        | Owns（唯一拥有/负责定义）                                                          | Consumes（主要消费）                                                                             | Produces（主要产物）                                                       | Needs Training?                      | uxfd 落点（唯一实现）                                                                                                                        | Paper 目录仅保留                                                                              |
| ------------------------------------ | -------- | ----------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `Paper/1D-2D_fusion_explainable`   | paper1   | Model       | 1D-2D 融合结构/对齐机制；fusion 消融设计                                           | 原始数据（通过 uxfd/data）；训练结果（`test_result.csv`）；解释 artifacts                      | 真实训练 run + 融合可解释 artifacts（对齐图/归因图）+ 论文表格/图          | Yes                                  | `uxfd/models` `uxfd/pipelines` `uxfd/explain` `uxfd/io` `uxfd/report`                                                              | `manuscript/` `paper_spec.yaml` `README.md`（复现说明）+ 最薄 wrapper（可选）           |
| `Paper/MOE_explainable`            | paper4   | Model       | MoE 专家/路由结构；路由可解释统计与消融（专家数/路由器）                           | 原始数据（uxfd/data）；训练结果；路由中间量                                                      | 真实训练 run + 路由解释 artifacts（expert weights/usage/paths）            | Yes                                  | `uxfd/models` `uxfd/pipelines` `uxfd/explain` `uxfd/io` `uxfd/report`                                                              | `manuscript/` `paper_spec.yaml` 消融说明 + 最薄 wrapper                                   |
| `Paper/Paper_fuzzy_XFD`            | paper5   | Model       | 规则/隶属度/推理机制；可审计规则解释与稀疏性协议                                   | 原始数据（uxfd/data）；训练/推理输出；规则中间量                                                 | 真实训练 run + 规则解释 artifacts（规则激活/覆盖/隶属度曲线）              | Yes                                  | `uxfd/models` `uxfd/pipelines` `uxfd/explain`（含 sparsity）`uxfd/io` `uxfd/report`                                                | `manuscript/` `paper_spec.yaml` 规则库说明（如需要）+ 最薄 wrapper                        |
| `Paper/TII_operator_attention`     | paper7   | Model       | 算子注意力结构/算子库设计；合成信号理论验证协议                                    | 原始数据（uxfd/data）；合成信号（uxfd/experiments）；训练/解释 artifacts                         | 真实训练 run + 算子权重/注意力图 + 合成验证报告                            | Yes                                  | `uxfd/models` `uxfd/pipelines` `uxfd/explain` `uxfd/report` `uxfd/io`（合成实验可放 `uxfd/theory_eval` 或 `uxfd/experiments`） | `manuscript/` `paper_spec.yaml` 理论附录/实验设计 + 最薄 wrapper                          |
| `Paper/Explainable_FD_Toolkit`     | paper2   | Toolkit     | **统一解释 API/模型适配器**；评估协议；可视化；collector/report；schema 校验 | A 类 paper 的 `run_meta.yaml/metrics.json/artifacts/`（也可兼容 `save/.../test_result.csv`） | 统一结果总表（master csv）+ per-paper/per-model 图表 + explainability 报告 | No                                   | `uxfd/explain` `uxfd/metrics` `uxfd/io` `uxfd/report` `uxfd/cli.py`                                                                | `manuscript/` `paper_spec.yaml`（协议声明）`README.md`（API 文档）+ 示例（仅调用 uxfd） |
| `Paper/LLM_Explainable_FD_Toolkit` | paper3   | LLM-Toolkit | **结构化证据→自然语言解释/对话**；幻觉/安全/成本评测协议；可审计日志        | paper2/uxfd 输出的结构化解释证据（attribution/rules/routes/metrics）                             | 解释文本/对话日志/安全评测表；LLM 工程指标（延迟/失败率）                  | No（Optional: 小规模 prompt tuning） | `uxfd/llm`（env/.env + mock）`uxfd/report` `uxfd/io`（审计日志落盘）                                                                   | `manuscript/` `paper_spec.yaml`（prompt/评测配置）`README.md`（使用说明）+ 最薄 wrapper |
| `Paper/Neuralsymbolic_theory`      | paper6   | Theory      | **理论命题/框架**；验证协议（验证/反例/边界条件）                            | A 类 paper 的真实 runs（run_meta/metrics/artifacts）；允许少量合成/小实验                        | 命题验证表、反例集、理论-实践映射图；纳入 master 表的可复现证据            | Optional（以分析/验证为主）          | `uxfd/theory_eval` `uxfd/report` `uxfd/io`                                                                                             | `manuscript/` `paper_spec.yaml`（命题与验证点）证明/附录材料 + 最薄 wrapper               |

> 强制口径（LLM）：所有 LLM key 只能来自环境变量（支持 `.env`），提供 `.env.example`；`.env` 必须 gitignore；缺 key 自动 mock，**不得中断非 LLM 流程**。

### D. 整合后的边界（逐 paper，避免职责混淆）

#### Paper1（paper1，Model）：`Paper/1D-2D_fusion_explainable`

- What stays in uxfd (唯一实现)：
  - 训练/评估编排（多 seed、多数据集、统一输出）与 legacy 入口适配
  - Fusion1D2D 模型实现与共享组件（或其主仓库 wrapper）
  - 解释算法与评估协议（归因/对齐可视化、faithfulness/stability/efficiency）
  - schema 写入/校验（`run_meta.yaml`/`metrics.json`/`artifacts/`）
  - 结果汇总与论文表格/图生成（collector/report）
- What stays in `Paper/1D-2D_fusion_explainable/`：
  - `manuscript/`、论文图表源文件与叙事材料
  - `paper_spec.yaml`（声明默认模型/解释器/消融列表）
  - 实验说明（复现命令、超参口径、注意事项）
  - 最薄 wrapper（可选）：一行调用 uxfd CLI 的脚本/Make target
- What it must NOT contain：
  - 任何新的数据加载器/collector/report 的重复实现
  - 硬编码数据路径/密钥（尤其是 LLM/W&B）
- Evidence contract：
  - Produces：真实 run（含 `test_result.csv` 真源）+ schema + `artifacts/figures|tables|logs`
  - Consumes：数据集（通过 uxfd/data），解释与评估协议（uxfd/explain）

#### Paper2（paper2，Toolkit）：`Paper/Explainable_FD_Toolkit`

- What stays in uxfd (唯一实现)：
  - 统一解释 API（模型适配器/UnifiedExplainer）与解释评估协议
  - 可视化与报告生成（统一图表样式、统计汇总、可回链证据）
  - schema 规范与校验、collector（扫描 `save/` 与 `outputs/`）
  - `doctor/collect/report` 等 CLI 子命令
- What stays in `Paper/Explainable_FD_Toolkit/`：
  - `manuscript/`（工具论文写作）
  - `paper_spec.yaml`（声明 schema 版本、评估协议口径、输出表格规范）
  - 文档与示例（仅作为 uxfd 的用法示例/薄 wrapper）
- What it must NOT contain：
  - 不得维护第二份可解释性核心实现（禁止再出现“toolkit_integration/explainability”的完整分叉）
  - 不得用模拟/随机指标替代论文主结果（demo 可保留但必须明确标注非证据链）
- Evidence contract：
  - Consumes：A 类 paper 的 `run_meta.yaml/metrics.json/artifacts`（兼容 `test_result.csv` 作为真源）
  - Produces：`results_table_master.csv` + per-paper `table_*.csv`/`fig_*.png` + explainability 报告（可回链到 run_dir）

#### Paper3（paper3，LLM-Toolkit）：`Paper/LLM_Explainable_FD_Toolkit`

- What stays in uxfd (唯一实现)：
  - LLM provider + settings（仅 env/.env；缺 key mock；不影响非 LLM 流程）
  - 结构化证据→文本解释（prompt 模板/解析器/审计日志）
  - 幻觉/安全/工程指标评测（失败率、延迟、token 成本、拒答策略）
  - 产物落盘与汇总（schema + artifacts + report）
- What stays in `Paper/LLM_Explainable_FD_Toolkit/`：
  - `manuscript/`（LLM 工具论文写作）
  - `paper_spec.yaml`（prompt 配置、评测协议、支持的 provider 列表）
  - 最薄 wrapper（可选）：一键跑“消费 structured evidence → 产出解释报告”
- What it must NOT contain：
  - 不得实现解释算法本体（IG/SHAP/规则提取/路由解释等应来自 paper2/uxfd）
  - 不得在代码/配置里硬编码任何 API key（必须来自 env/.env）
- Evidence contract：
  - Consumes：paper2/uxfd 输出的结构化解释证据（attribution/rules/routes + metrics）
  - Produces：`artifacts/prompts/*.jsonl`、`artifacts/responses/*.jsonl`、对话日志、`metrics.json`（含幻觉/成本/延迟等）

#### Paper4（paper4，Model）：`Paper/MOE_explainable`

- What stays in uxfd (唯一实现)：
  - MoE 模型/路由器/专家库的共享实现与训练编排
  - 路由解释抽取与评估（expert usage、route stability、faithfulness）
  - schema 写入/校验 + collector/report
- What stays in `Paper/MOE_explainable/`：
  - `manuscript/` + `paper_spec.yaml`（专家数/消融列表/叙事口径）
  - 实验说明（复现命令、对齐的 baseline 口径）
  - 最薄 wrapper（可选）
- What it must NOT contain：
  - 重复实现 collector/report 或 dataset 映射
  - 以“截图/口头描述”替代可回链 artifacts
- Evidence contract：
  - Produces：真实 run + 路由解释 artifacts（统计表、路径可视化）
  - Consumes：uxfd/data 与统一评估协议

#### Paper5（paper5，Model）：`Paper/Paper_fuzzy_XFD`

- What stays in uxfd (唯一实现)：
  - Fuzzy 模型、规则/隶属度/推理引擎的可复用实现
  - 规则解释与 sparsity 指标（必须进入 schema）
  - schema 写入/校验 + collector/report
- What stays in `Paper/Paper_fuzzy_XFD/`：
  - `manuscript/` + `paper_spec.yaml`（规则库/可视化规范/消融）
  - 实验说明（复现与错误案例协议）
  - 最薄 wrapper（可选）
- What it must NOT contain：
  - 第二套“可复现管理器/指标体系/汇总脚本”导致口径漂移
  - 随机/模拟结果作为主结果
- Evidence contract：
  - Produces：真实 run + 规则解释 artifacts（激活分布/覆盖率/隶属度曲线）
  - Consumes：uxfd/data 与统一 schema/collector

#### Paper6（paper6，Theory）：`Paper/Neuralsymbolic_theory`

- What stays in uxfd (唯一实现)：
  - 理论验证工具（从真实 runs 中提取证据、生成命题验证表/反例集）
  - 报告生成（把验证结果写入可投稿的 tables/figures）
  - schema/collector 对齐（理论验证也应可回链到 run_dir 或 analysis_dir）
- What stays in `Paper/Neuralsymbolic_theory/`：
  - `manuscript/`（定义/命题/证明/讨论）
  - `paper_spec.yaml`（命题列表、所需证据字段、边界条件）
  - 附录/图示材料（可选少量生成脚本，但优先上收到 uxfd/report）
- What it must NOT contain：
  - 不得承担训练入口/工具链平台的职责（不做第二套 pipeline）
  - 不得引入不可复现的“口述验证”（必须落盘与可回链）
- Evidence contract：
  - Consumes：A 类真实 runs（`run_meta.yaml/metrics.json/artifacts`）
  - Produces：命题验证表/反例集（纳入 master 表或单独 analysis runs 的 schema）

#### Paper7（paper7，Model）：`Paper/TII_operator_attention`

- What stays in uxfd (唯一实现)：
  - Operator Attention 模型与算子库（共享实现）
  - 算子权重/注意力图解释与评估协议
  - 合成信号生成与理论验证（推荐统一到 uxfd/experiments 或 uxfd/theory_eval）
  - schema 写入/校验 + collector/report
- What stays in `Paper/TII_operator_attention/`：
  - `manuscript/`（理论叙事与附录）
  - `paper_spec.yaml`（算子集合/验证实验列表/合成信号设置）
  - 最薄 wrapper（可选）
- What it must NOT contain：
  - 多份重复的合成信号脚本/算子库实现（必须单一来源）
  - 与 Paper2 的解释评估/汇总链路重复造轮子
- Evidence contract：
  - Produces：真实 run + 算子解释 artifacts（权重曲线/注意力图）+ 合成验证报告
  - Consumes：uxfd/data（真实数据）与 uxfd/experiments（合成数据）

### E. 最小落地建议（推荐方案A：CLI 选择 paper；不引入复杂 unified YAML）

**推荐 CLI 形态（设计目标）**

```bash
python -m uxfd.cli run \
  --paper paper1 \
  --datasets 1 2 \
  --seeds 20 42 2024 \
  --modes train eval explain collect report
```

**为什么比 `unified_papers_v1` 更适合当前阶段（≤3条）**

1) 避免引入大嵌套 YAML/override 复杂度，先把职责边界与证据链跑通。
2) 直接对齐仓库真实入口（`main.py --config_dir`）的历史技术债修复路径。
3) 对工具类/理论类更自然：它们只需 `--roots save outputs` 即可消费 runs，无需训练参数。

**3条示例命令（预期实现后）**

1) 运行模型论文（Paper1）在 CWRU/XJTU 多 seed：`CUDA_VISIBLE_DEVICES=0 python -m uxfd.cli run --paper paper1 --datasets 1 2 --seeds 20 42 2024 --modes train eval explain`
2) 运行工具论文（Paper2）只做汇总与出图（不训练）：`python -m uxfd.cli run --paper paper2 --modes collect report --roots save outputs --out results/results_table_master.csv`
3) 运行 LLM 工具（Paper3）消费结构化证据生成解释（缺 key 自动 mock）：
   `python -m uxfd.cli run --paper paper3 --modes llm report --input results/structured_explanations`

### F. 兼容策略（保证旧命令不被破坏）

1) 保留旧入口：`python main.py --config_dir <yaml>` 行为不变
2) 补齐参数兼容（向后兼容）：
   - `main.py` 新增 `--config_file` 作为 `--config_dir` 别名
   - `main_com.py` 同样新增 `--config_file`；并评估是否支持 `--model` 覆盖（不传则按 YAML）
3) 新增统一入口：`python -m uxfd.cli run --paper <paper_id> ...`（工具/理论类以 `collect/report/llm/theory_eval` 为主）
4) `main_fusion.py`：保持 `--config_file` 不变；后续转 wrapper 或标记 legacy

### G. 清理策略（deprecate / legacy / delete）

- 标记 deprecated（保留但不再推荐作为论文真源）：
  - `Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py`
  - `Paper/Explainable_FD_Toolkit/scripts/run_benchmark_standalone.py`
  - `configs/unified_baseline/collect_results.py` 的 `simulate_results()` 分支
- 迁移到 `legacy/`（保留历史，但不再作为主链路）：
  - `Paper/Explainable_FD_Toolkit/toolkit_integration/explainability/`
  - `script/run_PHM_*.sh`（参数/环境强耦合，且与入口不一致）
  - Paper7 合成信号脚本重复源合并，余者入 legacy
- 候选删除（最后阶段、需引用扫描与替代路径）：
  - 非官方 7 篇的历史副本目录（删除前必须给出替代路径与回滚方式）

### H. Definition of Done（验收清单，≥10项）

1) `python main.py --config_dir configs/a_018_THU/config_TSPN.yaml` 仍可运行
2) `python main.py --config_file configs/a_018_THU/config_TSPN.yaml` 可运行（新增别名兼容）
3) `python main_com.py --config_dir configs/config_com.yaml` 仍可运行
4) `python -m uxfd.cli doctor`：检查环境、数据根目录、git 状态、schema 校验器可用
5) `python -m uxfd.cli run --paper paper1 --datasets 1 2 --seeds 20 42 2024 --modes train eval explain --dry-run`：打印 dataset×seed×mode 计划
6) 任一真实 run 目录包含：`run_meta.yaml`、`metrics.json`、`artifacts/`
7) `python -m uxfd.cli run --paper paper2 --modes collect --roots save outputs --out results/results_table_master.csv` 生成 master 表
8) `python -m uxfd.cli run --paper paper2 --modes report --in results/results_table_master.csv --target_paper paper1` 生成论文表/图（csv+png/pdf）与复现命令清单
9) `.env.example` 存在；`.env` 在 `.gitignore`；LLM key 缺失时自动 mock（不崩）
10) collector 能同时扫描 `save/` 与 `outputs/` 并统一到同一 schema
11) manuscript 引用数字可回链到 `runs/<run_id>/`（run_meta 含 config 快照与 git commit）
12) 7 篇 paper 都可通过 `python -m uxfd.cli run --paper <paper_id> ...` 跑通至少 L0（schema minimum；工具/理论类以消费 runs 为主）

### I. 风险与回滚

| 风险点                            | 影响                     | 缓解                                                         | 回滚方式                            |
| --------------------------------- | ------------------------ | ------------------------------------------------------------ | ----------------------------------- |
| explainability 双实现迁移         | import 断裂/漂移         | 先 shim：旧路径 re-export 新实现                             | 保留旧目录 +`git revert` 单步提交 |
| 入口参数修复（`--config_file`） | 行为变化                 | 只做 alias，不改默认                                         | `git revert`                      |
| vbench_config 注入修复            | 可能改变“实际用数据集” | 写入 run_meta（dirty + config快照）；提供 strict_legacy 开关 | 增加开关回退                        |
| 输出目录统一（save vs outputs）   | 重复文件/磁盘占用        | 过渡期先“写 schema 到 save”，逐步迁移                      | 保留 save 不动，仅新增 outputs      |
| W&B 默认行为                      | 离线/无 key 报错         | 默认 `WANDB_MODE=offline`，可禁用                          | 环境变量切换 + revert               |

---

## 4) 确认区块（蓝图层）

---

✅ 请确认是否采用此“分型后的矩阵+边界”作为主仓库整合蓝图：

- 同意：回复 “同意”
- 修改：回复 “不同意” 并指出要改的 paper_id 或矩阵列

---
