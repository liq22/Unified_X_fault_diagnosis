# 统一故障诊断项目：回顾、现状与 12‑02 TODO（Codex）

> 汇总自：  
> - `Paper/doc/12_1/papers_status_summary_12_1.md`  
> - `Paper/doc/12_1/stability_assessment_12_1.md`  
> - `Paper/doc/12_1/unified_baseline_results_updated_12_1.md`  
> - `Paper/doc/12_1/codex/todo_and_plan_12_01_codex.md`  
> - `Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`  
> - `Paper/doc/12_1/gemini/plan_20251201.md`  
> - `Paper/doc/12_1/glm/todo_and_notes_12_01_glm.md`  
> - `Paper/doc/12_2/glm/current_status_12_2.md`  
> - `Paper/doc/12_2/glm/project_status_and_todo_12_02.md`  
> - `Paper/doc/12_2/glm/todo_checklist_12_2.md`  
>
> 目的：给出一个 12‑02 时点的整体“回顾 + 现状 + TODO”快照，并将 GLM 多个状态/TODO 文档中的信息整合为 Codex 可执行的行动清单。

---

## 一、回顾：截至 12‑01 的关键进展

### 1. 统一基线 v2/v3（THU_018_basic）

- 统一基线结果已在 12‑01 更新为 v2：`Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`。  
- 综合 GLM 的 `current_status_12_2.md` 与 `project_status_and_todo_12_02.md`，当前梯度结构为：
  - **Fusion1D2D**：  
    - 报告中给出 99.57% 测试准确率，有 5 次迭代，平均约 97% 水平；  
    - 在 GLM 视角中被归为“第一梯队（业界领先）”，Codex 侧仍视为“高性能快照，需要更多复现与跨数据集验证”。  
  - **TSPN**：  
    - 报告中采用 ~92% 作为典型数值，作为透明信号处理基线；  
    - 在 GLM 文档中作为“第二梯队（实用水平）”。  
  - **MoE_simple**：  
    - 当前 best run 约 63.04%，属于“概念验证级”专家系统 baseline；  
    - GLM 计划希望在后续优化到 75% 左右。  
  - **OperatorAttention**：  
    - GLM 中给出的当前准确率约 20%，可解释性强但性能处于“待优化”状态；  
    - Codex 侧已进行 L1 正则调优（目标值 1e‑5），需要等待新一轮实验完成后更新结果表。  
  - **FuzzyLogic_simple**：  
    - 当前 baseline 约 20% 左右，训练稳定但性能较低；  
    - GLM 计划中将其定位为“需通过规则库与架构优化提升至 70%+ 的长期目标”。  

### 2. 三篇实验型 Paper 的最低可发表集

根据 `papers_status_summary_12_1.md` 与相关结果文件：

- 📘 1D‑2D Fusion Explainable  
  - 已有：性能对比图、模态贡献热图、注意力权重图、贡献度数据与分析报告；  
  - 最佳测试准确率 99.57%，5 次迭代平均约 97.16%，1 次 run 出现 NaN 但最终测试仍在 95%+；  
  - 已达到“最小可发表实验”水平。

- 🟠 MoE Explainable  
  - 已有：专家激活热图、激活分布图、路径签名可视化、路由熵分析、专家决策混淆矩阵及配套数据和分析报告；  
  - 首个 3 专家简化版实验完成（约 63% 准确率），专家专门化与路由熵结果已有数据支撑；  
  - 可作为“物理约束专家系统基线”的第一版结果。

- 🔴 TII Operator Attention  
  - 已有：算子权重热图、权重演化图、注意力机制示意图、L1 正则化效果图、性能对比图和分析报告；  
  - 当前统一基线上的性能仍偏低（约 20% 验证精度），但训练稳定、可解释性可视图已齐备。  

### 3. 稳定性与统一配置

`stability_assessment_12_1.md` 中的核心结论：

- Fusion1D2D：5 次 run，平均性能高且方差适中，仅 1 次 NaN；  
- MoE / OperatorAttention / Fuzzy：当前样本数较少，主要关注“可训练、无 NaN”这一底线，尚需更多重复实验做稳定性分析。  

统一训练配置（来自 12‑01 文档）已基本固定为：

```yaml
dataset_task: THU_018_basic
model: [ModelName]
in_dim: 4096
in_channels: 2
out_channels: 3
num_classes: 5
epochs: 100
batch_size: 64
learning_rate: 0.001
```

---

## 二、现状：12‑02 起点的综合状态

1. **技术债层面**  
   - Fusion1D2D 的 shape 问题已在 12‑01 通过调试脚本和代码修复完成；  
   - OperatorAttention 的 L1 正则化从 1e‑4 / 1e‑3 进一步调低至 1e‑5，新实验仍在进行；  
   - FuzzyLogic_simple 的训练已可以在统一基线配置下稳定运行，性能有待提升。  

2. **统一基线文档层面**  
   - 11‑29 的结果表快照：`unified_baseline_results_codex_11_29.md`；  
   - 12‑01 的更新版：`unified_baseline_results_table_12_01_v2.md`；  
   - 多篇 Paper 的 README/proposal 已添加对统一基线表的引用。  

3. **规划层面**  
   - 12‑01 的 Codex TODO 与 Gemini 计划已将 11‑29 遗留任务拆解为可执行步骤：  
     - Fusion1D2D 验证 shape 问题闭环；  
     - OperatorAttention L1 调参与短跑实验；  
     - FuzzyLogic_simple 首条 baseline；  
     - 文档与统一表的同步更新。  
   - GLM 的 notes 已对旧计划中的错误（错误配置名、旧修补建议等）做了自我校正。  

总体来说：**统一基线 v2 已成型，3 个新方法（Fusion1D2D / MoE / OperatorAttention）均达到“可实验 + 可解释”的状态，其中 Fusion1D2D 近似“可发表”，MoE 属于概念验证，OperatorAttention 与 FuzzyLogic 处于性能需优化阶段。**

---

## 三、12‑02 之后的精简 TODO（Codex 视角）

### A. 收紧统一基线表（从 “快照” 到 “v1 baseline”）

1. 为 Fusion1D2D / MoE / OperatorAttention / FuzzyLogic 补齐以下信息：  
   - 至少 3 个不同随机种子的实验结果（可以先从 Fusion1D2D 和 MoE 开始）；  
   - 在统一表中增加“平均值/标准差”行或在备注中简述稳定性；  
   - 对 Fusion1D2D 的 99.57% 做一条清晰说明：这是某次 run 的 best，整体水平约在 97% 左右。  

2. 将 `unified_baseline_results_table_12_01_v2.md` 中“进行中”的条目更新为实际结果：  
   - OperatorAttention：填入最终测试/验证精度，并在备注中写明使用的 `l1_norm`。  
   - FuzzyLogic：填入首条 baseline 的实际数值并标记为“快照”。  

> 目标：在 12‑02～12‑05 期间，让统一基线结果表从“多条快照”逐步演化为“v1 baseline”（少量模型至少有多 seed 的稳定统计）。

### B. 为 3 篇实验型 Paper 建立“图表清单 + 对应实验”索引

1. 对 1D‑2D / MoE / OperatorAttention 三篇：  
   - 在各自的 `doc/research_proposal_*.md` 中新增一个“图表索引”小节：  
     - 列出每张核心图/表的文件名、路径和对应的实验配置（链接到统一 baseline 表的某一行）。  
   - 确保每个图表都有**明确来源**：对应哪次 run、使用什么配置。  

2. 在 Codex 侧整理一份总览文档（可在之后的 12‑xx 中完成）：  
   - 例如 `Paper/doc/12_x/codex/paper_figures_index.md`，集中列出 3 篇实验型 Paper 的所有关键图表及其数据来源。  

### C. 针对 FuzzyLogic 的最小性能优化计划

1. 在 12‑02 的 TODO 中加入：  
   - 检查当前 FuzzyLogic_simple 的规则设计与参数初始化；  
   - 尝试一组简单的优化方向（例如减少参数量、调整规则数或隶属函数范围）；  
   - 用 1～2 次实验验证是否存在明显可行的第一步优化（例如从 20% 提升到 40% 左右）。  

2. 在下一份 Fuzzy‑XFD 的 progress 文档中：  
   - 记录这些尝试，哪怕结果不理想，也作为后续理论分析的反例或动机。  

### D. 文档与计划的轻量整理

> 对齐 GLM 多份 TODO 文档（`Future_TODO_List.md`, `todo_checklist_12_2.md`, `todo_tasks_12_2.md`），Codex 侧仅保留“短期确定会做”的部分，其余作为中长期 backlog。

1. 在 12‑02 之后的新文档中，统一引用：  
   - 最新统一 baseline 表：`Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`；  
   - 将旧表（11‑29 版本）仅作为历史快照保留，不再更新。  

2. 新的计划或总结文档（无论是 Codex 还是 GLM / Gemini）应：  
   - 避免重新发明配置名 / 命令行参数；  
   - 优先引用现有规范（代码归属、agent 指南、统一 baseline 文档）。  

---

## 四、12‑02 当日建议的具体三步（Codex 版）

结合 GLM 的 `QUICK_STATUS_12_2.md` 与 `todo_checklist_12_2.md`，Codex 建议将 12‑02 当日行动压缩为三件事：

1. **更新统一 baseline 表（v2 → v2.1/v3 快照）**  
   - 查看最新 OperatorAttention / FuzzyLogic run 的日志或结果文件：  
     - 如已有稳定的 test/val 精度，则将 `unified_baseline_results_table_12_01_v2.md` 中这两行的“进行中”状态更新为具体数值 + “快照”备注；  
     - 若仍在训练中，则只更新备注中的时间点与 L1 配置，不急于填精确数值。  

2. **在 3 篇实验型 Paper 中各锁定 1 张“主图”**  
   - Fusion1D2D：性能对比图或模态贡献热图；  
   - MoE：专家激活热图或路径签名可视化；  
   - OperatorAttention：算子权重热图或性能对比图。  
   - 把这些主图在对应 Paper 的 proposal 中标成“Fig.1/Fig.2”等核心图，以便写作时有明确锚点。  

3. **在 12‑02 的下一份状态报告中明确区分“baseline v1 候选 vs 快照”**  
   - 对于 Fusion1D2D / TSPN，可开始视为“baseline v1 候选”，但仍需要在报告中说明复现和交叉数据集验证计划；  
   - 对于 MoE / OperatorAttention / FuzzyLogic，继续标记为“快照级结果”，强调当前的角色主要是“概念验证 + 方法可行性说明”。  

完成上述三步后，12‑02 的工作在文档侧就有了清晰的闭环：  
- GLM 视角的多份现状/计划文档得到 Codex 汇总；  
- 统一 baseline 与三篇实验型 Paper 的关系更加清楚，有利于后续聚焦在论文写作与中期实验扩展上。  
