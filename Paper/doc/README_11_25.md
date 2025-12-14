# 2024-11-25：Paper 子项目规划与 README 标准

本说明文件用于统一 `Paper/` 目录下各个论文子项目的规划与文档结构，确保它们都**以本主仓库为基础**，在 README 中清晰说明：要解决的问题、研究内容、技术路线、预期结果、讨论与 TODO。

---

## 1. 目标与原则

- 明确：每个子项目是如何在本仓库的统一框架上扩展的。  
- 统一：每篇论文的 README 结构，方便自己和合作者快速理解与维护。  
- 可追踪：在一个文件中集中记录各子项目当前状态与 TODO。

---

## 2. 子项目总览（@Paper 子目录）

当前重点维护的 7 个论文子项目如下（路径即为实际目录）：

- 1 `Paper/1D-2D_fusion_explainable`
- 2 `Paper/Explainable_FD_Toolkit`
- 3 `Paper/LLM_Explainable_FD_Toolkit`
- 4 `Paper/MOE_explainable`
- 5 `Paper/Paper_fuzzy_XFD`
- 6 `Paper/Neuralsymbolic_theory`
- 7 `Paper/TII_operator_attention`

其中每个目录都视为一篇独立论文（或系列论文）的工程与写作仓库，统一依托本项目提供的模型、数据与可解释性工具。

### 2.1 最新状态快照（按本文 Paper 顺序对齐）

**状态快照日期**：2025-12-04（主要参考 `Paper/doc/12_4/glm/` 与 `Paper/doc/12_3/` 的阶段性复盘文档）  
**重要约定**：后续所有“Paper编号”以本文为准；其他文档若出现编号不一致，请以**目录路径**作为唯一ID对齐。

| 序号 | 目录 | 当前定位 | 快照完成度 | 关键结果（快照） | 最近优先 TODO（7天内） |
|---:|---|---|---:|---|---|
| 1 | `Paper/1D-2D_fusion_explainable` | 多模态融合可解释 | ~95% | Fusion1D2D 准确率 99.57%（统一基线SOTA） | 3-seed 稳定性；多数据集泛化（CWRU/XJTU/THU_006）；投稿材料格式化 |
| 2 | `Paper/Explainable_FD_Toolkit` | 统一可解释性OS/基准 | ~80–85% | 统一接口+指标体系+基准评估框架已成型 | 补齐剩余模型适配（Fuzzy/OperatorAttention）；一键benchmark复现；2个工业demo+对比Captum/SHAP/LIME |
| 3 | `Paper/LLM_Explainable_FD_Toolkit` | 自然语言解释与交互 | ~75% | Deepseek/GLM集成；成本下降（文档口径 60–80%） | 解释质量评估标准；用户研究设计；幻觉/安全防护与UI落地 |
| 4 | `Paper/MOE_explainable` | 物理同构MoE专家系统 | ~85% | 已有可解释路由分析；但多文档间准确率/参数量口径需统一 | 复现最佳配置（含seed20路径）；3/5/8专家消融；训练稳定性改进（CV下降）与统一结果表输出 |
| 5 | `Paper/Paper_fuzzy_XFD` | 规则可审计轻量模型 | ~90% | FuzzyLogic 20%→70.7%（+253.7%）；参数 7.6K（口径一致） | 冲击75%+；补齐高风险错误案例（安全兜底）；规则/隶属度可视化与稳定性统计 |
| 6 | `Paper/Neuralsymbolic_theory` | NeSy统一理论框架 | ~70–85% | 四层架构+命题体系基本成型；命题2仍需增强证据链 | 命题2实验重做与图表更新；增加2–3个案例闭环；论文草稿整合成可投版本 |
| 7 | `Paper/TII_operator_attention` | 算子级注意力（理论主导） | ~60% | 理论框架与合成信号验证方案已设计；工业数据性能仅概念验证（~20%） | 立即跑合成信号验证并产图；完善定理证明附录；明确“理论贡献优先于准确率”的论文叙事 |

### 2.2 12_3 / 12_4 复盘文档索引（建议按目录对齐阅读）

- 总路线图/投稿时间线：`Paper/doc/12_3/codex/papers_roadmap_7_projects_12_03_codex.md`
- 总览/统一进展：`Paper/doc/12_4/glm/unified_project_review_12_04.md`
- 全局待办与风险：`Paper/doc/12_4/glm/project_status_and_todo_12_04.md`
- UXFD项目总评：`Paper/doc/12_4/glm/UXFD_project_review_status_todo_12_04_2025.md`
- Paper 1（融合）：优先参考总览与统一进展（同上）
- Paper 2（Toolkit）：`Paper/doc/12_4/glm/explainable_fd_toolkit_review_status_todo_12_04.md`、`Paper/doc/12_4/glm/Paper2_Status_Review_2025-12-03.md`
- Paper 3（LLM）：优先参考总览与全局待办（同上）
- Paper 4（MoE）：`Paper/doc/12_4/glm/moe_project_comprehensive_review_12_4.md`
- Paper 5（Fuzzy）：优先参考总览与全局待办（同上）
- Paper 6（NeSy）：`Paper/doc/12_4/glm/neuralsymbolic_theory_review_status_todo_12_04_2025.md`
- Paper 7（OperatorAttention）：`Paper/doc/12_4/glm/operator_attention_review_and_todo_12_04.md`、`Paper/doc/12_3/codex/summary_and_todo_12_03_codex.md`

### 2.3 全局P0（影响所有Paper的共性阻塞）

- 数据路径/数据集可用性：`THU_018` 路径在部分文档中被标记为阻塞项，需先统一数据根目录与 `configs/` 口径。
- 统一结果表真源：建议固定“统一基线结果表”的生成脚本与输出文件，所有论文/文档引用同一份表（避免准确率/参数量口径漂移）。

---

## 3. 统一 README 结构规范

每个子项目的 `README.md` 需要包含并**显式分段**说明下面六个部分（可以用二级或三级标题表示）：

1. **要解决的问题（Problem）**  
   - 这篇论文针对什么痛点、空白或局限？  
   - 与本主仓库已有工作相比，新贡献在哪些方面？

2. **研究内容（Research Content）**  
   - 主要研究对象与场景（例如：轴承、旋转机械、跨工况等）。  
   - 关键科学问题和子问题。  

3. **技术路线（Technical Route）**  
   - 总体框架图或文字描述。  
   - 如何基于本仓库中的模型 / 模块 / 数据进行扩展：  
     - 使用了哪些模型（如 TSPN、NNSPN、MoE、Operator Attention 等）；  
     - 使用了哪些数据集和配置文件；  
     - 如何与 `Explainable_FD_Toolkit` 或 LLM 工具集对接（如有）。

4. **预期论文中展示的结果（Expected Results in Paper）**  
   - 计划在论文中展示的主要实验表格和图（可以列出占位名称）。  
   - 对比基线、消融实验、大致性能预期或假设验证方式。

5. **讨论（Discussion）**  
   - 当前已知的风险与不足（如：数据规模、泛化性、可解释性度量难点）。  
   - 可能的拓展方向与后续工作设想。

6. **TODO（Roadmap / Checklist）**  
   - 用任务清单的形式列出下一步工作（数据整理、模型实现、实验、写作等）。  
   - 建议使用勾选列表形式，例如：  
     - [ ] 完成基础实验配置  
     - [ ] 跑通主结果  
     - [ ] 补充消融实验  
     - [ ] 完成初稿撰写

> 说明：已有 README 中的内容可以保留，但需要按照上述结构做一次整理与补充，使关键信息一眼可见。

---

## 4. 执行计划（Plan）

1. 在本文件中统一记录各子项目及 README 规范。  
2. 为每个子项目设计/补充符合规范的 README 结构骨架。  
3. 逐个更新以下路径的 README（如不存在则新建）：  
   - `Paper/1D-2D_fusion_explainable/README.md`  
   - `Paper/Explainable_FD_Toolkit/README.md`  
   - `Paper/LLM_Explainable_FD_Toolkit/README.md`  
   - `Paper/MOE_explainable/README.md`  
   - `Paper/Paper_fuzzy_XFD/README.md`  
   - `Paper/Neuralsymbolic_theory/README.md`  
   - `Paper/TII_operator_attention/README.md`  
4. 检查各 README 与主仓库顶层 `readme.md`、`Paper/doc/README.md` 的描述是否一致，并在需要时微调措辞。

---

## 5. 三层架构总览与角色定位

为避免不同论文在定位与代码实现上相互“抢地盘”、重复造轮子，本项目采用**三层解耦架构**来统一 7 篇 Paper 的关系与分工。

### 5.1 三层解耦架构

**第 1 层：信号处理基础设施层（Infrastructure）**  
- 统一基础设施（所有提案共享）：  
  - 信号处理原语库：TSPN 基础类、NNSPN 接口、特征提取标准接口。  
  - 数据管道与评估：统一数据格式、标准化预处理、通用评估指标。  
  - 实验框架：配置管理系统、实验跟踪协议（如 W&B）、结果可视化约定。  
- 主要代码位置：  
  - 主仓库根目录（`main.py`, `configs/`, `model/`, `trainer/`, `explainability/` 等）。  
  - `Paper/Explainable_FD_Toolkit`（作为可解释性“操作系统”，提供统一 API 和评估协议）。

**第 2 层：方法论研究层（Methods）**  
- 每篇方法论文在不重造基础设施的前提下，通过标准接口使用第 1 层的能力，专注于自身核心创新：  
  - 融合与对齐机制（1D-2D）  
  - 物理同构路由与专家结构（MoE）  
  - 规则-神经混合（Fuzzy-XFD）  
  - 算子级注意力（Operator Attention）  
- 这一层只定义清晰的**输入/输出接口**和理论/算法贡献，不再宣称要“统一平台”。

**第 3 层：应用集成与交互层（Applications & Integration）**  
- 聚焦领域特定应用和工具链整合：  
  - 可解释故障诊断工具集（Explainable_FD_Toolkit）在这里作为“可解释性 OS”的一部分。  
  - LLM_Explainable_FD_Toolkit 作为**自然语言接口专家**，利用第 1 层的标准 API 输出。  
- 该层不提出新的底层算法，而是组合与编排前两层的能力，为工程使用与展示服务。

**Neuralsymbolic_theory** 跨三层提供统一的理论框架和概念体系，对上述层次与方法给出形式化描述与指导原则。

### 5.2 7 篇 Paper 的角色重新定位

结合上述三层架构，7 个子项目在本仓库中的角色建议如下：

- `Paper/1D-2D_fusion_explainable`  
  - **角色**：多模态信号对齐与融合专家（方法层）。  
  - 重点：1D 时序信号 ↔ 2D 时频表示的对齐机制、跨模态特征学习与融合策略。  
  - 解耦：不再承担“统一平台”职责，只通过标准接口消费基础设施与工具集。

- `Paper/Explainable_FD_Toolkit`  
  - **角色**：可解释性“操作系统”（基础设施/应用边界层）。  
  - 重点：统一可解释性 API、标准化评估协议和可视化方式。  
  - 定位：不与方法论文竞争“创新点”，而是为所有方法提供稳定支撑。

- `Paper/LLM_Explainable_FD_Toolkit`  
  - **角色**：自然语言接口与对话专家（应用集成层）。  
  - 重点：基于 Explainable_FD_Toolkit 和主仓库模型的输出，做领域特定 Prompt 设计、对话管理和解释生成。  
  - 边界：只消费标准 API，不反向定义底层可解释方法。

- `Paper/MOE_explainable`  
  - **角色**：基于物理的专家路由专家（方法层）。  
  - 重点：统计特征驱动的专家路由决策、物理同构专家设计、路径签名与专家分工可解释性。  
  - 区分：与注意力机制（如 Operator Attention）在“路径级决策 vs 算子级加权”上的本质差异。

- `Paper/Paper_fuzzy_XFD`  
  - **角色**：规则–神经混合专家（方法层）。  
  - 重点：工业规则维护与渐进更新场景，将现有规则与可解释特征/深度模型结合。  
  - 应用：适合需要显式规则库、人机协同优化的应用环境。

- `Paper/Neuralsymbolic_theory`  
  - **角色**：神经-符号一体化理论提供者（跨层）。  
  - 重点：为所有方法定义统一的抽象层次、形式化可解释性概念与集成约束。  
  - 输出：提供设计指南、理论分析与跨方法对比框架。

- `Paper/TII_operator_attention`  
  - **角色**：信号处理算子注意力机制专家（方法层 + 理论层）。  
  - 重点：算子级注意力数学基础、与标准注意力的理论区分、适配信号处理的 Operator Attention 设计与分析。  
  - 联动：可与 MoE、1D-2D 融合一起使用，但保持独立理论贡献。

上述角色定位建议在各自的 README 与 research proposal 中逐步对齐，避免范围重复与职责混淆。

---

## 6. 每个子项目的 README TODO 概览

### `Paper/1D-2D_fusion_explainable`
- [ ] 补充「要解决的问题」与「研究内容」概述，强调 1D-2D 融合与可解释性。  
- [ ] 梳理当前架构与特征对齐理论，形成清晰「技术路线」。  
- [ ] 列出计划展示的主要结果（性能表、对齐可视化、解释性案例等）。  
- [ ] 添加「讨论」与「TODO」清单。

### `Paper/Explainable_FD_Toolkit`
- [ ] 明确工具集要解决的跨模型可解释性与工程落地问题。  
- [ ] 总结与主仓库模型的对接方式，完善技术路线。  
- [ ] 列出预期展示的案例（不同模型的统一解释界面、指标对比等）。  
- [ ] 添加讨论与 TODO。

### `Paper/LLM_Explainable_FD_Toolkit`
- [ ] 聚焦「LLM 如何增强故障诊断可解释性」的问题描述。  
- [ ] 说明信号处理层 → 知识层 → LLM 层的整合技术路线。  
- [ ] 规划论文中展示的对话案例、评测指标与对比实验。  
- [ ] 补充讨论与 TODO。

### `Paper/MOE_explainable`
- [ ] 用简洁语言总结「物理同构 MoE」要解决的核心问题。  
- [ ] 梳理专家模块、路由器与解释模块的整体技术路线。  
- [ ] 罗列预期结果（性能提升、路径签名、专家激活图等）。  
- [ ] 添加讨论与 TODO。

### `Paper/Paper_fuzzy_XFD`
- [ ] 明确模糊逻辑在故障诊断可解释性中的角色与问题定位。  
- [ ] 设计模糊规则 + 神经特征的技术路线说明。  
- [ ] 规划预期实验（规则可解释性、性能对比等）。  
- [ ] 添加讨论与 TODO。

### `Paper/Neuralsymbolic_theory`
- [ ] 从理论层面梳理主仓库与各子项目要统一解决的「神经-符号一体化」问题。  
- [ ] 制定理论框架与数学形式的技术路线。  
- [ ] 规划论文中展示的示意图、公式推导与案例链接。  
- [ ] 添加讨论与 TODO。

### `Paper/TII_operator_attention`
- [ ] 在 README 中凝练 Operator Attention 想要解决的核心问题。  
- [ ] 结合现有理论推导文档，整理为读者友好的技术路线说明。  
- [ ] 规划预期结果展示（注意力可视化、复杂度分析、对比实验等）。  
- [ ] 添加讨论与 TODO。

---

后续在完善各个子项目 README 时，可直接参照本文件中的规范与 TODO，逐项勾选与更新，并在需要时扩展新的子项目条目。  
