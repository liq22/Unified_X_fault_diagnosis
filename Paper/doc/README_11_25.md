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

- `Paper/1D-2D_fusion_explainable`
- `Paper/Explainable_FD_Toolkit`
- `Paper/LLM_Explainable_FD_Toolkit`
- `Paper/MOE_explainable`
- `Paper/Paper_fuzzy_XFD`
- `Paper/Neuralsymbolic_theory`
- `Paper/TII_operator_attention`

其中每个目录都视为一篇独立论文（或系列论文）的工程与写作仓库，统一依托本项目提供的模型、数据与可解释性工具。

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

## 5. 每个子项目的 README TODO 概览

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

