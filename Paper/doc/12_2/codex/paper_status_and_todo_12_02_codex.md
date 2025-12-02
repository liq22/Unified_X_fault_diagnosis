# 7 篇 Paper 最新状态与解耦 TODO（截至 2025-12-02）

> 本文件基于 6 个智能体（GLM / Codex / Gemini 等）在 `Paper/doc/12_2/glm` 与 `Paper/doc/12_2/codex` 中的最新结果整理而成。  
> 目标：更新前 6 个 paper 的状态，并为 7 篇 paper 给出**互相解耦**的最新 TODO，便于分别交给对应的 Claude Code agent 执行。  
> 代码与文档原则：每个 paper 只改动自己目录下的文件，公共代码放在主仓库公共模块中。

---

## 总体说明

- 统一基线最新权威快照：`Paper/doc/12_2/codex/unified_baseline_results_table_12_02_v3.md`  
- 代码归属规范：`Paper/doc/11_28/codex/code_placement_guidelines_11_28.md`  
  - 公共代码：`model/`, `data/`, `trainer/`, `utils/`, `configs/`  
  - 论文专属代码与文档：各自 `Paper/<ProjectName>/` 目录  
- 7 个 paper 与目录映射：
  - 📘 1D‑2D Fusion → `Paper/1D-2D_fusion_explainable/`
  - 🟢 Explainable FD Toolkit → `Paper/Explainable_FD_Toolkit/`
  - 🟣 LLM Explainable FD Toolkit → `Paper/LLM_Explainable_FD_Toolkit/`
  - 🟠 MoE Explainable → `Paper/MOE_explainable/`
  - 🩷 Fuzzy‑XFD → `Paper/Paper_fuzzy_XFD/`
  - 🟦 Neural‑Symbolic Theory → `Paper/Neuralsymbolic_theory/`
  - 🔴 Operator Attention → `Paper/TII_operator_attention/`

---

## 1️⃣ 📘 1D‑2D Fusion Explainable

**当前状态（来自统一基线 v3 与 GLM 报告）**
- THU_018_basic 上 5 次 run：best ≈ **99.57%**，整体约 **97% ± 2%**，形状问题已彻底修复。  
- 已有：性能对比图、模态贡献热图、注意力权重图，以及较完整的 README / proposal。  
- 在 7 篇论文中，实验与文档最接近“可投稿”状态。

**解耦后的局部 TODO（仅改动 `Paper/1D-2D_fusion_explainable/`）**
1. `doc/` 中补一个「图表与实验索引」小节  
   - 列出核心图表文件（性能曲线、模态贡献热图、注意力权重图）及其路径。  
   - 为每个图表标注对应的统一基线配置（引用 `unified_baseline_results_table_12_02_v3.md` 中的行号或模型名+seed）。  
2. 在 research proposal / README 中**重新表述结果**  
   - 明确区分：`99.57%` 是 single best run，论文中的结论基于“约 97% 的平均性能 + 方差”。  
   - 增加 1 小节说明：目前仅在 THU_018_basic 验证，后续将扩展到 CWRU / XJTU。  
3. 设计但暂不必立即跑完的扩展实验小节（写在 proposal 内）  
   - 多数据集验证（CWRU, XJTU）；  
   - 消融实验：去掉 2D 模块 / 统计特征，只保留 1D；  
   - 噪声增强与工况变化下的鲁棒性测试。  

---

## 2️⃣ 🟠 MoE Explainable

**当前状态**
- 简化版 `MoE_simple` 在 THU_018_basic 上首条统一基线结果 ≈ **63.04% test acc**，训练稳定。  
- 已有：专家激活热图、路径签名、路由熵分析、部分分析文字；属于“物理约束专家系统”的概念验证。  

**解耦后的局部 TODO（仅改动 `Paper/MOE_explainable/`）**
1. 在 proposal 中新增「统一基线对齐与定位」小节  
   - 写清：当前 63% 来源于统一基线 v3 中的哪一条 MoE_simple 配置；  
   - 清晰声明定位：不是追求 SOTA 精度，而是展示“物理同构专家路由 + 可解释性”的价值。  
2. 为已有图表建立图号与说明  
   - 在 `doc/` 中整理一个表：Fig.1 专家激活热图、Fig.2 专家决策混淆矩阵、Fig.3 路由熵 / 路径签名等；  
   - 每个图 2–3 句解释其物理意义（与频段、工况或故障模式的关系）。  
3. 设计下一步实验方案（写入 proposal，但执行由后续 agent 再做）  
   - 多 seed 稳定性实验（例如 3 个种子）；  
   - 专家数消融实验（3 / 5 / 8 个专家）；  
   - 对比：无物理约束 MoE / 简单集成（例如多个 TSPN 或 Fusion1D2D 的 ensemble）。  

---

## 3️⃣ 🔴 Operator Attention（TII OperatorAttention）

**当前状态**
- 统一基线下最新快照约 **20% test/val acc**，训练稳定、无 NaN，但性能偏低。  
- 具备完善的可解释性图表：算子权重热图、权重演化、L1 正则效果图、注意力结构示意等。  
- 理论与机制设计较完整，但缺少“性能可对话”的配置与简洁实验验证。

**解耦后的局部 TODO（仅改动 `Paper/TII_operator_attention/`）**
1. 在 README / proposal 中**收紧定位**  
   - 把论文定位从“性能方法”改为“算子级注意力机制 + 强解释性理论方法”；  
   - 明确写出：当前性能仍在优化阶段，现有结果主要支撑理论与解释性。  
2. 设计一组小规模、可快速验证的实验（写入 doc）  
   - 简单合成信号（例如单频 / 双频 / 带噪声），用来展示不同算子在特定频段上的权重变化；  
   - 对比标准 Self-Attention 或卷积注意力，验证算子注意力在结构解释上的优势。  
3. 在文档中区分两类结果  
   - 「统一基线上的真实工业数据性能」（目前 20% 左右，标记为快照）；  
   - 「合成信号上的机制验证实验」（未来计划，重点支撑理论主张）。  

---

## 4️⃣ 🩷 Fuzzy‑XFD（模糊逻辑 XFD）

**当前状态**
- `FuzzyLogic_simple` 已能在统一基线配置下稳定训练，最新快照在 v3 表中约 **70.7% test acc**（比早期 ~20% 有明显提升）。  
- 已有：模糊规则框架、隶属函数设计与部分推理示例；但系统性对比与安全性案例仍不足。  

**解耦后的局部 TODO（仅改动 `Paper/Paper_fuzzy_XFD/`）**
1. 在 README / proposal 中**重新定位**  
   - 明确说明：Fuzzy‑XFD 的主要价值在于“规则可审计、安全兜底与专家知识集成”，而不是追求最高分类精度。  
2. 设计一张三行对比表（写在 doc 中）  
   - 行：Rules-only / NN-only / Fuzzy-XFD（Hybrid）；  
   - 列：准确率、可解释性评分、不确定性/拒判率、典型适用场景；  
   - 后续 agent 只需按该表补数据即可。  
3. 设计 1–2 个“高风险错误案例”  
   - 描述：某些场景下 NN 高置信度误判，而 Fuzzy 层能拒绝或给出更安全的输出；  
   - 在文档中写清应采集的指标与可视化形式（例如决策边界图、规则激活情况）。  

---

## 5️⃣ 🟢 Explainable FD Toolkit

**当前状态**
- 已实现统一接口：`SignalData`, `ExplainabilityMethod`, `ModelPlugin` 等；  
- 有 demo、可视化脚本可以从统一基线模型中提取解释；  
- 但缺少系统化的解释方法 benchmark 与工程案例。

**解耦后的局部 TODO（仅改动 `Paper/Explainable_FD_Toolkit/`）**
1. 在 doc 中定义一个「解释方法 benchmark 表头」  
   - 例如：`| 模型 | 数据集 | 解释方法 | 覆盖度 | 稳定性 | 忠实度 | 计算成本 |`；  
   - 在文字中注明：模型来自统一基线，解释方法至少包含 1 个本征和 1 个事后方法。  
2. 在 README 或一个新建的 `benchmark_plan.md` 中写明第一轮 benchmark 计划  
   - 选 2 个模型（如 TSPN, Fusion1D2D）× 2–3 个解释方法；  
   - 指定需要生成的指标和图表类型（例如热图、时域/频域贡献曲线）。  
3. 增补 1–2 个工程使用场景描述  
   - 例如：“工程师如何用本 Toolkit 对统一基线模型进行一次完整的诊断解释”；  
   - 重点是流程与接口，而非具体实现代码。  

---

## 6️⃣ 🟣 LLM Explainable FD Toolkit

**当前状态**
- LLM 流水线（结构化解释 → 模板 LLM → 对话）已打通，支持 Deepseek / GLM 等国产 LLM，成本显著降低。  
- 有多个脚本和 demo 展示自然语言解释与交互，但缺乏系统性定量评估与风险分析。

**解耦后的局部 TODO（仅改动 `Paper/LLM_Explainable_FD_Toolkit/`）**
1. 在 doc / proposal 中增加「输入解释来源」小节  
   - 明确 LLM 的输入来自 Explainable FD Toolkit 提供的结构化解释结果；  
   - 说明与统一基线模型的对应关系（例如目前主要使用 TSPN / Fusion1D2D 的解释）。  
2. 设计一个小规模用户研究或定量评估方案  
   - 例如：对比“仅看可视化图 vs 阅读 LLM 文本解释 vs 两者结合”的诊断准确率或主观评分；  
   - 在文档中列出问题设计、评价指标和统计方式，具体执行可由后续 agent 完成。  
3. 补充一节「风险与防御」  
   - 列举可能的幻觉、错误建议类型；  
   - 提出简单的安全策略（例如：限制 LLM 输出为解释 / 不给直接控制建议；通过规则过滤危险内容）。  

---

## 7️⃣ 🟦 Neural‑Symbolic Theory

**当前状态**
- 已提出四层 NeSy 框架（信号层 / 特征层 / 符号层 / 语言层），并给出 7 个子项目的层级映射。  
- 文本较长，理论形式化不够集中，与具体实验的连接有待加强。

**解耦后的局部 TODO（仅改动 `Paper/Neuralsymbolic_theory/`）**
1. 选 1–2 条典型 pipeline 做“NeSy 实例化”  
   - 例如：`Fusion1D2D → Explainable Toolkit → LLM`，或 `TSPN → MoE → Fuzzy-XFD`；  
   - 在文档中，清晰地用 NeSy 术语标注每一层对应的模块和数据流。  
2. 提出至少 1 条简单但明确的命题（可带证明草图）  
   - 例如：在特定假设下，“某类 NeSy 结构在保持解释性的前提下不会降低某种稳定性指标”；  
   - 证明可以是非形式化的，但要体现“理论论文”的风格。  
3. 加一节「与现有 NeSy 文献的关系」  
   - 对比 2–3 篇代表性工作，说明本框架在工业故障诊断场景中补齐了哪些空白。  

---

## 8️⃣ 小结：给 7 个 Claude Code agent 的执行提示

- 每个 `paper-*` agent **只在自己对应的 `Paper/<Project>/` 目录下工作**，不得修改其他 paper 的代码与文档。  
- 统一基线相关信息一律只读引用 `Paper/doc/12_2/codex/unified_baseline_results_table_12_02_v3.md`，如需更新统一表，应由 Codex 统一修改。  
- 公共代码改动（如 `model/`, `data/vbench_dataset.py`, `data/vbench_utils.py`）只能由“仓库级 Codex 任务”执行，paper agent 不直接改公共模块。  
- 对于尚未完成的实验，本文件中的 TODO 以“设计与文档准备”为主，具体跑实验可以在后续任务中按需下发。  

通过上述解耦，7 个 paper 可以在统一基线与公共框架之上**并行推进、互不干扰**，同时又保持清晰的一致性与可复现性。

