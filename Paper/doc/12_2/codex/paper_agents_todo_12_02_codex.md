# 7 个 Paper 子项目 TODO 解耦任务列表（2025‑12‑02）

> 目的：将当前统一基线与整体计划中的 TODO 解耦为 7 组 **互不混淆的任务**，分别交给 7 个 `paper-*` Claude Code agent 执行。  
> 对应关系：  
> - 📘 `paper-1d2d-fusion` → `Paper/1D-2D_fusion_explainable/`  
> - 🟢 `paper-explainable-toolkit` → `Paper/Explainable_FD_Toolkit/`  
> - 🟣 `paper-llm-interface` → `Paper/LLM_Explainable_FD_Toolkit/`  
> - 🟠 `paper-moe-expert` → `Paper/MOE_explainable/`  
> - 🩷 `paper-fuzzy-logic` → `Paper/Paper_fuzzy_XFD/`  
> - 🟦 `paper-neuralsymbolic` → `Paper/Neuralsymbolic_theory/`  
> - 🔴 `paper-operator-attention` → `Paper/TII_operator_attention/`  

每个任务只涉及一个子项目目录 + 已存在的统一 baseline 文档，不再交叉修改其他 Paper 代码或文档。

---

## 任务 1：📘 1D‑2D Fusion 子项目 TODO（paper-1d2d-fusion）

**作用范围**：`Paper/1D-2D_fusion_explainable/`  

**子任务**：
1. 在 `doc/research_proposal_*.md` 中新增或完善“图表索引”小节：  
   - 列出当前已存在的三张核心图：  
     - `performance_comparison.png`（性能曲线）  
     - `contribution_heatmap.png`（模态贡献热图）  
     - `attention_weights.png`（注意力权重可视化）  
   - 为每张图注明：关联的实验配置（统一基线行）、大致实验编号或日志路径。  
2. 在 proposal 的“结果与讨论”小节补充文字描述：  
   - 用自然语言概括 5 次 run 的平均性能与方差；  
   - 明确指出 99.57% 是某次 best run，“总体水平 ~97%”作为更稳健的结论。  
3. 检查 README 中对统一 baseline 的引用：  
   - 确认指向最新统一表（目前为 `Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md`）；  
   - 若仍指向旧版本（11‑29），更新为 12‑1 v2。  

---

## 任务 2：🟢 Explainable FD Toolkit 子项目 TODO（paper-explainable-toolkit）

**作用范围**：`Paper/Explainable_FD_Toolkit/`  

**子任务**：
1. 在 Toolkit 的 `doc` 中新增“统一基线解释 benchmark 规划”小节：  
   - 定义一个最小 benchmark 表头，例如：  
     ```markdown
     | 模型 | 数据集 | 解释方法 | 覆盖度 | 稳定性 | 忠实度 | 备注 |
     ```  
   - 明确：  
     - 模型取自统一 baseline（TSPN / Fusion1D2D / MoE 等）；  
     - 解释方法至少包含 1 个本征 + 1 个事后方法。  
2. 在 `scripts/run_unified_explain_eval.py`（或现有脚本）中：  
   - 增加对统一 baseline 表的轻量引用（例如在注释中指出“默认使用 unified_baseline_results_table_12_01_v2.md 中的模型配置”）；  
   - 不修改任何训练逻辑，只补充文档与注释。  
3. 在 README 中添加一句统一说明：  
   - “本工具的性能对比与可解释性实验默认基于统一 baseline 配置，详见统一基线结果表。”  

---

## 任务 3：🟣 LLM_Explainable_FD_Toolkit 子项目 TODO（paper-llm-interface）

**作用范围**：`Paper/LLM_Explainable_FD_Toolkit/`  

**子任务**：
1. 在 `doc` 中为 LLM 实验增加“输入解释来源”说明：  
   - 明确 LLM 解释实验使用的模型输出和解释结果来自统一 baseline（例如 TSPN 和 Fusion1D2D 的解释）；  
   - 指出结构化解释字段与 LLM 输入字段的映射（可以引用 Toolkit 的 `SignalData` / explanation schema）。  
2. 在 `experiments/scripts` 下选择一个现有 LLM 流水线脚本（如 `test_unified_llm_pipeline_stub.py`）：  
   - 补充注释：说明此脚本默认假定解释结果与统一 baseline 实验一致；  
   - 不新增训练/调用，仅提升说明清晰度。  
3. 在 README 中添加统一 baseline 引用句：  
   - 说明性能层面的比较全部参考统一 baseline 结果表，本项目聚焦“解释质量和交互体验”的差异。  

---

## 任务 4：🟠 MoE Explainable 子项目 TODO（paper-moe-expert）

**作用范围**：`Paper/MOE_explainable/`  

**子任务**：
1. 在 MoE 的 proposal 中增加“统一 baseline 对齐”小节：  
   - 说明当前 63.04% 是 THU_018_basic 场景下的首个 MoE baseline，来自统一 baseline 表中的某一行；  
   - 指出后续计划：多 seed 实验、专家数消融、跨数据集验证等。  
2. 在 `results/` 下已有的图表（专家激活热图、路径签名、路由熵分析）中：  
   - 为每张图在 proposal 中指定图号与简短解读（例如 Fig.2: Expert Activation Heatmap）。  
3. 在 README 或 `README_scripts.md` 中：  
   - 明确说明如何从统一 baseline 配置运行 MoE_simple 实验（指向 `config_MoE.yaml` 和 `main.py` 调用方式）；  
   - 强调“统一 baseline 版本采用 MoE_simple 实现，复杂版本 MoE 视为扩展实验”。  

---

## 任务 5：🩷 Fuzzy-XFD 子项目 TODO（paper-fuzzy-logic）

**作用范围**：`Paper/Paper_fuzzy_XFD/`  

**子任务**：
1. 在 Fuzzy-XFD 的 proposal 中增加“当前统一 baseline 结果”描述：  
   - 用文字说明当前简单 Fuzzy baseline 约 20% 准确率，仅作为“最初原型”；  
   - 将“目标性能”（如 70%+）写成长期优化目标而非当前结论。  
2. 为 Fuzzy-XFD 设计一个最小性能优化实验方案（仅文档层面）：  
   - 在 `doc` 中写出一组可以优先尝试的方向（如规则库精简、隶属函数参数学习、与 TSPN 特征结合等）；  
   - 不在本任务中编写或运行代码，只定义实验设计。  
3. 在 README 中引用统一 baseline 表，并说明：  
   - “当前 Fuzzy baseline 结果属于快照，用于展示方法可行性；后续版本将进一步提升性能。”  

---

## 任务 6：🟦 Neuralsymbolic Theory 子项目 TODO（paper-neuralsymbolic）

**作用范围**：`Paper/Neuralsymbolic_theory/`  

**子任务**：
1. 在 `doc` 中更新 NeSy 映射表：  
   - 为 7 个子项目补充一列“当前统一 baseline 角色”（例如：TSPN = 基线，Fusion1D2D = 方法 A，MoE = 方法 B 等）；  
   - 用一两句解释各项目在 NeSy 四层结构中的重点层级（信号/特征/符号/语言）。  
2. 为“理论示例”选择 1–2 个具体 pipeline：  
   - 如 “TSPN → MoE → Fuzzy” 或 “Fusion1D2D → Toolkit → LLM”，在 doc 中画出 NeSy 四层映射草图；  
   - 只做理论表示，不新增代码。  
3. 在 README 中补充一句：  
   - “统一 baseline 结果为 NeSy 实证部分提供了数值背景，本项目主要从理论层面解释这些方法之间的关系。”  

---

## 任务 7：🔴 Operator Attention 子项目 TODO（paper-operator-attention）

**作用范围**：`Paper/TII_operator_attention/`  

**子任务**：
1. 在 `doc/Operator_Attention_Theory_Analysis.md` 或相关文档中：  
   - 增加一节“统一 baseline 实验对照”，说明：  
     - 当前 OA-TSPN 在 THU_018_basic 上的性能仍处于概念验证阶段（约 20%）；  
     - 其主要贡献在于算子级可解释性与复杂度优势。  
2. 在 `results/` 中已有的算子权重图和 L1 正则化效果图：  
   - 在文档中为这些图指定图号和简短解读，强调算子权重与物理算子之间的联系。  
3. 在 README 中引用统一 baseline 表：  
   - 明确 OperatorAttention 行目前为“快照/待优化”，并说明后续提升性能的方向（扩展算子库、调整注意力结构等）。  

---

## 使用方式建议

当你在 Claude Code 中调度某个 `paper-*` agent 时，可以直接引用本文件中的对应任务段，例如：

- 调用 `paper-1d2d-fusion` 时：  
  - “请按照 `Paper/doc/12_2/codex/paper_agents_todo_12_02_codex.md` 中 **任务 1** 的要求，更新 1D‑2D 子项目的 proposal 与 README。”  

这样每个 agent 都有清晰的、与统一 baseline 兼容的独立 TODO，不会互相踩边或重复工作。  

