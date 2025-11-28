# 7 个 Paper 子 Agent 执行规范（2025-11-28）

本文件用于指导 Claude Code 中的 7 个 `paper-*` 子 Agent 在本仓库中的行为，确保指令清晰、边界明确、无歧义，便于后续自动化协同。

---

## 一、所有 paper-* 子 Agent 的通用约束

### 1. 目录作用域

- 每个子 Agent 只能主动修改**自己对应的 Paper 子目录**，以及少量显式允许的统一配置文件。  
- 对应关系：  
  - `paper-1d2d-fusion` → `Paper/1D-2D_fusion_explainable/`  
  - `paper-explainable-toolkit` → `Paper/Explainable_FD_Toolkit/`  
  - `paper-llm-interface` → `Paper/LLM_Explainable_FD_Toolkit/`  
  - `paper-moe-expert` → `Paper/MOE_explainable/`  
  - `paper-fuzzy-logic` → `Paper/Paper_fuzzy_XFD/`  
  - `paper-neuralsymbolic` → `Paper/Neuralsymbolic_theory/`  
  - `paper-operator-attention` → `Paper/TII_operator_attention/`  

- 统一配置例外（仅在任务明确说明时允许修改）：  
  - `configs/unified_baseline/config_*.yaml` 中与本方法直接相关的配置。  
- 禁止事项：  
  - 不允许随意修改 `main.py`、`trainer/`、`model/` 等核心公共文件，除非上游“baseline / integration”任务显式要求。  
  - 不允许修改其他 Paper 子项目目录。  

### 2. 任务优先级

每个 paper Agent 默认按以下顺序执行任务（在没有特殊说明时遵循）：

1. 保证**最小测试脚本**存在且注释清晰（不要求实际运行成功）。  
2. 保证 README / proposal 中的“问题、创新点、技术路线、预期结果、讨论、TODO”六部分齐全且互相一致。  
3. 为实验准备好**配置文件和运行脚本**（可以是命令模板），但不强行发起长时间训练。  
4. 将已有实验结果整理进对应 `doc/results/` 或 `results/` 的说明文档（不伪造数据）。  

### 3. 行为约束

- 可以做的事情：  
  - 新增或优化本子项目的脚本、配置、文档、目录说明；  
  - 修复本方法专有代码的“小范围 bug”（例如 simple 版本模型的初始化）；  
  - 为后续实验预留文档结构、图表占位、命令模板。

- 不可以做的事情：  
  - 大范围重构公共代码；  
  - 擅自改动其他 paper 子目录；  
  - 引入新的第三方依赖（仅允许使用 `environment.yml` 中已有依赖）。  

### 4. 文档规范

- 所有新增/修改的说明文档必须为**中文**。  
- 每个 Paper 子项目的 README 中必须包含以下六个部分（可以是显式小节，内容要完整）：  
  1. 要解决的问题  
  2. 研究内容  
  3. 技术路线  
  4. 预期论文中展示的结果（明确到图/表类型）  
  5. 讨论  
  6. TODO（下一步如何优化框架和实验）  

---

## 二、`paper-1d2d-fusion` Agent 指令（1D-2D_fusion_explainable）

### 1. 作用范围

- 允许修改目录：`Paper/1D-2D_fusion_explainable/`  
- 可只读但非必须修改：`configs/unified_baseline/config_Fusion1D2D.yaml`  

### 2. 必做事项（按顺序执行）

1. **检查并维护最小测试脚本**  
   - 确认并在必要时更新：  
     - `Paper/1D-2D_fusion_explainable/scripts/test_unified_fusion1d2d_identity_fix.py`  
   - 要求：  
     - 注释清晰说明用途：验证 `model/Fusion1D2D.Fusion1D2D` 是否能在统一接口下完成一次前向；  
     - 输入输出形状与 `main.py` 中调用一致，只做结构检查，不启动训练。  

2. **整理 README 贡献与实验对应关系**  
   - 在 `Paper/1D-2D_fusion_explainable/README.md` 中：  
     - 确认“⭐ 主要创新点（Contributions）”三条与 research proposal 中的实验/图表逐条对应。  
   - 在 `Paper/1D-2D_fusion_explainable/doc/research_proposal_*.md` 中：  
     - 为每个创新点显式列出需要的 Table 和 Figure 名称（例如 `Table 1`, `Figure 2`），只写结构与标题，不填数据。  

3. **准备统一基线相关脚本**  
   - 若不存在，则新增：  
     - `Paper/1D-2D_fusion_explainable/scripts/run_unified_fusion_baseline.py`  
   - 要求：  
     - 内部给出调用 `python main.py --config_dir configs/unified_baseline/config_Fusion1D2D.yaml` 的命令模板；  
     - 用注释说明：  
       - 建议输出目录；  
       - 需要在哪个结果文档中记录（例如统一 baseline 对比表）。  

4. **结果文档占位**  
   - 在 `Paper/1D-2D_fusion_explainable/doc/` 下新增或补充：  
     - 例如 `fusion_baseline_results_plan.md`，只包括章节结构：  
       - 实验设置  
       - 基线对比表（表头说明）  
       - 可视化图列表（名称与用途）  
     - 不填写虚构实验结果。  

---

## 三、`paper-explainable-toolkit` Agent 指令（Explainable_FD_Toolkit）

### 1. 作用范围

- 允许修改目录：`Paper/Explainable_FD_Toolkit/`  

### 2. 必做事项

1. **统一接口文档**  
   - 在 `Paper/Explainable_FD_Toolkit/README.md` 与 `doc` 中：  
     - 确保以下三类接口有清晰定义和代码示例：  
       - `SignalData`  
       - `ExplainabilityMethod`  
       - `ModelPlugin`  
     - 每个接口至少包含：字段说明 + 一个最小使用示例（伪代码即可）。  

2. **设计最小基准评测规范**  
   - 在 `Paper/Explainable_FD_Toolkit/doc/` 下新增或更新：  
     - `explainability_benchmark_plan.md`（命名可微调，但需语义清晰）；  
   - 文档中要明确：  
     - 至少 3 个模型（如 TSPN / Resnet / 任意一个新方法）；  
     - 至少 2 种解释方法（本征 + 事后）；  
     - 至少 3 个指标（覆盖度、稳定性、忠实度等）；  
     - 给出统一的对比表表头（不填数据）。  

3. **准备评估脚本模板**  
   - 若不存在，新增：`Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py`  
   - 要求：  
     - 提供命令行参数（模型列表、解释方法列表、输出目录）；  
     - 内部只构建调用命令或伪代码，不实际发起训练/大规模评估；  
     - 注释指明：  
       - 结果文件命名规则；  
       - 建议将评估结果写入哪个文档（例如 `results/explainability_benchmark.md`）。  

---

## 四、`paper-llm-interface` Agent 指令（LLM_Explainable_FD_Toolkit）

### 1. 作用范围

- 允许修改目录：`Paper/LLM_Explainable_FD_Toolkit/`  

### 2. 必做事项

1. **对齐 README 贡献与系统架构**  
   - 在 `README.md` 中：  
     - 检查“主要创新点”三条是否与系统架构图和实验设计部分一一对应；  
   - 在 `doc` 中新增或完善一个映射表：  
     - 列出 Explainable_FD_Toolkit 的输出字段 → LLM 输入字段的对应关系。  

2. **准备对话评估计划**  
   - 在 `Paper/LLM_Explainable_FD_Toolkit/doc/` 下新增：  
     - `llm_eval_plan.md`（或等价名称）  
   - 文档应明确：  
     - 三种对话模式：  
       - 无结构化上下文；  
       - 使用结构化解释；  
       - 结构化解释 + 知识图谱/规则约束；  
     - 每种模式需要记录的指标：  
       - 可理解性评分；  
       - 诊断准确性；  
       - 幻觉/错误率。  

3. **最小流水线 stub**  
   - 若不存在，新增：`Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/test_unified_llm_pipeline_stub.py`  
   - 要求：  
     - 输入：一个结构化解释样例文件（例如 JSON/CSV，路径通过参数传入）；  
     - 输出：一个“模拟 LLM 输出”的文本文件（可以是固定模板生成）；  
     - 注释中明确：  
       - 将来接入真实 LLM 时，需要替换的函数或模块位置。  

---

## 五、`paper-moe-expert` Agent 指令（MOE_explainable）

### 1. 作用范围

- 允许修改目录：`Paper/MOE_explainable/`  
- 可只读：`model/MoE_simple.py`、`configs/unified_baseline/config_MoE*.yaml`  

### 2. 必做事项

1. **统一 MoE_simple 使用说明**  
   - 在 `Paper/MOE_explainable/README.md` 中：  
     - 明确说明：统一 baseline 默认使用 `model.MoE_simple.MoEModel`；  
   - 在 `doc` 中增加一个小节：  
     - 对比 full MoE 与 simple MoE 的定位：论文写作建议使用哪个作为主结果，另一个作为 ablation。  

2. **维护最小测试脚本**  
   - 确保存在且注释清晰：  
     - `Paper/MOE_explainable/scripts/test_unified_moe_simple_init.py`  
   - 要求：  
     - 仅进行模型构造与一次随机前向，检查输出形状；  
     - 注释写明与 `main.py` 中构造方式一致，参数取自统一 baseline 假定值。  

3. **规划专家解释图表**  
   - 在 `Paper/MOE_explainable/doc/research_proposal_*.md` 中：  
     - 列出计划展示的图表：专家激活热力图、路径签名可视化等；  
     - 每张图对应一条具体创新点（例如路径签名图对应“路径级可解释性”），只写标题与说明，不填数值。  

---

## 六、`paper-fuzzy-logic` Agent 指令（Paper_fuzzy_XFD）

### 1. 作用范围

- 允许修改目录：`Paper/Paper_fuzzy_XFD/`  

### 2. 必做事项

1. **校正“主要贡献”表述**  
   - 在 `Paper/Paper_fuzzy_XFD/README.md` 中：  
     - 确保“1.1 主要贡献”三条与当前 Fuzzy-XFD 实现对齐：  
       - 规则–深度统一框架；  
       - 规则库渐进更新机制；  
       - 高风险/分布漂移场景的兜底策略。  

2. **准备三模态对比实验说明**  
   - 在 `doc/research_proposal_*.md` 中新增小节“Rules vs NN vs Fuzzy-XFD”：  
     - 设计对比表头：模型 × 准确率 × 可解释性评分 × 不确定性/拒绝率；  
     - 每一列用一句话说明如何衡量（例如可解释性评分由专家打分）。  

3. **最小规则可视化脚本**  
   - 若不存在，新增：`Paper/Paper_fuzzy_XFD/scripts/plot_rule_activation_example.py`  
   - 要求：  
     - 即使没有真实数据，也用虚构矩阵演示如何绘制规则激活热力图；  
     - 注释说明：此脚本主要用于“论文图示 demo”，实际实验应替换为真实数据。  

---

## 七、`paper-neuralsymbolic` Agent 指令（Neuralsymbolic_theory）

### 1. 作用范围

- 允许修改目录：`Paper/Neuralsymbolic_theory/`  

### 2. 必做事项

1. **维护统一映射表**  
   - 在 `Paper/Neuralsymbolic_theory/doc/` 中新增或更新：  
     - `neusy_mapping_7papers.md`（名称可微调，但需表示“7 个 paper 的 NeSy 映射”）；  
   - 表格要求：  
     - 行：7 个子项目；  
     - 列：信号层 / 特征层 / 符号层 / 语言层中涉及的主要对象或“未涉及”；  
     - 每个格子必须填具体描述，禁止写“同上”或空白。  

2. **形式化约束草案**  
   - 在 research proposal 中新增小节“示例约束”：  
     - 至少给出 2 个约束的形式化表达（如算子稀疏性、规则一致性）；  
     - 对每个约束补一句“在训练/设计中如何落地”（例如通过 L1 正则、结构投影等）。  

---

## 八、`paper-operator-attention` Agent 指令（TII_operator_attention）

### 1. 作用范围

- 允许修改目录：`Paper/TII_operator_attention/`  
- 可只读：`model/operator_attention.py`、`model/TSPN_OperatorAttention.py`  

### 2. 必做事项

1. **保证 OA 封装与主仓库一致**  
   - 文档中引用 OperatorAttention 网络时：  
     - 统一使用 `main.py` 中的 `OperatorAttentionNetwork` 名称；  
     - 不再使用已废弃或与代码不符的类名。  

2. **维护最小测试脚本**  
   - 确保存在且注释清晰：  
     - `Paper/TII_operator_attention/scripts/test_unified_operator_attention_init.py`  
   - 要求：  
     - 说明其用途是检验 `OperatorAttentionNetwork` 是否能在统一接口下完成一次前向；  
     - 注明依赖：`SimpleOperatorAttention` + `OperatorLibrary`。  

3. **规划核心对比图表**  
   - 在 `Paper/TII_operator_attention/doc/` 下（例如 `Operator_Attention_Theory_Analysis.md` 或附加文档）中：  
     - 列出至少三张关键图/表：  
       - 性能与复杂度对比图（Self-Attention vs Operator Attention）；  
       - 算子权重热力图（FFT/HT/WF/I 权重随样本/故障类型变化）；  
       - 长序列复杂度曲线图（序列长度 vs 时间/显存）。  
     - 为每个图/表指定图号或表号，并写一句说明对应哪条创新点。  

---

本文件仅定义“子 Agent 在各自 Paper 子项目中的职责与边界”，不触发具体训练或大规模实验。后续调度 Agent 时，应在任务描述中引用对应小节（例如：“请 `paper-moe-expert` 按《claude_agents_instructions_11_28.md》中第五节的指令，更新 README 与测试脚本。”）。 

