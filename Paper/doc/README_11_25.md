# Paper 子项目整体关系说明

本 `Unified_X_fault_diagnosis` 仓库是一个**统一的、可解释的故障诊断方法与实验平台**。  
`Paper/` 目录下的各个论文子项目，都是在该平台的基础上，围绕「可解释故障诊断（Explainable FD）」这一主线展开的不同方向延伸研究。

下文以 7 篇正在撰写的论文为主线，说明它们与主仓库以及彼此之间的关系：

- `Paper/1D-2D_fusion_explainable`
- `Paper/Explainable_FD_Toolkit`
- `Paper/LLM_Explainable_FD_Toolkit`
- `Paper/MOE_explainable`
- `Paper/Paper_fuzzy_XFD`
- `Paper/Neuralsymbolic_theory`
- `Paper/TII_operator_attention`

---

## 一、统一代码平台的核心作用

1. **模型与方法统一**  
   - 提供透明信号处理网络（TSPN / NNSPN 等）、算子注意力（Operator Attention）、以及多种可解释特征提取方法。  
   - 论文中的大部分方法（1D-2D 融合、MoE、模糊推理、神经符号等）都在此基础上实现或复用。

2. **数据与实验统一**  
   - 通过 `configs/`、`data/`、`trainer/` 等模块，统一管理数据集（如 THU、CWRU 等）、训练流程和评估指标。  
   - 各论文子项目共享相同的数据处理和训练脚本，只在各自的 `Paper/**/scripts` 中封装具体实验配置。

3. **可解释性与工具统一**  
   - `explainability/` 与相关工具代码，为多个论文提供统一的可解释性接口（特征归因、路径分析、可视化等）。  
   - Toolkit 类论文（Explainable_FD_Toolkit、LLM_Explainable_FD_Toolkit）直接在该基础上进行工程化封装和扩展。

简而言之：**主仓库是「方法 + 实验平台」，Paper 下的各论文是「不同角度的学术输出」**。

---

## 二、7 篇论文各自的侧重点与定位

### 1. `1D-2D_fusion_explainable`：多模态 1D-2D 融合可解释诊断

- **基础依托**：复用主仓库中的透明信号处理网络和可解释特征提取模块，扩展为同时处理 1D 时序信号与 2D 频谱图的融合架构。  
- **研究重点**：  
  - 如何设计早期 / 中期 / 渐进式融合结构，在提升性能的同时保持决策过程可解释。  
  - 「物理-语义-几何」三层特征对齐理论，与主仓库中的信号处理算子和频谱特征形成理论闭环。  
- **与其他论文关系**：  
  - 在方法层面，为后续的 MoE、模糊、神经符号等提供多模态特征基础。  
  - 在工具层面，可通过 Explainable_FD_Toolkit 和 LLM_Explainable_FD_Toolkit 对融合模型进行统一解释和对话分析。

### 2. `Explainable_FD_Toolkit`：工程化可解释故障诊断工具集

- **基础依托**：  
  - 直接基于主仓库的模型实现与 `explainability/` 组件，将它们封装为可复用工具包。  
  - `toolkit_integration/` 中的代码与主仓库形成一一映射关系，是「从研究代码到工程工具」的桥梁。  
- **研究 / 工程重点**：  
  - 统一不同可解释方法（本征 + 事后）的接口、可视化与评估流程。  
  - 为各篇方法论文提供统一实验基线与可解释性评测环境。  
- **与其他论文关系**：  
  - 为 1D-2D 融合、MoE、模糊、神经符号等模型提供通用的解释接口和可视化前端。  
  - 是 LLM_Explainable_FD_Toolkit 的直接前置工程基础。

### 3. `LLM_Explainable_FD_Toolkit`：LLM 增强的可解释诊断工具集

- **基础依托**：  
  - 在 Explainable_FD_Toolkit 的基础上，结合主仓库中的信号处理、模型输出和知识图谱，接入大语言模型（LLM）。  
  - 利用已有的故障知识表示与可解释特征，将其转化为自然语言解释和交互式对话。  
- **研究重点**：  
  - 将频谱分析、特征归因等低层信息映射为高层自然语言解释。  
  - 融合领域知识图谱与 LLM，实现更加专业、上下文感知的诊断建议。  
- **与其他论文关系**：  
  - 对所有方法类论文（1D-2D 融合、MoE、模糊、神经符号等）的实验结果进行「统一语言层解释」。  
  - 在理论层面支撑 Neuralsymbolic_theory：为「神经 + 符号 + 语言」一体化提供应用示例。

### 4. `MOE_explainable`：基于物理机理约束的 MoE 可解释模型

- **基础依托**：  
  - 继承主仓库中透明信号处理网络（如 NNSPN）的设计思想，将传统信号处理算子实例化为专家模块。  
  - 训练与评估完全依托主仓库的数据与训练框架，实验结果可在 Explainable_FD_Toolkit 中统一可视化和评测。  
- **研究重点**：  
  - 将物理算子封装为专家网络 + 统计特征驱动的路由器，实现「物理同构」且内在可解释的深度模型。  
  - 建立从「路径签名」到「单样本归因」的完整解释链路。  
- **与其他论文关系**：  
  - 在方法层与 Operator Attention、神经符号和模糊推理共同构成「可解释结构设计」分支。  
  - 其物理同构思想可被 Neuralsymbolic_theory 进一步抽象为统一理论框架。

### 5. `Paper_fuzzy_XFD`：模糊逻辑驱动的可解释故障诊断（占位/规划中）

- **基础依托**：  
  - 计划复用主仓库中的可解释特征与诊断结果，将其映射到模糊规则和模糊推理系统。  
  - 训练数据和评测流程与其他论文共享，便于进行对比实验。  
- **研究重点（规划方向）**：  
  - 将「连续特征 + 模糊规则」结合，构建具有明确语义的决策边界。  
  - 探索模糊规则与神经网络特征之间的协同学习与互相约束。  
- **与其他论文关系**：  
  - 在方法上与 MoE、Operator Attention 一起扩展主仓库的可解释模型族群。  
  - 与 Neuralsymbolic_theory 在「符号 / 规则」层面存在天然衔接。

### 6. `Neuralsymbolic_theory`：神经-符号一体化可解释故障诊断理论

- **基础依托**：  
  - 从主仓库和前述各论文提炼共性结构——透明信号处理、物理约束、规则与知识图谱、LLM 解释等。  
  - 试图在理论层面给出「神经网络 + 符号知识 + 语言解释」的一体化框架。  
- **研究重点（规划中）**：  
  - 形式化描述透明信号处理网络、MoE、模糊规则和知识图谱之间的关系。  
  - 探索如何在统一框架下度量和优化「可解释性、鲁棒性与性能」之间的平衡。  
- **与其他论文关系**：  
  - 是一个**上层理论总结**：对 1D-2D 融合、MoE、模糊、Toolkit、LLM 等工作的统一抽象与理论化。  
  - 为今后的扩展工作（更多模型或工具）提供理论坐标系。

### 7. `TII_operator_attention`：Operator Attention 机制与透明算子选择

- **基础依托**：  
  - 与主仓库中的透明信号处理网络紧密相关，将注意力机制与具体的信号处理算子绑定。  
  - 其数学推导和理论分析直接服务于主仓库及其他论文中使用的「算子选择 / 路由」模块。  
- **研究重点**：  
  - 提出 Operator Attention 的数学形式与复杂度分析，强调可解释的算子选择过程。  
  - 将注意力权重与实际物理算子行为建立一一对应，增强模型可解释性。  
- **与其他论文关系**：  
  - 在结构上可视为 MoE_explainable 和 1D-2D_fusion_explainable 中「算子级路由 / 权重分配」的理论基础之一。  
  - 为 Neuralsymbolic_theory 提供「可解释注意力」的形式化案例。

---

## 三、整体视角下的关系小结

可以将整个体系按照「层次」理解为：

1. **基础平台层**：  
   - `Unified_X_fault_diagnosis` 主仓库  
   - 提供模型、训练、数据与基础可解释性组件，是所有论文的共同实验与实现基座。

2. **方法创新层（结构与算法）**：  
   - `1D-2D_fusion_explainable`：多模态 1D-2D 融合与对齐  
   - `MOE_explainable`：物理同构的专家路由架构  
   - `Paper_fuzzy_XFD`：模糊规则与可解释决策  
   - `TII_operator_attention`：算子注意力与可解释权重分配  

3. **工具与应用层（工程落地）**：  
   - `Explainable_FD_Toolkit`：统一的可解释故障诊断工具集  
   - `LLM_Explainable_FD_Toolkit`：在工具集基础上引入 LLM，实现自然语言与对话式解释  

4. **理论抽象层**：  
   - `Neuralsymbolic_theory`：对上述方法与工具的神经-符号一体化理论总结与抽象。

从时间和依赖关系上看，大致可以理解为：

> **主仓库（统一平台） → 方法创新（1D-2D / MoE / 模糊 / Operator Attention）  
> → 工具集与 LLM 增强（Explainable_FD_Toolkit / LLM_Explainable_FD_Toolkit）  
> → 理论抽象（Neuralsymbolic_theory）**

如需后续扩展新的论文方向，可以继续放在 `Paper/` 目录下，并在本说明中补充其与上述工作及主仓库之间的关系。

好，那我这次直接把它写成「7 个论文子项目的立项书模版」，每个子项目都从 **核心问题 / 研究内容 / 技术路线 / 近期里程碑 / 与其他项目接口** 五个角度写清楚。你后面写开题、写 introduction & section 2–3 的时候，基本可以直接改写成论文文本。

---

# Paper 子项目整体路线小结（简版）

统一背景不重复铺太长，只保留你做决策需要的那几句话：

* 主仓库 `Unified_X_fault_diagnosis` = **统一的可解释故障诊断平台**：
  算子网络（TSPN/NNSPN/TFON/TKAN）、数据+训练管线、基础解释接口。
* 我们的整体路线 ≈ InterpretML + Captum 在 FD 场景下的「机械振动 + 神经符号」版本：

  * InterpretML = 玻璃盒模型 + 黑盒解释的统一框架
  * Captum = PyTorch 下统一的 attribution + 评估 + 可视化库
* 7 篇论文分别落在：**结构方法、事后工具、LLM 解释、模糊/规则、MoE、operator attention 和总理论**，共同围绕振动/声学 XAI FD 这个方向。

下面是每篇的「可直接用来指导推进」版本。

---

## 1. `1D-2D_fusion_explainable`

### 多模态 1D–2D 融合可解释诊断

#### 1. 核心问题

* 如何在**1D 振动时序**与**2D 时频/谱图**的融合架构中：

  1. 在多工况、变工况下显著提升故障诊断性能；
  2. 同时保持「**物理可解释**」：解释清楚哪个频带/时间段/模式在起作用。
* 如何构建一个**统一的 1D–2D 融合解释框架**，能够回答：

  > *「某个故障模式在时域长什么样，在时频/谱域又长什么样，两者如何对齐？」*

#### 2. 研究内容（论文视角）

* 提出一族**多模态融合网络**（早期、中期、渐进式），针对振动 + 时频图。
* 设计 1D–2D 对齐的解释指标：

  * 时域关键片段 vs 时频关键区域的一致性；
  * 不同模态对同一故障特征的贡献度分解。
* 在多数据集（THU、CWRU 等）上做系统对比实验：单模态 vs 多模态 vs 多模态+可解释约束。
* 给出 2–3 组典型 case study：展示「物理–语义–几何」三层对齐。

#### 3. 技术路线

1. **模型设计**

   * 1D 分支：TSPN/TFON 风格的透明算子网络；
   * 2D 分支：对 STFT/CWT 图使用轻量 CNN/ViT；
   * 融合策略：

     * 早期融合：1D→变换→拼接→共享 backbone；
     * 中期融合：分别编码→特征拼接/注意力融合；
     * 渐进式融合：多层 cross-attention / gating。

2. **解释模块设计**

   * 内在解释：

     * 1D 分支：算子路径、频带权重、时域窗口重要性；
     * 2D 分支：时频块/频带的重要性热图。
   * 事后解释：

     * 用 Captum 风格的 Integrated Gradients/Saliency 做统一 attribution。

3. **评价体系**

   * 性能：准确率、F1、跨工况泛化；
   * 解释性：

     * 与理论故障频率、调制侧带的一致性；
     * 1D–2D 融合解释的互补度（如 2D 将 1D 冲击解释为调制频带）。

#### 4. 近期里程碑（1–2 个月）

* [ ] 选定 1 个主数据集（如 THU_018）+ 1 个对比数据集（CWRU）。
* [ ] 在现有 TSPN 基础上实现 2D 分支 + 至少 2 种融合结构（早期 + 中期）。
* [ ] 接好 `explainability/` 中的基础 attribution，跑通 1 个完整实验 pipeline（训练 + 解释 + 可视化）。
* [ ] 形成 2–3 个典型样本的解释图（1D+2D+路径），写出 1–2 页方法 + case 的初稿总结。

#### 5. 与其他项目接口

* 输出：

  * 为 `MOE_explainable` 提供「多模态专家」结构；
  * 为 Toolkit/LLM 提供多模态解释样例与中间表示。
* 输入：

  * 使用 `Explainable_FD_Toolkit` 的统一解释 API 和评估指标；
  * 使用 `TII_operator_attention` 的 operator attention 机制做融合层路由。

---

## 2. `Explainable_FD_Toolkit`

### 工程化事后可解释故障诊断工具集（neural symbolic 骨干）

#### 1. 核心问题

* 如何在 FD 场景下构建一个类似 InterpretML/Captum 的**统一 XAI 工具集**：

  * 同时支持玻璃盒结构（TSPN/NNSPN/TFON）和黑盒模型；
  * 提供统一 API + 可视化 + 评估指标；
  * 用真实振动/声学案例证明「以 neural symbolic system 为骨干」优于纯黑盒 XAI。

#### 2. 研究内容

* 工具集设计：

  * 一致的 Python API（类 InterpretML 的「统一入口」）；
  * 针对 FD 的特化可视化（频谱/时频 + attribution）。
* 方法集成：

  * 内在解释：算子路径、operator attention、MoE 专家开关；
  * 事后解释：Captum 的 gradient/perturbation 方法封装。
* 体系化实验：

  * 对比「纯黑盒模型 + 通用 SHAP/LIME」与「neural-symbolic 结构 + Toolkit」的性能 & 解释质量；
  * 输出一组 benchmark 结果，类似已有 XAI-FD 基准工作。

#### 3. 技术路线

1. **核心 API & 数据结构**

   * `UnifiedExplainer(model, method='intrinsic'/'ig'/...)`
   * `Explanation` 类：统一存放全局/局部重要性、路径、可视化句柄。

2. **方法集成层**

   * Intrinsic：

     * `SignalPathExplainer`（TSPN/NNSPN）
     * `OperatorAttentionExplainer`（from TII_operator_attention）
     * `MoEExpertExplainer`（from MOE_explainable）
   * Post-hoc：

     * Captum 的 IG/DeepLift/Saliency 封装。

3. **FD 特化可视化与指标**

   * 时域 + 频域 + operator graph 组合图；
   * 解释指标模块：fidelity、稳定性、物理一致性等。

#### 4. 近期里程碑

* [ ] 在 `explainability/` 下完成 `UnifiedExplainer` + `Explanation` 的最小实现。
* [ ] 集成 1 个 intrinsic（signal_path）+ 1 个 Captum（IG）方法，跑通 TSPN 的解释。
* [ ] 写出 1 个 CLI 脚本：`run_explain.py`，实现「一条命令生成解释与图」。
* [ ] 用 THU_018 + CWRU 跑完一组 baseline，对比「黑盒 CNN + SHAP」 vs 「TSPN + Toolkit」。

#### 5. 与其他项目接口

* 向上：

  * 为 `LLM_Explainable_FD_Toolkit` 提供结构化解释数据；
  * 为 `Neuralsymbolic_theory` 提供工程层的统一框架样例。
* 向下：

  * 被 `1D-2D_fusion_explainable`、`MOE_explainable`、`Paper_fuzzy_XFD`、`TII_operator_attention` 调用，用于统一评估与图表生成。

---

## 3. `LLM_Explainable_FD_Toolkit`

### LLM 增强的事后可解释诊断工具集

#### 1. 核心问题

* 如何把工具集输出的「结构化解释」（路径、频带、特征、规则等）变成：

  * **一致的、专业的、可追溯的自然语言解释**；
  * 支持用户追问的对话式诊断助手。

#### 2. 研究内容

* 定义 FD 场景下的解释中间表示（Explanation IR，EIR）。
* 设计多角色的问答模板：面向工程师、管理者、学生等不同用户。
* 探索「解释 + 规则 + 知识图谱 + LLM」的融合机制：

  * 如何确保 LLM 文本不偏离底层 EIR；
  * 如何利用符号规则、模糊规则增强 LLM 的专业性。

#### 3. 技术路线

1. **EIR 设计**

   * JSON schema，包含：
     `fault_label, key_bands, key_times, operator_path, expert_weights, rule_firings, confidence, dataset_meta...`
2. **LLM 解释模块**

   * Prompt 模块化（system + context + examples + constraints）；
   * 增加 consistency check：LLM 生成的数字/事实必须来自 EIR。
3. **对话流程设计**

   * 单轮解释报告生成；
   * 多轮追问（例如「为什么认为是外圈故障？」「如果工况变化会怎样？」）。

#### 4. 近期里程碑

* [ ] 定义并实现 EIR schema（Python dataclass + JSON 序列化）。
* [ ] 选取 3–5 个典型样本，从 Toolkit 导出 EIR，并手写 1 版「人工解释」。
* [ ] 设计第一版 prompt，使 LLM 输出尽量接近人工解释风格。
* [ ] 对比「有 EIR 限制」与「仅原始模型输出」两种 LLM 解释的一致性与专业性。

#### 5. 与其他项目接口

* 强依赖：`Explainable_FD_Toolkit`、`Paper_fuzzy_XFD`、`Neuralsymbolic_theory` 中的规则与知识。
* 输出：为整体工程侧提供「对话式可解释 FD 助手」demo，也为理论篇提供「神经–符号–语言一体化」实例。

---

## 4. `MOE_explainable`

### 基于物理机理约束的 MoE 可解释模型

#### 1. 核心问题

* 如何把**物理算子/特征**封装为「专家」，由一个可解释路由器控制，使得：

  * 每个样本只激活少数几个专家（稀疏、可解释）；
  * 专家激活模式与故障机理高度对齐。
* 如何借鉴「可解释 MoE」最新进展，在 FD 场景构造内在可解释的 MoE 结构。

#### 2. 研究内容

* 提出一个**物理机理约束 MoE**：

  * 专家 = 不同频带滤波器、不同时频算子、不同 1D–2D 模态分支；
  * 路由器 = 带物理先验的 gating 网络或 operator attention。
* 定义 MoE 解释指标：

  * 专家多样性、稀疏度、工况/故障特定专家激活；
  * 专家贡献与物理特征的相关性。
* 与通用 MoE（无物理先验）对比：

  * 性能（准确率、鲁棒性）+ 解释效果。

#### 3. 技术路线

1. **结构设计**

   * 专家库：

     * $$\text{Expert}_k(x) = \mathcal{F}_k(x)$$，其中 $$\mathcal{F}_k$$ 为特定频带/算子组合；
   * 路由器：

     * $$g(x) = \text{softmax}(W \phi(x))$$ 或 operator attention 风格；
     * 稀疏化约束（top-k gating、L1 正则）。

2. **训练与约束**

   * 损失：$$L = L_{\text{task}} + \lambda_1 L_{\text{sparsity}} + \lambda_2 L_{\text{physics}}$$
   * $$L_{\text{physics}}$$：约束专家激活与物理 prior（如特征频率）一致。

3. **解释方法**

   * 专家激活热图 + 样本级专家路径；
   * 与 Toolkit 的 attribution 联合使用，验证 MoE 解释的 faithfulness。

#### 4. 近期里程碑

* [ ] 在现有 TSPN/TFON 上实现一个「简化 MoE」版本（例如两个频带专家 + 一个 broadband 专家）。
* [ ] 在 THU_018 上训练，观察专家激活模式是否与不同故障类型对齐。
* [ ] 接入 Toolkit，输出「专家激活路径图」和解释指标。
* [ ] 与无物理先验 MoE/普通 CNN 做首次对比，完成 4–5 页初稿实验结果草图。

#### 5. 与其他项目接口

* 与 `TII_operator_attention` 共用「算子/专家权重」的视角；
* 为 `Paper_fuzzy_XFD` 提供可离散化为规则的专家激活；
* 为 `Neuralsymbolic_theory` 提供神经–符号中「专家/规则层」的主要案例。

---

## 5. `Paper_fuzzy_XFD`

### 模糊逻辑驱动的可解释故障诊断

#### 1. 核心问题

* 如何在 FD 中引入模糊逻辑，使决策边界具有清晰语义（低/中/高、轻/重故障），并与神经网络特征协同工作，而不是简单的「NN + 黑盒决策」。

#### 2. 研究内容

* 从 TSPN/MoE/Operator Attention/Toolkit 提取可解释特征；
* 为这些特征设计模糊隶属函数和规则库；
* 探索三种组合方式：

  1. NN 特征 → 模糊决策；
  2. 模糊规则 → NN 正则（规则 loss）；
  3. 联合推理（NN + Fuzzy Ensemble）。

#### 3. 技术路线

1. **特征与隶属度设计**

   * 选 5–10 个强可解释特征（如 RMS、峭度、特征频带能量、特定算子输出）；
   * 为每个特征定义 $$\mu_{\text{low}},\mu_{\text{med}},\mu_{\text{high}}$$ 等模糊集。

2. **规则库构建**

   * 结合领域知识与 XAI 结果挖掘规则：

     > IF 高频能量高 AND 峭度高 THEN 可能为外圈故障（置信度 X）。

3. **集成方式**

   * Pipeline A：NN 输出 → 特征 → FIS → 最终决策；
   * Pipeline B：训练 NN 时加入规则一致性损失；
   * Pipeline C：NN 与 FIS 独立预测，使用简单组合（vote/加权）。

#### 4. 近期里程碑

* [ ] 确定 1 个数据集 + 1 套候选特征；
* [ ] 使用 Toolkit 输出解释结果，手动整理出 10–20 条候选规则；
* [ ] 实现一个独立 FIS（模糊推理系统），验证规则在简单场景下的性能；
* [ ] 与基础 NN 对比：性能略降/持平，但解释可读性显著提升（用 case study 展示）。

#### 5. 与其他项目接口

* 输入：来自 Toolkit/MoE/Operator Attention 的解释与特征；
* 输出：结构化的模糊规则，供 LLM 工具集用自然语言方式呈现，也为 `Neuralsymbolic_theory` 提供符号推理素材。

---

## 6. `Neuralsymbolic_theory`

### 神经–符号一体化可解释故障诊断理论（含光滑算子理论）

#### 1. 核心问题

* 如何在 FD 场景中，系统性回答：

  > *「神经网络（算子网络 + MoE）+ 符号规则（模糊/知识图谱）+ 语言解释（LLM）应该如何组合？
  > 这些组合在可解释性、鲁棒性、性能上的本质 trade-off 是什么？」*

#### 2. 研究内容

* 提出一个面向 FD 的神经–符号三层结构：

  * 算子层（光滑算子网络、operator attention、MoE）；
  * 规则层（模糊规则、知识图谱、逻辑约束）；
  * 语言层（LLM 解释和交互）。
* 在算子层引入**光滑算子理论**视角：

  * 分析算子连续性、可微性与解释稳定性的关系；
  * 探索「光滑但稀疏/分段线性」结构如何兼顾解释与表示能力。
* 综合前述论文的实验结果，总结可解释 FD 的设计原则。

#### 3. 技术路线

1. **结构抽象**

   * 使用图/范畴/算子表示，统一表示 TSPN、MoE、1D–2D 融合、模糊 FIS、LLM 工具集；
2. **理论分析**

   * 对比不同结构的表达能力（capacity）、可解释性（interpretability）、稳定性（stability）；
   * 给出简单的定理/命题（哪怕是 toy setting）说明光滑算子和稀疏路由如何影响解释能力。
3. **经验总结**

   * 汇总所有子项目的定量指标（fidelity / 一致性 / 物理对齐度），进行 meta-analysis；
   * 抽取设计原则，如「算子可枚举 + 稀疏注意力 + 规则约束 → 更稳定的解释」。

#### 4. 近期里程碑

* [ ] 搭建一张「神经–符号–语言」总体结构图，标注 7 篇论文各自所在位置；
* [ ] 从现有/计划数据中抽取 2–3 个典型实验，对比不同结构的解释指标；
* [ ] 完成理论部分的提纲（Definition / Proposition / Discussion），至少写出 1–2 个形式化小结果；
* [ ] 写成一篇 6–8 页的「立项/综述+轻量理论」草稿，为后续顶刊投稿打底。

#### 5. 与其他项目接口

* 上：为后续新结构/新工具提供设计原则和理论框架；
* 下：吸收 1D–2D、MoE、模糊、Toolkit、LLM 等工作的实验结果与经验，形成统一抽象。

---

## 7. `TII_operator_attention`

### Operator Attention 机制与透明算子选择（优化中）

#### 1. 核心问题

* 如何给出一个**对振动/算子网络友好**的 operator attention 形式，使得：

  * 注意力权重与具体物理算子/频带一一对应；
  * 可以系统性分析不同工况行为空间中的算子选择模式；
  * 与 MoE gating、标准 attention 做出清晰差异和优势说明。

#### 2. 研究内容

* Operator Attention 的数学形式与复杂度分析；
* 在 TSPN/NNSPN/TFON 等算子网络中的实现与对比实验；
* 与 MoE gating、普通通道 attention 的可解释性比较。

#### 3. 技术路线

1. **形式定义**

   * $$\alpha_k(x) = \frac{\exp(f_k(x))}{\sum_j \exp(f_j(x))}$$
     其中 $$k$$ 对应具体算子（特定频带/滤波器/算子链），并保证 $$\alpha_k$$ 稀疏或有物理 prior；
   * 分析其梯度、复杂度与可解释性特点。
2. **实验设计**

   * 在相同 backbone 下，比较：

     * 无 attention；
     * 通用通道 attention；
     * operator attention；
     * MoE gating；
   * 在解释层面：比较不同机制的频带权重和物理可解释性。
3. **可视化与案例**

   * 对典型设备/工况，画出随时间/负载变化的 operator attention 轨迹；
   * 展示注意力分布如何对应到实际故障机理。

#### 4. 近期里程碑

* [ ] 梳理退稿意见，分类为「理论不足 / 实验不充分 / 差异不够清晰」三类，并逐条对应补强策略；
* [ ] 完善 operator attention 的数学定义与复杂度分析，写成独立小节；
* [ ] 在 1–2 个数据集上，完成与「无 attention / 通用 attention / MoE」的性能 & 解释对比；
* [ ] 用 Toolkit 的解释指标重新评估 operator attention 的解释性收益，形成新一版图表与实验 Section 草稿。

#### 5. 与其他项目接口

* 为 `MOE_explainable` 提供「可解释路由/权重」的基础形式；
* 为 `1D-2D_fusion_explainable` 提供算子/模态级的 attention 机制；
* 为 `Neuralsymbolic_theory` 提供具体的「算子层可解释机制」案例。
