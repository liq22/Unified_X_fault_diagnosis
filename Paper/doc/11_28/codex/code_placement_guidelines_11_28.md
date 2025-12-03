# 代码归属与解耦规范（公共部分 vs 各 Paper 子项目）

> 目的：  
> - 指导后续 Agent 判断「一段代码应该放在根项目公共部分，还是放在某个 Paper 子目录下」。  
> - 降低重复实现和耦合，让 7 篇 Paper 能共享基础设施，又保持各自独立创新。  
>
> 范围：适用于当前仓库中所有新建/调整的 Python 代码、配置、脚本。

---

## 一、总体原则：按“复用层级”划分

从上到下划分三层：

1. **公共基础设施层（根项目）**  
   放置位置：  
   - `model/`, `model_collection/`, `trainer/`, `data/`, `configs/`, `explainability/`, `scripts/`（根目录下）  
   满足任一条件的代码应放这里：  
   - 被 **≥ 2 个 Paper 子项目** 使用，或已经进入统一 baseline 对比矩阵；  
   - 可以通过 `main.py` / `main_com.py` + `configs/` 直接调用；  
   - 不依赖某一篇论文的特殊假设、专属图表或特定实验场景；  
   - 希望长期作为“统一平台能力”维护，比如通用模型、数据接口、评估指标。  

2. **方法 / 论文实验层（各 Paper 子目录）**  
   放置位置：  
   - `Paper/<Project>/code`, `Paper/<Project>/scripts`, `Paper/<Project>/experiments`, `Paper/<Project>/tools` 等  
   满足任一条件的代码应放在对应 Paper 目录：  
   - 仅为**某一篇 Paper 的实验/图表/消融**服务，其他方向不依赖；  
   - 属于某个想法的原型、ablation 版本，还没定型为“官方 API”；  
   - 强依赖该论文的特定设定（特殊损失、特定数据切分、专用可视化等）；  
   - 输出直接写入 `Paper/<Project>/results` 或 `Paper/<Project>/figures`，而不是公共 `results/`。  

3. **Glue / Adapter 层（尽量靠近基础设施）**  
   用途：将公共模型/数据/解释接口适配到具体 Paper 的跑法上。  
   - 若适配逻辑具备通用性（例如 VBench Dataset 或统一日志格式），优先放到根目录 `data/`、`explainability/` 或 `scripts/`；  
   - 若是某篇 Paper 专用的复杂流程（特殊 ablation、图编号、命名规则），放在该 Paper 的 `scripts/` 或 `experiments/` 下。  

---

## 二、结合当前项目的具体归属建议

### 2.1 数据与训练框架（必须放公共部分）

- `data/vbench_dataset.py`, `data/vbench_utils.py`  
  - 已经为多个模型和多个 Paper 提供统一数据接口；  
  - 属于“平台级能力”，应长期维护，**固定放在 `data/`**。  

- 训练与配置  
  - `trainer/`、`main.py`、`main_com.py`、`configs/unified_baseline/`：  
    - 为所有模型和 Paper 提供统一训练循环与 baseline 配置；  
    - 不允许拷贝到各 Paper 目录中改名使用，只能引用。  

### 2.2 模型代码

1. **主仓库模型（公共）**  
   - `model/TSPN.py`, `model/NNSPN.py`, `model/TFON.py` 等：  
     - 作为统一 baseline 的核心模型，必须放在 `model/`，所有 Paper 从这里 import。  

2. **Fusion1D2D 相关**  
   - 公共部分：  
     - `model/Fusion1D2D.py`, `model/Fusion1D2D_simple.py`：  
       - 作为统一 baseline 的 1D–2D 融合模型与简化版本，应固定放在 `model/`；  
       - 通过 `main.py` 的 `MODEL_DICT['Fusion1D2D']` 调用。  
   - Paper 专用部分：  
     - `Paper/1D-2D_fusion_explainable/code/**`：  
       - 专门为 1D-2D 论文设计的变体（特殊对齐损失、实验型注意力结构等）放这里；  
       - 只有当某个变体被 ≥ 2 篇 Paper 复用，才考虑上升为公共模块。  

3. **MoE 相关**  
   - 公共部分：  
     - `model/MoE_simple.py`：  
       - 已作为统一基线中 “MoE” 的实现挂到 `main.py`；  
       - 作为“平台级 MoE 简化版”，固定放在 `model/`。  
   - Paper 专用部分：  
     - `Paper/MOE_explainable/code/experts/`, `Paper/MOE_explainable/code/router/`：  
       - 针对 MoE 论文的物理专家设计、复杂路由策略，仅在该 Paper 使用，应放在 Paper 下；  
       - 若未来 Fuzzy/1D-2D 等项目也复用同一套专家或路由，再将共用部分抽取到 `model/`。  

4. **FuzzyLogic 相关**
   - 公共部分：  
     - `model/FuzzyLogic_simple.py`：  
       - 作为统一 baseline 的简单模糊模型实现，可放在 `model/` 供多篇 Paper 引用。  
   - Paper 专用部分：  
     - `Paper/Paper_fuzzy_XFD/code/fuzzy_system/**`：  
       - 包含特定工业规则、复杂模糊推理逻辑，仅用于 Fuzzy-XFD 论文；  
       - 不上升为公共部分，除非被其他项目认可并复用。  

5. **Operator Attention 相关**
   - 公共部分：  
     - `model/operator_attention.py`, `model/TSPN_OperatorAttention.py`：  
       - 提供算子注意力模块与与 TSPN 的集成版本，是可复用的“方法级能力”；  
       - 保持在 `model/`，用 `main.py` 中封装的 `OperatorAttentionNetwork` 调用。  
   - Paper 专用部分：  
     - `Paper/TII_operator_attention/doc/**`, `Paper/TII_operator_attention/scripts/**`：  
       - 理论推导、特定 demo、ablation 实现属于 TII 论文范畴，保持在 Paper 下。  

6. **Explainable Toolkit / LLM Toolkit**

- Explainable_FD_Toolkit  
  - `Paper/Explainable_FD_Toolkit/toolkit_integration/**`：  
    - 虽然逻辑上类似“跨 Paper 共享工具”，但该工具包本身就是一个 Paper 项目；  
    - 建议保持在 Paper 下，通过绝对导入或接口调用，不再在根目录复制一份。  

- LLM_Explainable_FD_Toolkit  
  - `Paper/LLM_Explainable_FD_Toolkit/code/llm_explainable_toolkit/**`：  
    - LLM 相关依赖较重，且属于应用层；  
    - 不应强行上移到根目录，以避免根项目对 LLM 环境产生硬依赖。  

---

## 三、后续 Agent 的“放置决策树”

后续 Claude Code / 子 Agent 在创建或移动代码时，应按以下决策顺序判断放置位置：

1. **问题 1：这段代码会被多个 Paper 复用吗？**  
   - 是：→ 倾向放在公共部分（`model/`, `data/`, `trainer/`, `explainability/` 等）；  
   - 否：→ 继续问题 2。  

2. **问题 2：这段代码是否依赖某个 Paper 的特定设定？**  
   （例如：专属损失函数、特定 ablation、只为某一张图/一个表服务）  
   - 是：→ 放在对应 `Paper/<Project>/code` 或 `Paper/<Project>/scripts` 下；  
   - 否：→ 继续问题 3。  

3. **问题 3：这段代码是否已经稳定到可以作为“官方 API”使用？**  
   - 是：→ 放公共部分，同时在 README / configs 中记录正式用法；  
   - 否（原型、实验版本）：→ 放在对应 Paper 目录下，并在文件名或路径中标注 `_experimental` / `ablation` 等。  

> 简化记忆：  
> - “能通用 + 已定型” → 公共部分；  
> - “只为一篇文 + 处于探索期” → Paper 子目录；  
> - “适配/Glue” → 看是跨项目通用（放公共）还是特定论文专用（放 Paper）。  

---

## 四、对现有代码的落地建议（摘要）

1. **保持现状的公共模块**  
   - `data/vbench_dataset.py`, `data/vbench_utils.py`  
   - `model/TSPN.py`, `model/NNSPN.py`, `model/TFON.py`  
   - `model/Fusion1D2D.py`, `model/Fusion1D2D_simple.py`  
   - `model/MoE_simple.py`, `model/FuzzyLogic_simple.py`  
   - `model/operator_attention.py`, `model/TSPN_OperatorAttention.py`  

2. **明确只放 Paper 的模块**  
   - `Paper/1D-2D_fusion_explainable/code/**`（特定对齐/融合变体）  
   - `Paper/MOE_explainable/code/**`（物理专家库、复杂路由）  
   - `Paper/Paper_fuzzy_XFD/code/fuzzy_system/**`（工业规则与模糊系统实现）  
   - `Paper/TII_operator_attention/**` 下的理论与实验脚本  
   - `Paper/Explainable_FD_Toolkit/**` 与 `Paper/LLM_Explainable_FD_Toolkit/**`（作为独立工具包项目）  

3. **如需调整的地方**  
   - 若未来发现某个 Paper 目录下的模块被多个方向重复引用，优先：  
     1）抽出“最小可复用子集”移入 `model/` 或 `explainability/`；  
     2）在原 Paper 目录中只保留特定论文的封装/拓展层。  

---

本规范主要供后续 Claude Code / 子 Agent 在创建、移动、重构代码时参考。  
