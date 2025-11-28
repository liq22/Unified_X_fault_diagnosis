# 7 个 Paper 关键测试脚本与基线收尾计划（2025-11-27）

> 背景：  
> - Fusion1D2D 与 FuzzyLogic 的 `Identity` / `SignalProcessingModuleDict` 初始化问题已修复，  
>   Fusion1D2D 在统一框架下已通过最小前向测试。  
> - MoE_simple 已有一组实验成功完成（测试准确率约 74.3%），证明 MoE 路径可用。  
> - 仍存在的问题：部分实验仍引用旧版模型或参数不匹配（MoE vs MoE_simple、OperatorAttention 简化版等）。  
> 目标：为 7 个 Paper 子项目补齐「最小可运行测试脚本」与「统一基线收尾任务」，让后续智能体一看就能执行。

---

## 一、总体目标与原则

- 为每个关键 Paper 子项目至少提供一个 **最小可运行测试脚本**，用于：
  - 验证与主仓库模型/数据接口的兼容性（导入、初始化、前向传播）。  
  - 快速诊断配置/初始化错误，而不必直接跑完整训练。  
- 对已进入统一基线矩阵的模型（如 MoE、Fusion1D2D），补齐：
  - 「使用哪个版本（simple vs full）」的明确说明；  
  - 统一基线实验的**收尾执行计划**（剩余模型：MCN、TFN 等）。  
- 所有新脚本与说明文件优先放在对应 Paper 子目录下，避免污染父项目。

---

## 二、Fusion1D2D（📘 1D-2D_fusion_explainable）

**当前状态**
- 主仓库 `model/Fusion1D2D.py` 已修复 `Identity(args)` 等初始化问题。  
- 已在 `Paper/1D-2D_fusion_explainable/scripts/` 下新增  
  `test_unified_fusion1d2d_identity_fix.py`，可完成最小前向检查。

**下一步计划**
- [ ] 在 `Paper/1D-2D_fusion_explainable/scripts/` 中新增（或补充说明）：  
  - `README_scripts.md`：说明各脚本用途，特别是区分：
    - `run_minimal_demo.py`：子项目内部的 1D-2D demo 训练；  
    - `test_unified_fusion1d2d_identity_fix.py`：主仓库统一模型接口的快速检查脚本。  
- [ ] 在统一基线计划中标记：  
  - Fusion1D2D 的主实验使用 `model/Fusion1D2D.Fusion1D2D`，  
    配置文件来自 `configs/unified_baseline/config_Fusion1D2D.yaml`。

---

## 三、MoE（🟠 MOE_explainable）

**当前状态**
- 存在两个路径：`model/MoE.py` 与 `model/MoE_simple.py`。  
- 已有一次使用 `MoE_simple` 的实验成功完成（约 74.3% 准确率）。  
- 部分实验仍错误引用 `MoE.py`，导致初始化或接口不匹配。

**测试脚本与统一策略**
- [ ] 在主仓库 `main.py` 侧明确：  
  - 统一基线默认使用 `MoE_simple` 版本（可在文档中注明）。  
- [ ] 在 `Paper/MOE_explainable/scripts/` 中新增：  
  - `test_unified_moe_simple_init.py`：  
    - 只做以下动作：导入主仓库 `model.MoE_simple.MoE`（或相应类），构造最小 `args`，  
      用随机 tensor 做一次前向传播；  
    - 打印输出形状和简单统计，用于智能体快速确认 MoE 能在当前配置下工作。  
  - `README_scripts.md` 中说明：  
    - 统一基线实验一律使用 `MoE_simple`；  
    - 若未来切换回完整版 MoE，应同步修改统一基线配置与测试脚本。

**基线收尾**
- [ ] 按 `Paper/doc/11_27/plan_baseline_and_integration_11_27.md`：  
  - 至少再完成 2 次 MoE_simple 统一基线实验（总计 ≥3 次），  
  - 将平均性能与稳定性指标填入统一基线对比表。

---

## 四、OperatorAttention（🔴 TII_operator_attention）

**当前状态**
- 简化版 OperatorAttention（`SimpleOperatorAttention`）在某些实验中出现**初始化参数不匹配**错误。  
- 主仓库已有 `model/operator_attention.py` 与 `TSPN_OperatorAttention` 等实现。

**测试脚本与修复思路**
- [ ] 在 `Paper/TII_operator_attention/plan/11_26/` 中补充说明：  
  - 明确统一基线使用的 OperatorAttention 具体类与入口（例如：  
    通过 `main.py` 中 `MODEL_DICT['TSPN_OpAtt']`）。  
- [ ] 在 `Paper/TII_operator_attention/scripts/` 中新增：  
  - `test_unified_operator_attention_init.py`：  
    - 导入统一使用的 OperatorAttention 模型（而不是孤立的 Simple 版本）；  
    - 构造最小 `args`，用随机 tensor 做一次前向传播；  
    - 若 `SimpleOperatorAttention` 仍需保留，则在脚本内显式演示正确的初始化参数顺序与示例。

**后续基线实验**
- [ ] 在统一基线矩阵中补齐三条对比线：  
  1. baseline TSPN（无 Attention）；  
  2. 带 Self-Attention（若实现存在）；  
  3. 带 OperatorAttention 的 TSPN。  
- [ ] 在 `Paper/TII_operator_attention/results/` 下新增  
  `unified_baseline_operator_attention_summary.md`，总结三者在性能与复杂度上的差异。

---

## 五、Fuzzy-XFD（🩷 Paper_fuzzy_XFD）

**当前状态**
- 主仓库 FuzzyLogic 模型已修复 Identity/ModuleDict 问题，但当前主流程主要使用 `FuzzyLogic_simple`。  
- Fuzzy-XFD 子项目侧已有独立的规则/模糊代码与计划文档。

**测试脚本**
- [ ] 在 `Paper/Paper_fuzzy_XFD/scripts/` 新增：  
  - `test_unified_fuzzylogic_simple_init.py`：  
    - 测试主仓库 `FuzzyLogic_simple` 在统一数据接口下的前向传播；  
  - 若计划使用主仓库全版本 FuzzyLogic，则再加一个  
    `test_unified_fuzzylogic_full_init.py`，明确区分两者用途。

**与统一基线的衔接**
- [ ] 在 Fuzzy-XFD 的 README / research proposal 中：  
  - 增加一小节说明：统一基线场景下，  
    Fuzzy-XFD 对接的是哪一个 Fuzzy 模型版本（simple/full），  
    以及对比对象（TSPN/ResNet）的配置。

---

## 六、Explainable Toolkit & LLM Toolkit（🟢 / 🟣）

**Explainable_FD_Toolkit**
- [ ] 在 `Paper/Explainable_FD_Toolkit/examples/` 或 `scripts/` 下新增：  
  - `test_unified_modelplugin_tspn_resnet.py`：  
    - 使用统一基线的 TSPN / ResNet 作为 `ModelPlugin` 实现；  
    - 对 1–2 个 batch 生成解释结果，验证接口兼容性。

**LLM_Explainable_FD_Toolkit**
- [ ] 在 `Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/` 中新增：  
  - `test_unified_llm_pipeline_stub.py`：  
    - 不调用真实 LLM，仅用假数据/模板解释结果；  
    - 验证数据流：SignalData → ExplainabilityMethod 输出 → LLM 接口 stub → 文本解释。

---

## 七、Neural-Symbolic Theory（🟦 Neuralsymbolic_theory）

**定位**
- NeSy 项目更多是理论与架构层，不依赖具体模型能否训练完毕，但需要一两个**可运行的示例**支撑理论。

**测试/示例脚本**
- [ ] 在 `Paper/Neuralsymbolic_theory/scripts/` 中新增：  
  - `demo_nesy_mapping_tspn_moe_fuzzy.py`：  
    - 不必完整训练模型，只需：  
      1. 构造三个模型的简化版本（或从主仓库导入已训练权重）；  
      2. 对同一批数据跑前向，收集中间表示；  
      3. 将这些表示映射到 NeSy 框架中定义的结构（例如：规则节点、专家节点等）。  
    - 作为 NeSy 论文中的“统一实验接口”示例。

---

## 八、执行顺序建议（与当前 Todos 对齐）

结合你当前的 Todos：

1. **优先完成模型初始化问题修复（短期 0.5–1 天）**
   - Fusion1D2D：已验证通过，仅需在 1D-2D 子项目 README 中补充说明；  
   - MoE：通过测试脚本 + 文档，统一使用 `MoE_simple`；  
   - OperatorAttention：通过测试脚本精确定义使用的类与参数；  
   - FuzzyLogic：为 simple/full 两种版本各加一个最小测试脚本。

2. **随后为 7 个 Paper 补齐测试脚本骨架（1–2 天）**
   - 按上文每个 Paper 的“测试脚本”条目，新建空壳或最小可跑脚本；  
   - 在各自 `README_scripts.md` 中写清用途与调用方式。

3. **最后收尾统一基线实验（Phase 3–4）**
   - 补齐 MCN、TFN 等经典基线；  
   - 对 MoE / Fusion1D2D / OperatorAttention / Fuzzy 进行至少 3 次主实验；  
   - 将结果集中填入统一基线对比表，并在各 Paper 的 doc 中增加引用。

这样，后续任意一个智能体只要：
- 先在对应 Paper 目录运行 `test_*_init.py` 验证接口；  
- 再根据统一基线计划运行主实验脚本；  
就能在较小心智负担下，安全推进 7 个论文方向的阶段 3/4 工作。  

