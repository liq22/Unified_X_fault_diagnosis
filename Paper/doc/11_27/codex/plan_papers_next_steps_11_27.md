# 7 个 Paper 子项目阶段 3 下一步计划（2025-11-27）

> 依据：  
> - `docs/11_26/codex/plan_stage3_integration_11_26.md`（阶段 3 总体集成计划）  
> - `Paper/doc/11_27/codex/plan_baseline_and_integration_11_27.md`（统一基线调优计划）  
> - `docs/11_27/glm/实验执行清单.md` & `统一基线实验完成计划.md`（GLM 视角执行清单）  
>
> 目标：在当前 TSPN 基线「性能极佳但稳定性不足」的前提下，为 7 个 Paper 子项目给出阶段 3 的可执行下一步计划，便于后续智能体按子项目维度推进。

---

## 总体约束与优先级回顾（仓库级）

- **统一优先级（短期 3–5 天）**：  
  1. **先稳定 TSPN**（已降低 `l1_norm`，正在验证梯度裁剪效果）；  
  2. **补齐经典基线**：ResNet / SincNet / WKN / MCN / TFN（统一 VBench + trainer）；  
  3. **跑完 7 个新方法的第一轮主实验**：Fusion1D-2D / MoE / OperatorAttention / Fuzzy 等。  
- 7 个 Paper 的具体计划应在不违背上述优先级的前提下展开：  
  - 不依赖数值对比的工作（文档、接口、理论）可以并行推进；  
  - 依赖统一基线/指标的实验，应在基线逐步稳定时按优先级接入。

下面按子项目逐个给出阶段 3 的「下一步建议任务」，每项都参考了其各自的 `plan/11_26/codex` 与 research proposal。

---

## 1. 📘 1D-2D_fusion_explainable：融合主实验接入统一基线

**短期（基线补齐期间可做）**

- [ ] 检查并对齐 `configs/a_018_THU/config_Fusion1D2D.yaml` 与 `configs/unified_baseline/config_Fusion1D2D.yaml`：  
  - 确保使用 `data/vbench_dataset.py` + 统一 trainer；  
  - 确保训练/评估指标与 TSPN/ResNet 基线设置一致。  
- [ ] 在 `Paper/1D-2D_fusion_explainable/scripts/` 中新增或完善：  
  - `run_unified_fusion_baseline.py`：包装统一基线配置，便于从子项目视角复现实验。

**中期（在 TSPN + ResNet 基线稳定后）**

- [ ] 在 THU_018_basic 上运行 Fusion1D2D 的主实验（至少 3 次），填入 `Paper/doc/11_26/results/unified_baseline_comparison.md`：  
  - 与 TSPN / ResNet / SincNet / WKN/MCN/TFN 在同一表内对比性能与参数量。  
- [ ] 启动「三层对齐 + 可解释性」的最小版实验：  
  - 先选 1–2 个配置（如 early fusion + 简化对齐损失），生成对齐指标与可视化图，作为后续大规模实验的“样板”。  

---

## 2. 🟢 Explainable_FD_Toolkit：统一解释性评估的落地与批量分析

**短期**

- [ ] 基于已创建的 `SignalData/ExplainabilityMethod/ModelPlugin` 协议：  
  - 为 TSPN 与 ResNet 实现最小可用的 `ModelPlugin` 封装（可放在 Toolkit 内部或 `explainability/` 层）。  
- [ ] 新增一个 `scripts/run_unified_explain_eval.py`：  
  - 输入：模型列表（TSPN, ResNet），解释方法列表（至少 1 个本征 + 1 个事后）；  
  - 输出：解释覆盖度、稳定性、一致性等基础指标 CSV，用于更新统一对比文档。

**中期**

- [ ] 等 MoE / Fusion1D2D / OperatorAttention / Fuzzy 的统一基线跑完后：  
  - 将这些模型也接入 Toolkit 的批量解释接口；  
  - 生成第一版“多模型多方法”解释性主表，作为 Toolkit 论文核心结果之一。  

---

## 3. 🟣 LLM_Explainable_FD_Toolkit：用统一基线解释结果驱动 LLM 实验

**短期**

- [ ] 在暂不依赖外部 LLM 的前提下：  
  - 使用 Explainable_FD_Toolkit 生成 TSPN 与 ResNet 的解释结果（小样本）；  
  - 在 LLM 工具包中通过模板/stub 生成自然语言解释，验证中间表示设计是否合理。

**中期**

- [ ] 在 ResNet + TSPN + 至少一个新方法（如 Fusion1D2D 或 MoE）基线稳定后：  
  - 设计第一版 LLM vs 传统解释的对比实验：  
    - 同一组样本，生成传统图/文字解释 + LLM 文本解释；  
    - 通过定性/定量方式评估可理解性与效率。  
  - 输出一个小型对比表，先以 THU_018_basic 为例，再扩展。

---

## 4. 🟠 MOE_explainable：统一基线下的 MoE 性能与可解释性实验

**短期**

- [ ] 确认 `configs/unified_baseline/config_MoE.yaml` 与 `Paper/MOE_explainable` 内部实验配置一致：  
  - 使用相同的数据接口与 trainer；  
  - 确保模型名 `MoE` 已正确注册到 `main.py` 的 `MODEL_DICT`。  
- [ ] 在 `Paper/MOE_explainable/scripts/` 中添加或更新：  
  - `run_moe_unified_baseline.py`：使用统一基线配置，跑 3 次主实验。

**中期**

- [ ] 在 THU_018_basic 上完成 MoE 与 TSPN/ResNet 的统一对比（至少 3 次）：  
  - 填入 unified baseline 表（性能/稳定性/参数量）。  
- [ ] 利用 Explainable_FD_Toolkit 或本地脚本生成：  
  - 路径签名热力图、专家激活分布等关键图，作为 MoE 论文的阶段性图形成果。  

---

## 5. 🩷 Paper_fuzzy_XFD：规则-神经混合在统一基线数据上的首批实验

**短期**

- [ ] 基于 THU_018_basic 的 TSPN/ResNet 已有特征或数据：  
  - 先实现/验证纯模糊诊断 baseline（读取特征文件或直接从 Dataset 中提取统计特征）。  
  - 在 `Paper/Paper_fuzzy_XFD/scripts/` 中添加 `run_fuzzy_baseline_unified.py`。

**中期**

- [ ] 在同一数据集和任务下，比较：  
  - 纯模糊系统 vs 纯深度模型（TSPN 或 ResNet）vs Fuzzy-XFD 融合方案 A（先模型后模糊）。  
  - 将结果填入 Fuzzy-XFD 的性能表模板中，并同步更新 unified baseline 文档中的“可解释性能力”描述。  
- [ ] 选取 1–2 个规则 + 样本案例，生成图示，为 Fuzzy-XFD 论文的图 1/2 做准备。  

---

## 6. 🟦 Neuralsymbolic_theory：用统一基线结果丰富 NeSy 框架案例

**短期**

- [ ] 从 `docs/11_26/results/unified_baseline_comparison.md` 中，提取当前已有的：  
  - 性能指标（TSPN + baseline 未完成部分可留空标记）；  
  - 可解释性能力排序草案。  
- [ ] 在 NeSy draft 中增加一小节“初步实证观察”：  
  - 说明哪些类型的结构（透明结构 / MoE / Fuzzy / OpAtt 等）在统一设置下表现出哪些性质。

**中期**

- [ ] 当 MoE / Fuzzy / OpAtt / Fusion1D2D 的主结果逐步补齐后：  
  - 选取 2–3 个对比最明显的实验（例如 TSPN vs MoE vs Fuzzy），用 NeSy 语言进行“理论视角下重解读”；  
  - 将这些案例嵌入 NeSy 论文的实验或案例分析部分，为形式定义提供直观支撑。  

---

## 7. 🔴 TII_operator_attention：统一基线下的算子注意力性能与可解释性

**短期**

- [ ] 基于 `configs/a_018_THU/config_TSPN_opatt.yaml` 与 `configs/unified_baseline/config_OperatorAttention.yaml`：  
  - 确认 Operator Attention 版本的 TSPN 已注册并可在统一 trainer 下运行；  
  - 在 `scripts/example_usage_operator_attention.py` 或新的脚本中，跑通最小实验。  

**中期**

- [ ] 在 THU_018_basic 上完成三组对比：  
  1. baseline TSPN（无 Attention，使用稳定配置）；  
  2. 带 Self-Attention 的版本（如有实现，可作为中间对照）；  
  3. 带 Operator Attention 的版本。  
- [ ] 将性能/复杂度/可解释性指标填入 unified baseline 表，并在 `Paper/TII_operator_attention` 目录下补充：  
  - 算子权重热力图；  
  - 算子权重随工况/SNR 变化的曲线；  
  - 与 Self-Attention 的可视化对比示意图。  

---

## 8. 建议执行顺序（按优先级组合视角）

结合全局与子项目计划，建议智能体按以下顺序推进：

1. **先补齐经典基线 + TSPN 稳定性验证**（仓库级工作，已在 codex 顶层计划中定义）；  
2. 并行推进：  
   - 🟢 Explainable Toolkit 的多模型解释接口；  
   - 🔴 Operator Attention 的统一基线对比；  
   - 🟠 MoE 与 📘 Fusion1D2D 在 THU_018_basic 上的首轮统一 baseline；  
3. 在性能/稳定性初步清晰后，逐渐接入：  
   - 🩷 Fuzzy-XFD 的规则/混合实验；  
   - 🟣 LLM 工具包的解释质量对比实验；  
   - 🟦 NeSy 的框架化总结与案例整合。

每次完成一个子项目的阶段任务，建议在该项目 `plan/11_26/codex` 计划文件中打勾，并在 `Paper/doc/11_26/results` 或 `docs/11_26/results` 目录中追加结果汇总，保持「计划—执行—结果」闭环。  

