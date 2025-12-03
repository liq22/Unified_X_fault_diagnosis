# 统一故障诊断项目 TODO 梳理与 12‑01 计划（Codex）

> 来源整理：  
> - `Paper/doc/11_29/codex/unfinished_tasks_and_plan_11_29_codex.md`  
> - `Paper/doc/11_29/codex/mvp_unified_baseline_11_29_codex.md`  
> - `Paper/doc/11_29/codex/high_level_plan_after_mvp_11_29_codex.md`  
> - `Paper/doc/11_29/glm/notes_unfinished_and_plan_11_29_glm.md`  
>
> 目标：将 11‑29 时点尚未完成的事项收敛为 12‑01 之后可以直接执行的 TODO 与短期计划。

---

## 一、统一基线相关 TODO（技术层）

### 1. Fusion1D2D 验证阶段 shape 问题闭环

- 现状：  
  - Simple 版 Fusion1D2D 已能初始化和前向，但在完整训练/验证流程中曾出现 `validation_step` 的 `shape '[64, 3, -1]'` 错误；  
  - 文档中已有定位与大致分析，但尚未用**最小脚本**做闭环。  

- 12‑01 之后的具体 TODO：  
  1. 在 `scripts/` 下实现 `debug_fusion1d2d_shape.py`：  
     - 使用 `data/VbenchDataset` 加载 THU_018_basic 的一个小 batch；  
     - 构造 `Fusion1D2D` 模型（统一基线配置）；  
     - 手动调用与 `validation_step` 等价的 forward 逻辑，重现并修复 shape 问题。  
  2. 在新的状态报告（12‑01 文档）中记录：问题位置、修复方法和验证结果。  

### 2. OperatorAttention L1 正则化调参与效果记录

- 现状：  
  - 配置 `configs/unified_baseline/config_OperatorAttention.yaml` 中 `args.l1_norm` 仍为 `0.0001`；  
  - 文档中规划了“降低一个数量级”的策略，但尚未有实验对比结果。  

- 12‑01 之后的具体 TODO：  
  1. 将 `l1_norm` 调整为 `1e-4` 或 `1e-5`，在 YAML 中注明调整日期与原因；  
  2. 运行一轮短实验（不必跑满所有 epoch），记录：  
     - L1 值的典型范围（调整前 vs 调整后）；  
     - 验证损失与验证精度的变化趋势；  
  3. 在 12‑01 状态文档中简要总结调参结论（有效/无显著变化/需进一步试验）。  

### 3. FuzzyLogic_simple 统一 baseline 首条结果

- 现状：  
  - FuzzyLogic_simple 的统一 baseline 条目已在 `unified_baseline_results_codex_11_29.md` 中预留；  
  - 但尚未完成一次完整 run 并记录结果。  

- 12‑01 之后的具体 TODO：  
  1. 在 `configs/unified_baseline/` 下新增或确认 FuzzyLogic 对应配置（例如 `config_FuzzyLogic.yaml` 或类似命名）；  
  2. 运行一条完整基线实验（THU_018_basic），得到首个 test accuracy；  
  3. 将结果填入统一基线表，并在备注中标为“快照：首个可运行 baseline”。  

### 4. 统一基线结果表 v1 的更新与标注

- 现状：  
  - `unified_baseline_results_codex_11_29.md` 已存在表头与若干快照行；  
  - 部分行（OperatorAttention、FuzzyLogic_simple）仍为占位或粗略描述。  

- 12‑01 之后的具体 TODO：  
  1. 用最新实验结果更新 OperatorAttention 行的准确率与备注；  
  2. 在 FuzzyLogic_simple 跑通后补充其行；  
  3. 确保每一行都有明确备注：`快照 / 待复现 / 已验证`。  

---

## 二、论文与文档相关 TODO（Paper 层）

### 1. 3 篇实验型 Paper 的结果与基线引用完善

目标 Paper：
- `Paper/1D-2D_fusion_explainable`  
- `Paper/MOE_explainable`  
- `Paper/TII_operator_attention`  

未完成事项：
- README / research proposal 中已经添加统一基线引用语句，但尚未：  
  - 明确列出当前快照结果（例如 Fusion1D2D 的一次 best run、MoE 的 63%、OperatorAttention 当前水平）；  
  - 指出需要补齐的实验（更多种子、更多数据集）对应哪些图/表。  

12‑01 之后的具体 TODO：  
1. 在各自的 `doc/research_proposal_*.md` 中：  
   - 为“结果与讨论”小节添加：  
     - 当前统一基线引用（统一表路径）；  
     - 已有快照结果（用文字描述，不重复表格数值）；  
     - 计划中的扩展实验（简要列出，如 “更多 seed / CWRU 数据集 / 消融实验”）。  
2. 确定每篇 Paper 的“主图”和“主表”：  
   - 例如：1D‑2D 的性能曲线 + 模态贡献图；MoE 的专家热力图；OperatorAttention 的算子权重热图。  

### 2. 4 篇工具 / 理论型 Paper 的统一基线引用模板

目标 Paper：
- Explainable_FD_Toolkit  
- LLM_Explainable_FD_Toolkit  
- Paper_fuzzy_XFD  
- Neuralsymbolic_theory  

未完成事项：
- README 中虽然有项目定位，但尚未统一使用“统一 baseline”作为性能对比的外部参照；  
- 一些文档仍各自罗列结果，而没有指向统一表。  

12‑01 之后的具体 TODO：  
1. 在每个 README 的“实验/应用/评估”相关小节中加入统一引用句：  
   ```markdown
   本文中涉及的模型性能对比统一参考 `Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`，
   本节重点关注本工具/理论在解释性或结构上的贡献。
   ```  
2. 将后续出现的任何新对比数值，都优先整理到统一表中，再在各 Paper 文档用文字解释差异。  

---

## 三、GLM 计划与 Codex 计划的协调要点

结合 `notes_unfinished_and_plan_11_29_glm.md`，对 12‑01 之后的协作提出几点约束：

1. **结果表以 Codex 版本为准**  
   - GLM 文档中若需展示统一 baseline 数字，应引用 Codex 的统一结果表，而不是再维护一套独立表格。  

2. **区分“快照”与“最终 baseline”**  
   - 在任何 GLM 生成的报告中：  
     - 对当前少数 run 得到的结果，使用“快照 / snapshot”等字样；  
     - 当 Codex 明确标记某行结果为“已验证”后，再在 GLM 总结中提升其地位。  

3. **避免复活已被修正的旧建议**  
   - 不再建议修改 `OperatorAttention_simple.py` 的 `target_channels`；  
   - 不再使用 `config_Fusion1D2D_simple.yaml` 或 `--config_file` 之类的旧参数；  
   - 新计划应以最新的 Codex 版本代码结构与接口说明为基准。  

---

## 四、12‑01 的优先执行顺序建议

1. **优先完成 Fusion1D2D shape 问题闭环（任务 1）**  
   - 这是统一基线中最突出的技术风险，尽早用最小脚本确认并修复。  

2. **其次推进 OperatorAttention L1 调参与 FuzzyLogic_simple baseline（任务 2 & 3）**  
   - 完成后，统一基线表的核心模型行就比较完整，可支撑大部分论文的对比。  

3. **最后补文档与 Paper 级引用（任务 4 + 第二部分）**  
   - 确保所有 Paper 读者知道性能数字来自哪张统一表，各自文档只讨论“相对差异”和方法本身的贡献。  

完成上述 TODO，即可认为 11‑29 遗留任务已转化为 12‑01 起可执行的短期计划，后续再按照 `high_level_plan_after_mvp_11_29_codex.md` 分阶段推进中期与长期目标。  

