# 12‑01 GLM 待办事项与注意点整理

> 本文件是基于 `Paper/doc/11_29/glm/notes_unfinished_and_plan_11_29_glm.md` 的延续，  
> 旨在帮助后续 GLM 生成的计划/报告与 Codex 的统一基线与代码结构保持一致。

---

## 一、GLM 在 12‑01 之后应优先关注的技术 TODO

1. **确保状态报告中的结果引用统一表**  
   - 当描述 THU_018_basic 上 TSPN / Fusion1D2D / MoE / OperatorAttention / Fuzzy 的表现时：  
     - 仅在文字中使用“大致范围”和趋势描述；  
     - 如需列出具体数值，应引用 `Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`。  

2. **在状态更新中标明“快照”身份**  
   - 对尚未复现或跨数据集验证的结果，应明确标为“快照 / snapshot”；  
   - 对未来 Codex 标记为“已验证 baseline”的结果，才可以在 GLM 报告中当作“基准”使用。  

3. **配合 Codex 完成关键技术点闭环**  
   - 如果 Codex 在 12‑01 期间完成了：  
     - Fusion1D2D 的 shape 问题修复；  
     - OperatorAttention L1 调参实验；  
     - FuzzyLogic_simple baseline；  
   - 建议 GLM 后续状态报告中：  
     - 用“问题 → 修复 → 结果”的结构，简明复述这些结论；  
     - 避免再提出与这些修复相冲突的旧建议。  

---

## 二、GLM 撰写新计划/总结时的约束提醒

1. **不要再使用过时的配置名或参数**  
   - 不要再提及 `config_Fusion1D2D_simple.yaml` 或 `--config_file`；  
   - 持续使用 Codex 确认过的路径与接口，例如：  
     - `configs/unified_baseline/config_Fusion1D2D.yaml`  
     - `--config_dir` 作为命令行参数。  

2. **不再建议直接修改简单模型的内部实现**  
   - 如 `OperatorAttention_simple` 的 `target_channels` 之类的旧修补方案，已经被替换为更稳健的逻辑；  
   - 若发现新问题，应先描述现象和日志，再由 Codex 评估是否需要结构性修改。  

3. **计划粒度控制**  
   - 避免在单份计划中同时提出“200 epoch 长跑 + 全面重构 signal_processing/专家模块”的组合；  
   - 改为：  
     - 清楚分出短期可执行任务（几天内完成）；  
     - 将中长期任务（跨数据集验证、重构、自动化等）单列为“未来阶段”。  

---

## 三、与 Codex 协同的推荐模式

1. **GLM：负责叙事与状态快照**  
   - 专注于整理“发生了什么、现在到哪里了”，用自然语言描述进展与问题。  

2. **Codex：负责结构化计划与代码修改**  
   - 统一维护 baseline 表、MVP、代码规范与具体修复方案。  

3. **彼此引用**  
   - GLM 文档在需要“最终数字、配置”的地方引用 Codex 文档；  
   - Codex 在需要说明“执行过程与历史状态”时引用 GLM 文档。  

---

本 TODO/notes 文件可视为 12‑01 之后 GLM 行为的“轻量守则”，有助于后续多轮 plan/summary 保持与当前代码和统一基线的一致性。  

