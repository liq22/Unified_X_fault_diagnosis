# 统一故障诊断项目未完事项梳理与 11‑29 执行计划（Codex）

> 参考文档：  
> - `Paper/doc/11_28/glm/execution_plan_unified_baseline_11_28.md`  
> - `Paper/doc/11_28/glm/implementation_plan_11_28.md`  
> - `Paper/doc/11_28/glm/experiment_status_summary_11_28.md`  
> - `Paper/doc/11_28/codex/code_placement_guidelines_11_28.md`  
>
> 目标：把 11‑28 规划中尚未完成或需要调整的事项整理出来，形成 11‑29 后续执行的精简版 Codex 计划。

---

## 一、11‑28 计划中已完成 vs 未完成

### 1. 已完成（无需在 11‑29 再规划）

- ✅ 初始化 & shape 问题排查  
  - Fusion1D2D / MoE / OperatorAttention 初始化错误已通过代码修复 + 缓存清理解决；  
  - OperatorAttention 的 shape 逻辑在 `model/OperatorAttention_simple.py` 中已改为通过截断/补零适配，不再依赖“target_channels=2”的临时修补。  

- ✅ 规范与文档  
  - 代码归属与解耦规范：`Paper/doc/11_28/codex/code_placement_guidelines_11_28.md`；  
  - 7 个 Paper 子 agent 行为规范：`Paper/doc/11_28/claude_agents_instructions_11_28.md` + `.claude/agents/paper-*.md` 已对齐；  
  - GLM 状态与计划文档：`experiment_status_summary_11_28.md`、`execution_plan_unified_baseline_11_28.md`、`next_steps_action_plan.md` 已由 Codex 校对并修订。  

- ✅ 数据接口  
  - `data/vbench_dataset.py` / `data/vbench_utils.py` 已整理为推荐数据入口；  
  - `data/README.md` 已更新，说明与 VBench 上游数据仓的对应关系和配置方法。  

### 2. 未完成 / 需要在 11‑29 继续推进的事项

1. **统一基线结果表（Codex 版）尚未创建实际文件**  
   - 11‑28 计划中多次提到：`Paper/doc/11_28/codex/unified_baseline_results_codex_11_28.md`，目前尚未真正创建；  
   - 统一表的表头结构已在计划中构思，但尚未落地。  

2. **Fusion1D2D 的验证阶段 shape 问题仍需二次确认**  
   - 虽然 simple 版模型初始化与前向已通过测试脚本，但在完整训练/验证流程下，`validation_step` 仍出现过 shape 相关错误；  
   - 需要专门的最小复现脚本 + 针对性修复。  

3. **OperatorAttention 的 L1 正则化尚未实测调整效果**  
   - 配置中 `l1_norm: 0.0001` 仍是原值，调整策略已在文档中规划，但尚未执行与记录效果；  
   - 当前实验中 L1 值偏高的问题在 11‑28 状态报告中仍存在。  

4. **MoE / OperatorAttention / Fuzzy 的统一 baseline 首轮结果尚未系统汇总**  
   - MoE 已有快照结果（约 63% test acc）、OperatorAttention / Fuzzy 有初步运行情况；  
   - 但在 Codex 文档里还没有一张统一对比表，包含“快照 / 待复现 / 已验证”等状态标注。  

5. **Paper 级引用统一 baseline 的段落尚未逐个写入**  
   - 1D‑2D / MoE / OperatorAttention / Fuzzy / Toolkit / LLM / NeSy 7 篇 Paper 的 proposal/README 中，还没有统一格式的“基线引用段落”；  
   - 目前只是计划层面认为“应当这样做”。  

---

## 二、11‑29 之后的 Codex 精简执行计划

> 思路：只保留**当前阶段必须做、且 Codex 能直接推进**的事项，其他保持为中长期规划。

### 任务 A：创建统一基线结果表骨架（高优先级）

- 路径：`Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`  
- 内容要求：  
  - 复制 11‑28 计划中的表头设计，稍作精简：  
    ```markdown
    | 模型 | 数据集 | 准确率(快照) | F1(可选) | 参数量 | 备注(快照/待复现/已验证) |
    |------|--------|-------------|---------|--------|--------------------------|
    ```  
  - 先只填入已有“快照级”结果：TSPN（历史 baseline）、Fusion1D2D、MoE、OperatorAttention、Fuzzy（如有）；  
  - 在备注列明确写上“快照 / 待复现 / 已验证”三类状态，不对数值下结论。  

### 任务 B：Fusion1D2D 验证阶段 shape 问题专项修复

- 在仓库根目录下新增/完善一个最小复现脚本：例如 `scripts/debug_fusion1d2d_shape.py`：  
  - 使用 `VbenchDataset`（少量样本）+ 当前 `config_Fusion1D2D.yaml`；  
  - 仅跑一两个 batch 的 `validation_step`，定位 `shape '[64, 3, -1]'` 的准确来源；  
  - 修复点优先集中在：输入长度/通道数的 reshape 逻辑，而不是再改 out_channels 或 target_channels。  
- 修复完成后，在 `experiment_status_summary_11_28.md` 或新的 11‑29 状态报告中补一句简短记录（修复点 + 验证方式）。  

### 任务 C：OperatorAttention L1 正则化调参与记录

- 在 `configs/unified_baseline/config_OperatorAttention.yaml` 中：  
  - 将 `args.l1_norm` 从 `0.0001` 降低为 `1e-4` 或 `1e-5`（建议先 1e-4），并在 YAML 注释中注明“11‑29 调整，以降低 L1 值过高问题”；  
  - 新开一次简短实验（可只跑若干 epoch），观察 L1 值与损失/精度曲线变化；  
- 在 11‑29 的简短实验状态笔记中记录：  
  - 调整前/后的 L1 值范围；  
  - 是否缓解了“L1 过高”问题；  
  - 是否对收敛速度有明显影响。  

### 任务 D：为 7 个 Paper 建立基线引用模板（文档级）

- 在每个 Paper 的 README 或 proposal 中，统一添加一段模板文本（先不强制填具体行号）：  
  ```markdown
  > 统一基线结果参考：`Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`。  
  > 本节仅讨论本方法相对统一基线的差异与优势。
  ```  
- 此步骤可以先从 3 篇实验驱动型 Paper 开始（1D‑2D、MoE、OperatorAttention），其他 4 篇作为后续跟进。  

---

## 三、对 GLM 计划的 Codex 审阅结论（简要）

1. 已在 11‑28 阶段修正的内容：  
   - `execution_plan_unified_baseline_11_28.md` 中对 Fusion1D2D 配置文件名、OperatorAttention L1 调整等细节的错误已修正；  
   - “强制 200 epoch” 等过强约束已改为“按配置 epoch + 视收敛情况调整”；  
   - 一次性大重构信号处理/专家模块的表述已调整为“小步梳理 + 按需抽取”，并与 `code_placement_guidelines_11_28.md` 对齐。  

2. 仍然需要 Codex 自己推进的部分（已体现在上面的 A–D 四个任务中）：  
   - 统一 baseline 结果表实际文件尚未创建；  
   - Fusion1D2D 的验证 shape 问题尚未以最小脚本形式彻底定位；  
   - OperatorAttention 的 L1 调参与效果记录仍停留在计划层；  
   - 7 个 Paper 对统一基线的“引用段落”还未写入各自文档。  

3. 其他 GLM 计划中提到的“容器化/完整 CI/CD/长时间 200 epoch 训练”等项目，现阶段视为中长期规划，不纳入 11‑29 的硬性执行范围，由后续阶段视资源再决策。  

---

本文件仅作为 11‑29 起 Codex 视角下的“未完事项清单 + 精简计划”。  
具体最小可行版本（MVP）的目标和 6 个关键任务，详见：  
`Paper/doc/11_29/codex/mvp_unified_baseline_11_29_codex.md`。实际执行时，可根据 GPU/时间情况优先完成 MVP 中的 P0/P1 任务。  

