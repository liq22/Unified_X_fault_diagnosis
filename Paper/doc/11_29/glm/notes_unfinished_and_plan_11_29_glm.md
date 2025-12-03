# GLM 视角下 11‑28 计划未完事项与 11‑29 建议

> 说明：本文件从 GLM 生成文档的视角，梳理哪些 11‑28 的计划项尚未完全覆盖，以及 11‑29 可以如何继续推进。Codex 的更严格版本参考：`Paper/doc/11_29/codex/unfinished_tasks_and_plan_11_29_codex.md`。

---

## 一、11‑28 GLM 计划中已被 Codex 校正的部分

1. **命令与配置名修正**  
   - 早期计划中使用了不存在的 `config_Fusion1D2D_simple.yaml`，Codex 已确认统一基线使用 `configs/unified_baseline/config_Fusion1D2D.yaml`，且 `main.py` 中已绑定简化版 Fusion1D2D 模型。  
   - 部分计划示例使用 `--config_file` 参数，Codex 已统一为 `--config_dir`（与当前 `main.py` 定义一致）。  

2. **OperatorAttention shape 修复策略更新**  
   - 旧计划建议直接修改 `OperatorAttention_simple.py` 中的 `target_channels`，当前代码已采用更稳健的“截断/补零 + reshape”方案，无需再按旧建议修改。  

3. **过强的训练长度与重构目标调整**  
   - “至少 200 epoch” 与“一次性合并 signal_processing/专家模块代码”等目标已被 Codex 调整为：  
     - 先按配置 epoch（如 100）+ 根据收敛情况决定是否延长；  
     - 优先梳理重复点和接口，在遵守代码归属规范前提下，小步抽取公共模块。  

---

## 二、GLM 计划中仍需实际完成的关键事项

1. **统一基线结果表落地**  
   - 虽然计划中多次提到 `unified_baseline_results_codex_11_28.md`，实际文件尚未创建；  
   - 11‑29 建议在 Codex 侧创建 `Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md` 作为首次骨架。  

2. **Fusion1D2D 验证阶段 shape 问题闭环**  
   - GLM 报告中仅记录了错误信息与大致位置（`validation_step`），尚未给出最终修复确认；  
   - 建议在 11‑29 通过最小脚本复现 + 修复，并在新的状态报告中写明“问题位置 + 修复方法 + 验证方式”。  

3. **OperatorAttention L1 正则调参效果记录**  
   - 当前只在文档层面提出“将 L1 降低一个数量级”，尚无实验结果对比；  
   - 建议做一次短跑实验（少量 epoch），用图/表记录调整前后 L1 值与性能变化趋势。  

4. **MoE / OperatorAttention / Fuzzy 等模型的统一 baseline 快照整合**  
   - GLM 文档中已有零散的结果（如 MoE 的 63.04%），但尚未在单一表格中呈现；  
   - 建议在 11‑29 的统一结果表骨架中先以“快照”的形式写入，后续再标记哪些需要复现。  

---

## 三、11‑29 GLM 侧的具体建议动作

1. **与 Codex 的结果表保持同步**  
   - 当 Codex 创建 `unified_baseline_results_codex_11_29.md` 后，GLM 可以在自己的文档中：  
     - 引用这张表作为“统一 baseline 来源”；  
     - 在各 Paper 的说明文档中使用统一格式引用，而不是再各写一套表。  

2. **实验状态更新方式**  
   - 建议 GLM 在后续状态报告中：  
     - 区分“快照（snapshot）”和“最终结果（baseline v1/v2）”；  
     - 对于需要进一步验证的数据，明确写 “待复现” 或 “尚需交叉数据集验证”。  

3. **避免重复提出已被 Codex 修正的操作**  
   - 例如不再建议修改 `OperatorAttention_simple.py` 的 `target_channels`，避免与当前修复方案冲突；  
   - 避免在新计划中继续使用 `--config_file` 或不存在的配置名。  

---

本文件主要作为 GLM 视角的“自我校对笔记”。实际执行与代码修改以 Codex 版本计划为准。  

