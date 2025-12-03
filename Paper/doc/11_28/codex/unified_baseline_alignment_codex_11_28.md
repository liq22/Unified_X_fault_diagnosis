# 统一基线实验 Codex 对齐说明（2025-11-28）

> 对齐对象：  
> - `Paper/doc/11_28/glm/UNIFIED_BASELINE_COMPLETION_REPORT.md`  
> - `Paper/doc/11_28/glm/unified_baseline_experiment_report.md`  
>
> 目标：从 Codex 视角校对当前实验与文档状态，澄清“已完成 vs 计划中”的边界，并为后续 Agent 提供无歧义的执行指引。

---

## 1. 状态总体判断

1. GLM 文档中关于**测试脚本与 README_scripts.md** 的描述基本准确：  
   - 7 个 Paper 子项目均已存在 README_scripts.md；  
   - Fusion1D2D / MoE / OperatorAttention / Fuzzy / Explainable Toolkit / LLM Interface / NeSy 均已具备最小测试脚本或等价测试入口。  
2. GLM 将“统一基线实验系统建设”视为 Phase 2 完成、Phase 3 进行中，这与 Codex 视角一致：  
   - Phase 1/2 更偏向代码与文档建设；  
   - Phase 3 主要是各模型统一基线实验的**批量运行与结果汇总**。  
3. 需要强调的一点是：  
   - 一些性能数字（如 Fusion1D2D 全部 5 次迭代、MoE 完整性能等）在 Codex 侧尚未完全核实；  
   - 在 Codex 文档中，这些结果应被视为“GLM 报告的当前实验状态”，而非 Codex 已复核的最终结论。  

---

## 2. 关键一致点与注意事项

### 2.1 一致点

- **Fusion1D2D**  
  - Identity 初始化问题已修复；  
  - 通过最小前向测试，统一接口可用；  
  - 存在至少一条训练曲线，验证性能良好。  

- **MoE_simple**  
  - 已有一次成功实验（测试准确率约 63%–74%）；  
  - 当前统一基线默认应使用 `MoE_simple` 版本。  

- **OperatorAttention**  
  - 简化版 OperatorAttention 已能在单独脚本中跑通；  
  - 统一训练封装 `OperatorAttentionNetwork` 已在 `main.py` 中实现并经最小前向验证。  

- **测试脚本与 README_scripts**  
  - GLM 文档中列举的 7 个 Paper 对应测试脚本基本存在；  
  - Codex 已为 Fusion1D2D / MoE / OperatorAttention 补充或修正了统一基线测试脚本。  

### 2.2 注意事项 / 潜在偏差

1. **“已完成” vs “正在进行”**  
   - GLM 报告中部分模型的“完成状态”包含正在训练的 run 或 NaN 但精度尚高的 run；  
   - 在 Codex 视角，应将这些状态标记为“已启动/部分完成”，而非“完全收敛+稳定”。  

2. **性能数字的使用**  
   - 所有具体准确率/损失数值，在论文或统一比较表中引用时，应显式标注来源（GLM 执行批次）并在 Codex 审阅后再次确认；  
   - Codex 文档不应将 GLM 数字直接视为“最终 benchmark”，而是“当前运行快照”。  

3. **MCN / TFN / 其他基线**  
   - GLM 报告中提到 MCN、TFN 已“验证为空文件”；  
   - 从 Codex 角度：这是一个待修复的配置/实现问题，需要在后续统一基线计划中单独列出。  

---

## 3. 对后续 Agent 的执行指引

### 3.1 总体原则

1. **不直接修改 GLM 报告**：  
   - GLM 文档视为“自然语言记录 + 运行日志”，Codex 不在本阶段直接编辑；  
   - 如需修正或补充信息，应在 Codex 自己的文档中说明（例如本文件）。  

2. **所有“完成”结论需在 Codex 视角二次确认**：  
   - 在撰写综合结论或论文摘要时，必须以 Codex 自己汇总的结果文档为依据；  
   - GLM 提供的数值可用作参考或对比。  

### 3.2 针对具体 Agent 的动作建议

1. `paper-1d2d-fusion`  
   - 任务：对照 GLM 的 Fusion1D2D 实验描述，在 1D-2D Paper 的 proposal 中补全预期表格/图表结构，但**不直接引用数字**；  
   - 后续：等统一 baseline 表真正汇总后，再由 Codex 将最终数值填入。  

2. `paper-moe-expert`  
   - 任务：将 GLM 报告中 MoE 的性能描述，转化为“预期对比表结构”（如 vs TSPN/Resnet），暂不写入具体数值；  
   - 后续：在统一 baseline 结果稳定后再决定哪些数值写进论文草稿。  

3. `paper-operator-attention`  
   - 任务：参考 GLM 对 OperatorAttention 的训练进度描述，在 TII 论文的实验计划中加入“统一 baseline 对齐”小节；  
   - 标明：当前进度=已跑部分 epoch，最终性能需待完整训练完成后更新。  

4. 其他 Paper Agents  
   - Explainable_FD_Toolkit / LLM_Interface / Fuzzy-XFD / NeSy：  
     - 可以将 GLM 文档中提到的“脚本已存在、功能已验证”写入自身 README 的“工程实现状态”部分；  
     - 避免在论文贡献中把“脚本存在”本身当作主要创新点，而是作为支撑工具。  

---

## 4. 后续 Codex 文档与计划

1. 若统一基线实验进一步推进（例如全部模型完成 ≥3 次重复）：  
   - 建议在 `docs/` 或 `Paper/doc/` 下新增 `unified_baseline_results_codex_*.md`，由 Codex 独立汇总最终对比表。  

2. 对于 MCN/TFN 等当前“空文件/未实现”的模型：  
   - 建议在下一轮计划文件中（例如 `plan_baseline_and_integration_11_29_codex.md`）单独列为高优先级待补齐项。  

3. 对 GLM 与 Codex 的协同建议：  
   - GLM 继续负责详细运行日志与自然语言总结；  
   - Codex 负责结构化计划、接口规范和最终对比表，以此区分职责，减少歧义。  

