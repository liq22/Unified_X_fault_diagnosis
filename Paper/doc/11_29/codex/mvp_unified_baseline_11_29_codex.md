# 统一基线 MVP 规划（截至 2025‑11‑29）

> 目标：在 THU_018_basic 上**稳定跑通统一基线 v1**（TSPN + Fusion1D2D + MoE + OperatorAttention + Fuzzy 简单版），  
> 并在一个统一结果表中给出“可引用的快照结果”，同时让 7 篇 Paper 都有清晰的“基线引用说明”。

---

## 一、MVP 必须交付的 6 件事

按优先级排序，只要完成这 6 项，就视为统一基线 MVP 达成。

### 1. 创建统一基线结果表骨架（P0）

- 文件路径：`Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`  
- 内容要求：  
  - 固定表头：
    ```markdown
    | 模型 | 数据集 | 准确率(快照) | 参数量 | 备注(快照/待复现/已验证) |
    |------|--------|-------------|--------|--------------------------|
    ```  
  - 先填入至少 3 行快照结果：  
    - **TSPN**：历史 baseline（备注：已验证 / 历史结果）；  
    - **Fusion1D2D**：当前在 THU_018_basic 上的最佳 run（备注：快照，需复现）；  
    - **MoE_simple**：当前约 63% 的首个可用结果（备注：快照）。  
  - 后续可追加 OperatorAttention / Fuzzy 行，不是 MVP 硬要求。  

### 2. Fusion1D2D 验证阶段 shape 问题闭环（P0）

- 在仓库根目录 `scripts/` 下创建最小调试脚本，例如：`scripts/debug_fusion1d2d_shape.py`：  
  - 使用 `VbenchDataset` 加载少量 THU_018_basic 数据；  
  - 构造 Fusion1D2D 模型（统一基线配置）；  
  - 手动执行一次与验证阶段等价的 forward/validation_step，重现并修复 shape 错误；  
  - 修复点应集中在 reshape 逻辑与输入长度/通道数对应关系，而非随意改 out_channels。  
- 修复完成后，在 11‑29 的状态记录中写明：  
  - 问题位置（函数/代码行）；  
  - 修复方式（具体逻辑）；  
  - 已通过最小脚本验证。  

### 3. OperatorAttention L1 正则化一次实际调参（P1）

- 在 `configs/unified_baseline/config_OperatorAttention.yaml` 中：  
  - 将 `args.l1_norm` 从 `0.0001` 调整为更小的数值（建议先 `1e-4`），并在 YAML 注释中标注“11‑29 调整”；  
  - 启动一轮短实验（无需跑满 100 epoch），观测：  
    - L1 值范围是否明显降低；  
    - 损失与验证精度是否有明显变化。  
- 在 11‑29 实验状态文档中补一句：  
  - 调整前后 L1 范围；  
  - 是否缓解了“L1 过高”问题；  
  - 对收敛速度的大致影响。  

### 4. FuzzyLogic_simple 跑通一条统一 baseline（P1）

- 使用 `model/FuzzyLogic_simple.py` 和一份统一 baseline 配置（可在 `configs/unified_baseline/` 新建，例如 `config_FuzzyLogic.yaml`）：  
  - 只需完成一条“可工作”的 run：得到 test accuracy 与基本训练日志；  
  - 不强求多次复现或极致性能。  
- 在统一基线结果表中新增一行：  
  - 模型：FuzzyLogic_simple  
  - 备注：快照（首个可工作 baseline）。  

### 5. 3 篇实验型 Paper 的“基线引用 + 快照描述”（P1）

目标 Paper：
- `Paper/1D-2D_fusion_explainable`  
- `Paper/MOE_explainable`  
- `Paper/TII_operator_attention`  

每篇 Paper 在 README 或 research proposal 的“结果/实验”相关小节中增加：

1. **统一引用句**（固定格式）：  
   ```markdown
   统一基线结果见 `Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md` 中的相关行。
   本节仅讨论本方法相对统一基线的差异与优势。
   ```  
2. **当前快照描述**（简要，用自然语言说明目前已有的实验快照）：  
   - Fusion1D2D：当前在 THU_018_basic 上的最佳 run（例如 ~99%），并说明“不稳定 run 仍在分析”。  
   - MoE_simple：当前约 63% 的首个物理约束 MoE baseline。  
   - OperatorAttention：当前训练精度区间（如 ~20%）及 L1 调整的观察。  

> 注意：快照描述要明确说明“当前数据仅为阶段性结果，后续仍需更多重复实验与其他数据集验证”。

### 6. 4 篇工具 / 理论型 Paper 的基线引用模板（P2）

目标 Paper：
- Explainable_FD_Toolkit  
- LLM_Explainable_FD_Toolkit  
- Paper_fuzzy_XFD  
- Neuralsymbolic_theory  

在其 README 或 proposal 中相关章节（实验/应用）添加统一引用句（**不要求立即填具体数值**）：

```markdown
本文中涉及的模型性能对比统一参考 `Paper/doc/11_29/codex/unified_baseline_results_codex_11_29.md`，
本节重点关注本工具/理论在解释性或结构上的贡献。
```

---

## 二、本轮 MVP 明确不做的事情

为保证计划可落地，本轮 MVP 明确 **不作为硬性目标** 的事项（可以在后续阶段单独规划）：

1. 不强制所有模型训练 ≥ 200 个 epoch  
   - 统一基线阶段以“按配置 epoch（如 100）+ 视收敛情况延长”为原则。  

2. 不立即补齐所有传统基线（MCN、TFN 等）的完整矩阵  
   - 可以在统一结果表中暂留空行或“待补齐”备注。  

3. 不要求当前阶段完成跨数据集验证（如 CWRU/XJTU/DIRG）  
   - THU_018_basic 的统一 baseline v1 即可支撑现阶段 Paper 写作。  

4. 不做大规模重构（signal_processing / 专家模块 / 完整 CI/CD）  
   - 当前仅按 `code_placement_guidelines_11_28.md` 的原则做小步归类与标记，不拆大模块。  

5. 不把单次最好结果（如 Fusion1D2D 在 THU_018_basic 上的 99.57%）宣称为“通用 SOTA”  
   - 在 MVP 阶段仅以“某次 run 的亮点数值”形式记录，后续需复现与跨数据集验证后再强化表述。  

---

## 三、建议执行顺序（11‑29 起）

1. **先做 P0 任务**：  
   - 创建统一 baseline 结果表骨架并填入 TSPN / Fusion1D2D / MoE 三行快照；  
   - 写 debug 脚本 + 修复 Fusion1D2D 验证 shape 问题。  

2. **然后推进 P1 任务**：  
   - OperatorAttention L1 调参 + 一次短跑实验记录；  
   - FuzzyLogic_simple 跑出首条统一 baseline 结果。  

3. **最后补齐文档侧 P1/P2 任务**：  
   - 实验型 Paper：写基线引用段落 + 快照描述；  
   - 工具/理论型 Paper：写统一 baseline 引用模板。  

完成以上步骤，即可认为**统一基线 MVP 已达成**，后续可以切换重心到各 Paper 的深入实验设计与写作。  

