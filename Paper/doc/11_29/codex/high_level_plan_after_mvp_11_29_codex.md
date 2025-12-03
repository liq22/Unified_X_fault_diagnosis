# 统一基线 MVP 完成后的后续规划（2025‑11‑29）

> 本 plan 用于在统一基线 MVP 完成后，指导后续的实验深化、论文收紧和平台固化工作。  
> 搭配阅读：  
> - `mvp_unified_baseline_11_29_codex.md`（MVP 目标与 6 个关键任务）  
> - `unified_baseline_results_codex_11_29.md`（当前基线结果快照）

---

## 一、短期（3–5 天）：在统一基线上完成第一轮“有效实验”

### 1. 统一基线 v1 完整化

- 在 `unified_baseline_results_codex_11_29.md` 中补齐：  
  - FuzzyLogic_simple 的一条可用结果（THU_018_basic）；  
  - OperatorAttention_simple 在降低 L1 正则后的一条更新快照。  
- 对 TSPN / Fusion1D2D / MoE_simple / OperatorAttention_simple / FuzzyLogic_simple 做一次**定性稳定性评估**：  
  - 关注种子间方差、是否存在 NaN 运行、验证/测试指标是否基本一致。  

### 2. 三篇实验型 Paper：各做一组“最小可发表实验”

- 📘 1D‑2D Fusion  
  - 至少完成：  
    - TSPN vs Fusion1D2D 在 THU_018_basic 上的性能曲线（训练/验证准确率）；  
    - 至少一个模态贡献可视化示例（1D vs 2D 在某工况下的贡献度或注意力权重图）。  

- 🟠 MoE_explainable  
  - 重点围绕物理约束与专家解释性：  
    - 生成专家激活热力图（故障类型 × 专家）；  
    - 选 1–2 个典型样本，画出路径签名（专家组合轨迹）。  

- 🔴 TII_operator_attention  
  - 完成一组最小对比：  
    - baseline TSPN vs OA‑TSPN 的性能/复杂度对比（至少在 THU_018_basic 上）；  
    - 导出一张算子权重热力图（FFT/HT/WF/I 权重随样本/故障类型变化）。  

### 3. 文档侧做一次“小收口”

- 在上述三篇 Paper 的 README 或 research proposal 中：  
  - 明确列出已经完成的图表（图号/表号 + 对应实验）；  
  - 用 TODO 列出仍需补齐的实验（例如更多 seed、额外数据集）但不展开细节。  

---

## 二、中期（1–2 周）：按论文方向收紧到“投稿草稿”层面

### 1. 选择优先投稿顺序

- 第一批（优先完成）：  
  - 📘 1D‑2D Fusion  
  - 🔴 OperatorAttention  
- 第二批：  
  - 🟠 MoE_explainable  
  - 🩷 Paper_fuzzy_XFD  
- 第三批：  
  - 🟢 Explainable_FD_Toolkit  
  - 🟣 LLM_Explainable_FD_Toolkit  
  - 🟦 Neuralsymbolic_theory  

### 2. 第一批两篇：构建“完整实验集”

- 1D‑2D Fusion  
  - 在 THU_018_basic 的基础上：  
    - 补齐 2–3 个关键消融（无对齐 / 少一层对齐 / 全对齐）；  
    - 在至少一个额外数据集（如 CWRU 或 XJTU）上做一次简化实验，验证趋势一致。  

- OperatorAttention  
  - 设计 Self‑Attention vs OperatorAttention 的对比（哪怕是简化 Self‑Attention）：  
    - 性能对比（Acc/F1）；  
    - 复杂度对比（时间/显存）；  
  - 做一组长序列复杂度实验，画出 L vs 时间/显存曲线。  

### 3. 论文写作结构统一

- 对于第一批两篇 Paper，建议统一采用：  
  1. 引言（问题与动机）  
  2. 方法（结构 + 关键公式）  
  3. 实验设置（统一基线说明 + 数据集 + 配置）  
  4. 主要结果（引用统一 baseline 表，讨论相对变化）  
  5. 可解释性分析（图例与案例）  
  6. 消融研究（少而精）  
  7. 讨论与局限  

---

## 三、长期（若干周）：平台完备与交叉验证

### 1. 完成经典基线矩阵

- 在统一 baseline 表中逐步补齐：  
  - ResNet / SincNet / WKN / MCN / TFN 等传统模型的 THU_018_basic 结果；  
  - 不做极限调参，仅保证配置清晰、结果可复现。  

### 2. 跨数据集与鲁棒性验证

- 选 2–3 个代表方法（如 TSPN / Fusion1D2D / MoE / OperatorAttention）：  
  - 在 CWRU、XJTU、DIRG 等数据集上各跑一组简化实验；  
  - 做噪声/负载扰动实验，构建“鲁棒性表”（如不同 SNR 或工况下的性能变化）。  

### 3. 平台化与轻量自动化

- 在现有脚本基础上：  
  - 把统一 baseline 一整套实验组织成简单的 runner（bash 或 python driver），可一键启动/恢复主要实验；  
  - 不追求复杂 CI/CD，仅验证：  
    - 环境可安装；  
    - 主入口脚本可运行；  
    - 少量关键测试通过（如最小 forward）。  

---

本 plan 是在统一基线 MVP 之后的“第二层规划”。执行优先级：  
- 先完成短期 3–5 天内的实验和文档收口；  
- 再按论文优先级推进中期目标；  
- 长期项目视时间与资源再细化。  

