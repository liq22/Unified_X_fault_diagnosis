# UXFD（7篇可解释生态）顶刊总控论文：执行蓝图（Planner/Executor）

**更新日期**：2025-12-14  
**目标档位**：顶刊/顶会（用户确认）  
**数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU；可扩展 FEMTO/THU/MFPT/UNSW）  
**定位**：不直接写正文，而是把“论文可投稿所需证据链”拆成可执行、可验收、可复现的工程任务。  

---

## 1) 一句话定位

问题：可解释故障诊断碎片化、难复现、难量化对比 → 现在：PHM-Vibench统一接口成熟、7条方法线已具备生态 → 核心思路：统一接口 + 统一评估协议 + 多方法族证据链 → 收益：性能/泛化/解释可靠性/复现能力同时可投稿。

---

## 2) 顶刊“证据链最小集合”（必须满足）

### 2.1 可复现三件套
- 配置文件（原样保存）
- 运行命令（含seed、CUDA）
- 输出目录（日志/表格/图表/metrics.json）

### 2.2 统计口径
- 性能：至少 3-seed mean±std 或 95%CI
- 泛化：至少 CWRU 与 XJTU 两个数据集

### 2.3 可解释性不止可视化
至少包含：
- Faithfulness（Deletion/Occlusion）
- Stability（扰动一致性）
- Efficiency（解释耗时）

可选增强（顶刊强烈建议）：
- Consistency（跨样本/跨域一致性）
- Human agreement（用户研究/专家一致性）

协议文档：`Paper/doc/12_14/codex/explainability_eval_protocol.md`

---

## 3) 论文结构（写作时的“证据→产物”映射）

- Introduction：引用“碎片化/不可复现”痛点 → 证据：7篇README与统一基线；产物：贡献点列表  
- Methodology：三层架构 + 统一接口 + 统一评估协议 → 证据：Toolkit接口/脚本；产物：Figure 1/2 + Table(接口)  
- Results：性能表 + 解释评估表 + 跨数据集泛化表 → 证据：统一输出（tables_template）  
- Discussion：机制解释 + 消融 + 失败案例 → 证据：消融与案例研究素材  

结果表模板：`Paper/doc/12_14/codex/results_tables_template.md`

---

## 4) 执行路线（强制分阶段）

### P0：对齐口径（1–2天）
- 产物：`status_audit_7_papers.md`（已生成）、`repro_index.md`（已生成）
- 验收：7篇都能在README中指出唯一复现入口与证据路径（不要求此刻跑完，但路径必须真实存在）

### P1：README与核心文档收口（2–4天）
- 产物：每篇 `paper_blueprint.md` + README更新（含可勾选TODO/复现入口/证据链）
- 验收：7篇README全部可审计（存在 `[x]/[ ]` 清单）并且引用统一评估协议

### P2：顶刊关键实验补齐（按优先级滚动）
- Paper1：3-seed + 多数据集泛化  
- Paper4：专家消融 + 稳定性改进  
- Paper7：合成信号验证 + 图表/报告  
- 验收：输出能填满 Table 2/4/5 的最低行数，并可复现

---

## 5) 目标期刊/会议（候选，不写死）

> 每篇Paper可独立投稿；总控论文可走“系统/基准/统一框架”路线。

- 顶刊候选（示例）：Nature Machine Intelligence / Science Advances / TPAMI（需根据投稿策略再定）
- 顶会候选（示例）：NeurIPS / ICML / ICLR（理论/系统轨道）
- 工业信息/信号方向顶刊：TII / TSP / TNNLS（按Paper定位选择）

---

## 6) 下一步（由Executor执行）

- 按 `repro_index.md` 对齐每篇Paper的“最小复现命令”
- 更新 `Paper/README.md` 为总入口（7篇顺序+链接+证据链）
- 在7篇目录内生成 `paper_blueprint.md`（解耦核心文档）

