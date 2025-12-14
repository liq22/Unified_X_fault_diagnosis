# 结果表模板（顶刊口径：PHM-Vibench多数据集 + 解释评估）

**更新日期**：2025-12-14  
**用途**：统一7篇Paper对外汇报的表格字段与口径，避免不同文档出现准确率/参数量/seed不一致。  

---

## Table 1：数据集与任务设定（必做）

| Dataset | Dataset ID（PHM-Vibench） | Task | #Classes | Sampling | Window/Stride | Train/Val/Test | Notes |
|---|---|---|---:|---|---|---|---|
| CWRU | RM_001_CWRU | Classification | TBD | TBD | TBD | TBD | 驱动端/风扇端通道口径 |
| XJTU | RM_002_XJTU | Classification | TBD | TBD | TBD | TBD | 工况/域划分口径 |

---

## Table 2：主结果（性能）（必做）

> 必须包含：`mean±std`（至少3-seed）或 `95%CI`；并注明训练协议（in-domain / cross-domain）。

| Model | Paper | Dataset | Protocol | Acc↑ | F1↑ | Params↓ | Latency(ms)↓ | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| TSPN | baseline | CWRU | in-domain | TBD | TBD | TBD | TBD | |
| Fusion1D2D | Paper1 | CWRU | in-domain | TBD | TBD | TBD | TBD | |

---

## Table 3：消融实验（必做，按Paper定制）

| Paper | Ablation | Dataset | Metric | Result | Δ vs Full | Notes |
|---|---|---|---|---:|---:|---|
| Paper1 | 去掉几何对齐 | CWRU | Acc | TBD | TBD | |
| Paper4 | experts=3/5/8 | THU/XJTU | Acc | TBD | TBD | 同训练轮数与早停口径 |

---

## Table 4：可解释性评估（必做）

> 与 `Paper/doc/12_14/codex/explainability_eval_protocol.md` 对齐。

| Explainer Type | Paper | Dataset | Faithfulness(Del@k)↑ | Stability@σ↑ | Consistency↑ | Sparsity↓ | Time(ms)↓ | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| intrinsic | Paper7 | synthetic | TBD | TBD | TBD | TBD | TBD | 物理一致性另表 |
| post-hoc | baseline | CWRU | TBD | TBD | TBD | TBD | TBD | Captum/IG |

---

## Table 5（可选但顶刊推荐）：跨数据集泛化（LODO / Transfer）

| Train Dataset(s) | Test Dataset | Model | Acc↑ | F1↑ | Explain-Stab↑ | Notes |
|---|---|---|---:|---:|---:|---|
| CWRU | XJTU | Fusion1D2D | TBD | TBD | TBD | transfer baseline需对齐 |
| XJTU | CWRU | FuzzyLogic | TBD | TBD | TBD | |

---

## Table 6（可选）：人类一致性/用户研究（Paper3为必做）

| Condition | Task | Metric | Result | N | Notes |
|---|---|---|---:|---:|---|
| 无解释 | 诊断+建议 | 时间(s)↓ | TBD | TBD | |
| 可视化 | 诊断+建议 | 正确率↑ | TBD | TBD | |
| 文本解释（LLM） | 诊断+建议 | 理解度(1-5)↑ | TBD | TBD | 需防幻觉策略说明 |

