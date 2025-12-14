# 7篇Paper现状审计与TODO执行检查（Planner/Executor）

**更新日期**：2025-12-14  
**审计范围**：`Paper/README.md` + 7个子项目 `README.md` + `Paper/doc/12_14/codex/summary_and_todo_7_papers_12_14_codex.md` + `Paper/doc/12_4/glm/*`  
**排序规则**：按 `Paper/doc/README_11_25.md` 的 1→7 子项目顺序；以目录路径为唯一ID。  
**目标档位**：顶刊/顶会（用户确认）  
**数据口径**：PHM-Vibench 多数据集验证（用户确认），至少包含 CWRU、XJTU（可扩展 FEMTO/THU/MFPT/UNSW 等）。  

---

## 0. 总体结论（要点）

1. **入口文档不足**：`Paper/README.md` 目前仅是目录说明，缺少“7篇总览/状态/复现入口/证据链索引”。  
2. **README可审计性不一致**：Paper7（OperatorAttention）有大量可勾选TODO；但 Paper1/4/6 README 缺少勾选清单，无法审计执行情况。  
3. **TODO时间线口径过期**：Paper2（Toolkit）与 Paper3（LLM）README 中路线图仍以 2024 为主，需要升级为 2025 口径并绑定“可复现交付物”。  
4. **顶刊证据链缺口集中在三类**：  
   - 多数据集（PHM-Vibench：CWRU/XJTU/…）泛化与统计显著性（mean±std、95%CI）  
   - 可解释性**定量评估协议**（faithfulness/stability/consistency/efficiency/human agreement 等），不能只做可视化  
   - 统一复现入口（命令+配置+输出目录）与统一结果表真源（避免口径漂移）  

---

## 1. 分Paper审计（现状 + TODO执行差距）

> 说明：这里的“已完成/进行中/未开始”以仓库文档/README勾选为证据；若无勾选，则标注为“**缺少可审计记录**”，后续需在README中补齐勾选清单。

### Paper 1：`Paper/1D-2D_fusion_explainable`

- README现状：结构完整，但**无TODO勾选清单**（`[x]=0, [ ]=0`），无法审计执行情况。  
- 复盘文档口径（12_3/12_4）：主结果已出（99.57%），缺 3-seed 稳定性与多数据集泛化。  
- 关键差距（顶刊）：
  - 多数据集（CWRU/XJTU/THU_006/…）泛化表
  - 3-seed 稳定性（CI/显著性）
  - 跨模态解释的**定量指标**（不仅Grad-CAM图）

### Paper 2：`Paper/Explainable_FD_Toolkit`

- README现状：存在TODO勾选（`[x]=2, [ ]=13`），但路线图年份偏旧，且与 12_4 Toolkit专项TODO存在口径差异。  
- 已有证据：脚本与benchmark目录齐全（`Paper/Explainable_FD_Toolkit/scripts/`、`benchmark_results/`）。  
- 关键差距（顶刊）：
  - 与 Captum/SHAP/LIME 的系统对比（指标/速度/工程性）
  - 工业demo（至少2个可复现脚本+英文图表）
  - 统一结果表真源与一键复现入口固化

### Paper 3：`Paper/LLM_Explainable_FD_Toolkit`

- README现状：存在TODO勾选（`[x]=1, [ ]=18`），但路线图年份偏旧；需要将“用户研究/解释质量评估/安全机制”明确为可验收任务。  
- 关键差距（顶刊）：
  - 解释质量评估协议（human agreement/任务完成时间/错误率等）
  - 幻觉防护与证据链（结构化解释→文本）
  - PHM-Vibench 多数据集场景下的端到端demo

### Paper 4：`Paper/MOE_explainable`

- README现状：内容很丰富，但**无TODO勾选清单**（`[x]=0, [ ]=0`），无法审计执行情况。  
- 复盘文档口径：多seed不稳定是核心风险；需要完成 3/5/8 专家消融与稳定性改进。  
- 关键差距（顶刊）：
  - 稳定性（CV下降）必须给出统计证据链
  - 参数量/准确率口径需要锁定为“统一基线真源输出”（避免历史漂移）
  - 多数据集泛化（PHM-Vibench：CWRU/XJTU/…）

### Paper 5：`Paper/Paper_fuzzy_XFD`

- README现状：TODO勾选较多（`[x]=2, [ ]=66`），但底部时间戳与当前突破不一致，TODO结构偏“从零开发”，与现状（已达70.7%）不匹配。  
- 复盘文档口径：重点是冲击 75%+、安全关键错误案例、跨数据集验证。  
- 关键差距（顶刊）：
  - 安全关键失败案例（2–3个）+ 解释证据链
  - PHM-Vibench跨数据集泛化 + 稳定性统计
  - 与Toolkit接口对齐的规则/隶属度可解释评估

### Paper 6：`Paper/Neuralsymbolic_theory`

- README现状：理论叙事强，但**无TODO勾选清单**（`[x]=0, [ ]=0`），无法审计执行情况。  
- 复盘文档口径：命题体系已有，但命题2需要加强证据链；需要整合论文初稿并补2–3个案例闭环。  
- 关键差距（顶刊）：
  - 命题2实验与图表（可复现脚本+输出）
  - 统一术语/符号并形成可投稿的单一文档版本
  - 与7篇方法的映射验证报告（可运行检查）

### Paper 7：`Paper/TII_operator_attention`

- README现状：TODO勾选完整（`[x]=29, [ ]=45`），可审计。  
- 关键卡点：合成信号验证**尚未实际执行并产出报告/图表**（顶刊顶会证据链缺失）。  
- 关键差距（顶刊）：
  - 合成信号验证报告（8类信号：权重热图/一致性评分/对照理论预期）
  - 定理证明补全（附录结构）
  - 工业数据仅作为可运行性补充（主叙事以理论+合成验证为主）

---

## 2. 统一建议（下一步优先级）

### P0（先把“可复现与可审计”补齐）
- 将 Paper1/4/6 README 增加**可勾选TODO清单**与“状态快照（日期）/复现入口/证据路径”。  
- 将 Paper2/3 README 的“2024路线图”升级为“2025-12-14路线图”，并绑定**验收标准**与**交付物**（脚本/图表/表格/报告）。  
- 固化 PHM-Vibench 多数据集验证协议：至少 CWRU + XJTU；给出跨数据集实验矩阵（同一seed/同一统计口径）。  

### P1（顶刊证据链冲刺）
- Paper1：3-seed + CWRU/XJTU/… 泛化（统计显著性+误差条）  
- Paper4：3/5/8 experts 消融 + 稳定性改进（CV下降）  
- Paper7：合成信号验证实际运行 + 论文级图表与报告  

### P2（工程与论文收口）
- 统一“可解释评估协议”写成可引用文档，并在Toolkit里提供一键跑通入口（或最小复现脚本）。  
- 7篇在各自文件夹放置“核心文档（paper_blueprint.md）”，确保解耦、可验收、可复现。  

