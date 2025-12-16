# 2025-12-15：6篇Paper 顶刊执行计划（Planner Agent / 待确认）

> 目标：顶刊（用户确认）  
> 数据口径：PHM‑Vibench 多数据集验证（至少 CWRU + XJTU，外加 THU/PHM 子集按可用性补齐）  
> 范围：Paper1–Paper6（按 `Paper/doc/README_11_25.md` 官方顺序）

---

## A. 一句话定位（1–2行）
以“统一可解释评估协议 + 多数据集 multi‑seed 复现”构建顶刊级证据链：让 6 篇解耦子论文的每一个关键数字都可追溯到结果真源、复现命令与配置快照，从而同时提升可信度与可投稿性。

---

## B. 论文骨架（严格分节）

> 每篇子论文都按该骨架写；Method/Experiments/Results 用各自方法替换；所有指标与图表输出格式统一。

- Title：任务 + 方法关键字 + explainability 关键字 + multi‑dataset
- Abstract：Problem→Key idea→Key results（性能+faithfulness/stability+复现信息）→Contributions
- Keywords：XFD/PHM/time‑series/faithfulness/stability/generalization 等
- Introduction：为什么需要“可解释且可复现”；研究缺口；贡献与边界
- Methodology：问题定义；模型结构；解释生成机制；训练推理流程
- Equations：核心损失/约束；解释性指标定义（用 $$...$$）
- Results：多数据集+多seed主结果；解释性定量；消融；鲁棒性；失败案例
- Discussion：机制解释；对比SOTA；消融洞察；失败分析；局限性
- Conclusion：总结贡献；复现入口；未来工作
- References：只保留可核验BibTeX；不确定标【TO-VERIFY】直到核验

---

## C. 一步步执行计划（核心）

| Phase | Step | Objective | Inputs | Actions | Deliverables | Acceptance Criteria | Risk & Mitigation |
|---|---:|---|---|---|---|---|---|
| P0 | S1 | 锁定“结果真源”与口径 | 6篇README/脚本/现有results | 定义唯一结果表真源（CSV/JSON）生成入口；统一seed列表与数据集ID；所有文档只引用真源 | 统一结果表（CSV）+ 生成脚本 + 口径说明 | 任意数字可追溯到文件+命令+配置快照 | 口径漂移→强制“真源表+生成脚本”门禁 |
| P0 | S2 | 修复可复现实验入口 | Paper1/4/5 脚本 | 去硬编码 conda/绝对路径；修复相对路径；确保每篇至少1条最小可跑命令 | 6条最小复现命令（每篇1条）+ 输出样例 | 每条命令能产出结果文件（含seed/数据集） | 环境差异→给出环境约束与替代入口 |
| P0 | S3 | explainability评估协议统一落地 | 统一评估协议文档+实现 | faithfulness/stability/efficiency 输出格式统一；跨模型复用 | explainability输出（CSV/PNG） | 每个模型至少：1条faithfulness曲线+1个stability统计 | 指标不一致→只允许统一入口输出 |
| P1 | S4 | 多数据集主实验 | 数据路径/划分/预处理 | 在 CWRU/XJTU/THU 子集跑主模型与关键baseline；每个≥3 seed | 主结果表（mean±std/CI）+ 日志 | 覆盖≥3数据集；每个结果含seed统计 | 算力不足→先2seed短跑再补齐 |
| P1 | S5 | 消融与鲁棒性/失败案例 | 模块开关+噪声脚本 | 必要消融；噪声/工况变化；收集失败样本并解释 | 消融表+鲁棒性图+失败案例集 | 每篇≥1张消融表、≥1张鲁棒性图、≥1个失败案例 | 结果不显著→准备“解释可信/可靠性”叙事 |
| P2 | S6 | 投稿包收敛 | manuscript/figures/bib | 统一结构补齐；图表清单；bib核验；repro checklist | 每篇投稿包（可编译/可导出） | 6篇都能导出PDF或定稿；无未核验引用 | 引用缺失→未核验不进入主稿 |

---

## D. 预期结果表（必须是表格）

| Experiment | Metric | Baseline | Expected Trend | Minimum Acceptable | Notes |
|---|---|---|---|---|---|
| E1 多数据集主性能 | Acc/F1/AUC | ResNet/Transformer/SVM等 | 性能↑或持平且解释性↑ | 主模型不显著劣于强baseline（给CI） | PHM‑Vibench口径 |
| E2 Multi‑seed稳定性 | mean±std, 95% CI | 同模型不同seed | std↓、CI窄 | ≥3 seed 可复现 | Paper1需先修入口 |
| E3 Faithfulness | Del@k / AOPC | Random mask | Del@k下降更快 | 显著优于随机 | 输出曲线+统计 |
| E4 Stability | Spearman/IoU@noise | Random/不稳定解释 | 稳定性↑ | ≥0.8（或给阈值理由） | 多σ扰动 |
| E5 Efficiency | ms/sample, VRAM | SHAP/LIME | 时间↓/资源↓ | 满足工程约束 | 记录硬件 |
| E6 Ablation | ΔAcc/ΔXAI | 去模块版本 | 关键模块贡献为正 | 至少1个模块贡献可量化 | 防“堆模块” |
| E7 Robustness | Acc@SNR/工况 | baseline | 鲁棒性↑ | 0dB不崩溃 | 与稳定性联动 |

---

## E. Discussion 写作模板（可直接填空）

1) Major findings：我们发现【方法X】在【数据集/工况】上实现【性能】并在【faithfulness/stability】达到【数字】，表明【结论】。  
2) Why it works：【机制A】约束【可解释对象】与【物理/语义】一致性，从而提升【可信/泛化】。  
3) Comparison to prior work：相比【SOTA1/2】，我们在【指标】更好/相当，但代价是【开销】。  
4) Ablation insights：【模块1】提升【性能/稳定性】；【模块2】提升【faithfulness】；去掉【模块】导致【现象】。  
5) Failure cases：在【场景】下失败由于【原因】；解释出现【不稳定/不忠实】提示【改进】。  
6) Limitations：受限于【数据/工况/标注】与【协议假设】，影响【外推】。  
7) Future work：优先推进【多数据集补齐】、【解释评估增强】、【工业闭环demo】。

---

## F. 图表清单（面向发表）

- Figure 1：方法总览框架图（每篇一张，统一风格）
- Figure 2：模块细节图（对齐/路由/规则/证据链等）
- Figure 3+：核心结果图（多数据集+误差条/CI）
- Table 1：数据集与设置（PHM‑Vibench口径+划分+预处理）
- Table 2：主结果对比（多seed统计）
- Table 3：消融
- Table 4：可解释性评估（faithfulness/stability/efficiency/understandability）

---

## G. 参考文献与 BibTeX 计划

- 需要检索的文献类型：XAI综述；faithfulness/stability评估；PHM/FD多数据集基准；MoE/NeSy/规则/LLM解释方向SOTA【TO-VERIFY】  
- BibTeX规范：每篇最终输出 `references.bib`；不确定条目保留【TO-VERIFY】直到核验通过；主稿不引用未核验条目。  
- 严禁编造文献：缺少作者/题名/venue/年份/DOI的信息不得进入最终bib。

---

## H. 需要用户确认（必须）

---
✅ 请确认是否按此计划执行：
- 若确认：回复 “YES”
- 若需要修改：回复 “NO” 并指出要改的条目编号（例如 C.P0-S2 或 D.E3）
---

## 【ASSUMPTION】默认假设（如需你纠正）

1) 任务统一为“分类诊断”（非RUL）且以 PHM‑Vibench 为主口径。  
2) 优先把“口径与复现入口修复（P0）”做扎实，再补齐多数据集与图表（P1/P2）。  
3) 6篇指 Paper1–6（不含 Paper7）。  

## 最少还需要你补充的3条信息

1) PHM‑Vibench 的真实数据根目录 + THU_018 的 HDF5 键结构（用于彻底解决 HDF5 vs npy 阻塞）。  
2) 可用 GPU 数量/每天可跑时长（用于排 multi‑seed 与多数据集并行）。  
3) 6篇各自的目标期刊与优先级（Nature MI / IEEE TII / MSSP / TSMC 等的取舍与截止）。  

