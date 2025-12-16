# 2025-12-15：6篇Paper 现状审计（Review）与缺口清单（codex）

> 范围：按 `Paper/doc/README_11_25.md` 的官方顺序，审计 Paper1–Paper6（不含 Paper7 Operator Attention）。  
> 目标：检查 GLM 汇总的 TODO/现状是否与仓库真实文件、脚本、结果一致；找出阻塞项与优先修复路径（顶刊口径）。

---

## 总览结论（先看这段）

1) **口径漂移与“文档超前于实现”仍在**：多篇 GLM 文档给出“已完成的稿件/脚本/数字”，但仓库里要么路径不存在，要么结果文件显示相反结论。  
2) **Paper1 稳定性实验入口是硬阻塞**：现有稳定性脚本产出“全失败”，且配置路径指向不存在的 `configs/unified_baseline/...`，需先修复复现入口才能谈多数据集与顶刊证据链。  
3) **Paper5（Fuzzy‑XFD）存在“70.7%蓝图 vs 47%现实”的重大矛盾**：必须先用统一数据口径 + multi‑seed 真跑出可复现结果，再决定是否继续以“70.7%突破”作为叙事核心。  
4) **Paper2（Toolkit）更接近“工程完成”，但论文稿件入口与文档描述不一致**：建议锁定唯一稿件真源（`manuscript/draft_md/draft.md` 或补齐 `manuscript/drafts/*`），并把所有结果与图表回链到该真源。  

---

## Paper 1：`Paper/1D-2D_fusion_explainable`

### 现状（仓库证据）
- 目录结构完整，包含 `manuscript/paper.md`、多数据集配置（CWRU/XJTU/THU_006），以及稳定性实验记录目录：`Paper/1D-2D_fusion_explainable/`。
- 但稳定性测试汇总显示：三组 seed **全部失败**：`Paper/1D-2D_fusion_explainable/experiments/stability_test/stability_test_summary.md:1`。

### 关键问题（阻塞）
- 稳定性脚本硬编码配置路径 `configs/unified_baseline/config_Fusion1D2D.yaml`：`Paper/1D-2D_fusion_explainable/scripts/run_3seed_stability_test.py:18`，与 Paper1 自身配置目录（`Paper/1D-2D_fusion_explainable/configs/`）不对齐。
- 稳定性脚本还硬编码 conda 环境与 `main_com.py` 入口，导致复现对机器环境强依赖（不可顶刊）。

### 建议（优先任务）
- 先修复 **最小可复现实验入口**：确保“1条命令 + 1个配置 + 1个结果文件”闭环（再谈3‑seed、多数据集、解释评估）。

---

## Paper 2：`Paper/Explainable_FD_Toolkit`

### 现状（仓库证据）
- Toolkit 的 benchmark、可视化、对比 Captum 的脚本/结果目录非常齐全：`Paper/Explainable_FD_Toolkit/benchmark_results/`、`Paper/Explainable_FD_Toolkit/results/`、`Paper/Explainable_FD_Toolkit/scripts/`。
- 论文稿件目前以 `Paper/Explainable_FD_Toolkit/manuscript/draft_md/draft.md` 为主（draft 形式存在）。

### 关键问题（口径不一致）
- GLM 文档声称已产出 `paper.md/experiments.md/references.bib`（并以项目根路径描述），但仓库实际缺少这些根文件：  
  - 文档声称：`Paper/doc/12_14/glm/explainable_fd_toolkit_paper_status_12_14.md:165`  
  - 实际缺失：`Paper/Explainable_FD_Toolkit/paper.md`、`Paper/Explainable_FD_Toolkit/experiments.md`、`Paper/Explainable_FD_Toolkit/references.bib`

### 建议（优先任务）
- 统一“论文真源入口”：要么补齐 `Paper/Explainable_FD_Toolkit/manuscript/drafts/{paper,experiments,references}.(md|bib)`，要么在 README 明确 `manuscript/draft_md/draft.md` 是唯一真源，并将图表/结果回链到真源章节。

---

## Paper 3：`Paper/LLM_Explainable_FD_Toolkit`

### 现状（仓库证据）
- `manuscript/drafts/` 下有完整 `paper.md`、`experiments.md`、`references.bib`：`Paper/LLM_Explainable_FD_Toolkit/manuscript/drafts/paper.md`。
- `code/llm_explainable_toolkit/` 与 `code/tests/` 存在，具备最小可运行基础。

### 关键问题（顶刊风险）
- 多处 GLM 文档为 2025‑01 时间线，且包含大量“ROI/用户研究/工业案例”定量数据，这些需要和真实实验/日志对齐，否则会被视为不可核验。

### 建议（优先任务）
- 把“结构化解释→自然语言解释”的 **证据链** 落地为可复现输出：至少提供 1 个最小 demo 命令（离线模板 LLM 也可）+ 生成的 JSON/报告样例 + 解释质量评估协议（faithfulness/consistency + hallucination guard）。

---

## Paper 4：`Paper/MOE_explainable`

### 现状（仓库证据）
- MoE 解释性分析结果（路由熵、专家激活等）已有文件输出：`Paper/MOE_explainable/results/moe_analysis_report.txt`。

### 关键问题（可复现与数字口径）
- GLM 文档声称“93.85% / 36K 参数”，但当前仓库的参数统计脚本 `Paper/MOE_explainable/scripts/verify_model_parameters.py` 因 sys.path 处理不当无法运行（导入失败），导致数字缺少可复现证据链。

### 建议（优先任务）
- 修复“参数量统计 + 训练结果真源”两件事：  
  1) 任何参数/准确率必须来自统一脚本输出；  
  2) 统一配置（3/5/8专家、seed列表）与结果表生成脚本锁定。

---

## Paper 5：`Paper/Paper_fuzzy_XFD`

### 现状（仓库证据）
- Paper5 目录有 `manuscript/paper.md`、`manuscript/experiments.md`、`manuscript/references.bib`，并有若干图（membership/rule heatmap 等）。
- 但当前 Paper5 目录下 baseline 结果为 `accuracy=0.47`：`Paper/Paper_fuzzy_XFD/results/fuzzy_baseline_results.json:609`。

### 关键问题（重大矛盾/硬阻塞）
- GLM 文档宣称 P0 验证的“预期 70.7%±0.5%”，并指出数据格式阻塞（HDF5 vs npy）：`Paper/doc/12_14/glm/P0_validation_results_summary.md:7`。  
- 这与 Paper5 目录现有结果（47%）冲突：必须先把数据口径与训练入口统一，才能判定“突破是否真实可复现”。

### 建议（优先任务）
- 先执行统一 P0：**数据口径统一 → multi‑seed 真跑 → 输出 mean±std/CI**。任何“70.7%突破”叙事必须在此之后再写入主稿。

---

## Paper 6：`Paper/Neuralsymbolic_theory`

### 现状（仓库证据）
- 理论与实验脚本、图表、以及 `manuscript/paper.md` 与 `manuscript/references.bib` 已存在：`Paper/Neuralsymbolic_theory/manuscript/references.bib`（文件较长，已具备引用条目雏形）。

### 关键问题（顶刊口径）
- 最大缺口在“真实数据集验证 + 统一基线对比 + explainability协议落地”，否则会被视为理论性强但实证薄弱。
- GLM 文档中存在乱码符号与时间线漂移（影响协作与审稿材料严谨度）：`Paper/doc/12_14/glm/next_steps_12_14.md`。

### 建议（优先任务）
- 用统一协议把 4层理论映射落地到至少 2 个真实数据集（CWRU+XJTU），输出可复现的 Table/Figure（而不仅是概念图）。

---

## 跨项目共性阻塞项（必须统一解决）

1) **结果真源缺失**：需要统一“生成脚本 + 结果表（CSV/JSON）+ 配置快照”的真源体系，杜绝 README/文档里直接写口头数字。  
2) **复现入口不稳定**：硬编码 conda 路径、绝对数据路径、错误的相对路径会导致“跑不起来”。  
3) **数据口径未锁定**：PHM‑Vibench（HDF5/npy）+ CWRU/XJTU 的预处理与划分必须形成唯一规范。  
4) **可解释评估协议要统一**：faithfulness/stability/efficiency 等必须用同一实现与输出格式，且能跨模型/跨论文复用。  

