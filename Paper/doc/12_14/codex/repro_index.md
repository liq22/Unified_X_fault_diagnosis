# 7篇Paper复现入口索引（PHM-Vibench多数据集）

**更新日期**：2025-12-14  
**目标**：为每篇Paper提供“最小可复现入口”（命令/配置/输出位置/验收），并明确多数据集验证口径（PHM-Vibench，含CWRU/XJTU）。  

> 注意：本仓库入口参数存在 `--config_dir` / `--config_file` 两种写法（历史文档混用）。本索引优先使用仓库说明里已有的 `--config_dir`；若某些配置/脚本要求 `--config_file`，会在条目中单独说明。

---

## 通用约定（所有Paper共用）

- 环境：`conda activate UXFD`（以仓库 `environment.yml` 为准）
- 统一多数据集任务：PHM-Vibench（建议使用 `configs/PHM_Vibench/` 下配置）
- 输出要求：每次复现至少保存
  - 配置文件（原样）
  - 运行命令（含 `CUDA_VISIBLE_DEVICES`、seed）
  - 结果目录（日志/metrics/图表/表格）

---

## 0）PHM-Vibench 基础配置入口（推荐）

### 多数据集（CWRU + XJTU + …）
- `python main.py --config_dir configs/PHM_Vibench/config_TSPN.yaml`
- `python main.py --config_dir configs/PHM_Vibench/config_NNSPN.yaml`
- `python main.py --config_dir configs/PHM_Vibench/config_TKAN.yaml`
- `python main.py --config_dir configs/PHM_Vibench/config_FuzzyLogic.yaml`

> 这些配置在 `configs/PHM_Vibench/` 中显式列出 `dataset_ids`，默认包含CWRU与XJTU等。

---

## 1）`Paper/1D-2D_fusion_explainable`（Fusion1D2D）

### 最小复现（统一基线单数据集：THU_018_basic）
- `python main.py --config_dir configs/unified_baseline/config_Fusion1D2D.yaml`

### 多seed稳定性（示例）
- `python main.py --config_dir configs/unified_baseline/config_Fusion1D2D_seed20.yaml`
- `python main.py --config_dir configs/unified_baseline/config_Fusion1D2D_seed42.yaml`
- `python main.py --config_dir configs/unified_baseline/config_Fusion1D2D_seed2024.yaml`

### 多数据集（PHM-Vibench）
- 已有本Paper目录多数据集配置（建议优先跑通CWRU/XJTU）：
  - `python main.py --config_dir Paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml`
  - `python main.py --config_dir Paper/1D-2D_fusion_explainable/configs/config_XJTU.yaml`
  - `python main.py --config_dir Paper/1D-2D_fusion_explainable/configs/config_THU_006.yaml`

- [ ] TODO：如需严格统一到 PHM-Vibench 的 `dataset_ids` 口径，可新增 `configs/PHM_Vibench/config_Fusion1D2D.yaml` 并在README中锁定为唯一入口。

**验收**：
- 产出3-seed统计（mean±std、95%CI）+ 至少CWRU/XJTU的泛化表（若补齐配置）。

---

## 2）`Paper/Explainable_FD_Toolkit`（Explainable FD Toolkit）

### 最小复现（工具包独立benchmark）
- `python Paper/Explainable_FD_Toolkit/scripts/run_benchmark_standalone.py`
- `python Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py`

**验收**：
- 在 `Paper/Explainable_FD_Toolkit/benchmark_results/`（或脚本指定目录）生成 JSON/CSV/Markdown 报告与图表。

---

## 3）`Paper/LLM_Explainable_FD_Toolkit`（LLM自然语言解释）

### 最小复现（示例入口，以子项目脚本为准）
- 推荐最小demo入口（端到端链路的最小版本）：
  - `python Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/run_minimal_llm_demo.py`
- 交互式演示（可选）：
  - `python Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/interactive_llm_demo.py`

### 推荐对齐（结构化解释来源）
- 先跑 `Paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py` 生成结构化解释，再由LLM层消费。

**验收**：
- 产出：结构化解释样例 + 自然语言解释输出 + 评估记录（至少包含响应延迟与失败率）。

---

## 4）`Paper/MOE_explainable`（MoE）

### 最小复现（统一基线）
- `python main.py --config_dir configs/unified_baseline/config_MoE.yaml`

### seed20复现/专家消融（统一基线已有配置）
- `python main.py --config_dir configs/unified_baseline/config_MoE_seed20_reproduce.yaml`
- `python main.py --config_dir configs/unified_baseline/config_MoE_5experts.yaml`
- `python main.py --config_dir configs/unified_baseline/config_MoE_8experts.yaml`

**验收**：
- 输出 3/5/8 专家消融表 + 多seed稳定性统计（CV下降的证据链）。

---

## 5）`Paper/Paper_fuzzy_XFD`（FuzzyLogic）

### 最小复现（统一基线）
- `python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml`

### 多数据集（PHM-Vibench）
- `python main.py --config_dir configs/PHM_Vibench/config_FuzzyLogic.yaml`

**验收**：
- 输出：70.7%结果可复现（含seed/配置/日志）+ 跨数据集（至少CWRU/XJTU）结果表 + 2–3个安全关键失败案例解释。

---

## 6）`Paper/Neuralsymbolic_theory`（NeSy理论）

### 最小复现（理论验证demo）
- `python Paper/Neuralsymbolic_theory/run_validation_demo.py`
- `python Paper/Neuralsymbolic_theory/simple_validation_demo.py`

**验收**：
- 产出：命题验证结果（尤其命题2的可复现实验脚本与图表）+ 映射验证报告（对应7篇方法）。

---

## 7）`Paper/TII_operator_attention`（Operator Attention）

### 最小复现（合成信号验证：理论主证据链）
- `python Paper/TII_operator_attention/code/synthetic_verification.py --verbose`

### 工业数据概念验证（统一基线配置）
- `python main.py --config_dir configs/unified_baseline/config_OperatorAttention_optimized.yaml`

**验收**：
- 合成信号：8类信号权重热图 + 物理一致性评分报告（目标>0.9，阈值可在论文中解释）  
- 工业数据：可运行性与可解释性分析（性能不作为主卖点）
