# Paper 目录说明（中文）

本目录包含基于本仓库方法框架构建的 **各类论文子项目**，每个子目录通常对应一篇或一组论文。

- 典型子项目：  
  - `1D-2D_fusion_explainable`：多模态 1D-2D 融合可解释故障诊断。  
  - `Explainable_FD_Toolkit`：可解释故障诊断工具集（“可解释性 OS”）。  
  - `LLM_Explainable_FD_Toolkit`：基于 LLM 的自然语言解释与对话接口。  
  - `MOE_explainable`、`Paper_fuzzy_XFD`、`Neuralsymbolic_theory`、`TII_operator_attention` 等。  
- `doc/`：Paper 总览与关系说明；  
- `results/`：跨 Paper 的汇总结果。  

原则：  
- Paper 子项目应主要包含方法逻辑、实验配置与论文文稿；  
- 通用代码尽量上收到父项目（`model/`, `data/`, `explainability/` 等）。  

---

## 7篇Paper 总览（官方顺序）

> 顺序与角色边界以 `Paper/doc/README_11_25.md` 为准；以目录路径为唯一ID（避免不同文档出现Paper编号漂移）。

| # | 子项目 | 定位（摘要） | README |
|---:|---|---|---|
| 1 | `Paper/1D-2D_fusion_explainable` | 1D+2D多模态融合可解释方法 | `Paper/1D-2D_fusion_explainable/README.md` |
| 2 | `Paper/Explainable_FD_Toolkit` | 可解释性“操作系统”（统一API/评估/可视化） | `Paper/Explainable_FD_Toolkit/README.md` |
| 3 | `Paper/LLM_Explainable_FD_Toolkit` | 自然语言解释与对话交互（消费结构化解释） | `Paper/LLM_Explainable_FD_Toolkit/README.md` |
| 4 | `Paper/MOE_explainable` | 物理同构MoE专家路由（路径级可解释） | `Paper/MOE_explainable/README.md` |
| 5 | `Paper/Paper_fuzzy_XFD` | 模糊规则可审计（规则级可解释） | `Paper/Paper_fuzzy_XFD/README.md` |
| 6 | `Paper/Neuralsymbolic_theory` | 神经-符号一体化理论（跨层） | `Paper/Neuralsymbolic_theory/README.md` |
| 7 | `Paper/TII_operator_attention` | 算子级注意力（理论主导+合成信号验证） | `Paper/TII_operator_attention/README.md` |

---

## 顶刊统一证据链（必读）

> 目标：顶刊/顶会（用户确认） + PHM-Vibench 多数据集验证（至少 CWRU + XJTU）。

- 总控蓝图：`Paper/doc/12_14/codex/uxfd_master_paper_blueprint.md`
- 7篇TODO汇总：`Paper/doc/12_14/codex/summary_and_todo_7_papers_12_14_codex.md`
- 7篇现状审计：`Paper/doc/12_14/codex/status_audit_7_papers.md`
- 复现入口索引：`Paper/doc/12_14/codex/repro_index.md`
- 可解释评估协议：`Paper/doc/12_14/codex/explainability_eval_protocol.md`
- 结果表模板：`Paper/doc/12_14/codex/results_tables_template.md`


