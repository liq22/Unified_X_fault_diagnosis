# Paper README 模板（顶刊口径 / 可验收 / 可复现 / 可审计）

> 适用范围：`Paper/*/README.md`（7篇Paper统一模板），按 `Paper/doc/README_11_25.md` 的6段规范，同时新增“现状快照/复现入口/证据链/可解释评估协议绑定”。  

---

## 0. 现状快照（Current Status）

- **最后更新**：YYYY-MM-DD  
- **完成度**：__%（主观可接受，但必须列出证据）  
- **目标档位**：顶刊/顶会（可列3个候选）  
- **数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU）  
- **统一证据链**：
  - 结果表模板：`Paper/doc/12_14/codex/results_tables_template.md`
  - 可解释评估协议：`Paper/doc/12_14/codex/explainability_eval_protocol.md`
  - 复现入口索引：`Paper/doc/12_14/codex/repro_index.md`

---

## 1. 要解决的问题（Problem）

- 痛点：______  
- 现有方法不足：______  
- 本Paper解决的核心缺口：______  

---

## 2. 研究内容（Research Content）

- 任务：分类/定位/RUL/跨域等  
- 数据：PHM-Vibench（CWRU/XJTU/…）  
- 假设：______（如果有）  

---

## 3. 技术路线（Technical Route）

- 方法总览：______（建议给一张图/mermaid）  
- 与主仓库复用关系：使用哪些 `model/`、`trainer/`、`configs/`  
- 与其他Paper解耦边界：明确“不做什么”  

---

## 4. 预期论文结果（Expected Results in Paper）

- 主结果：Table 2（性能）  
- 可解释评估：Table 4（faithfulness/stability/…）  
- 多数据集泛化：Table 5（LODO/transfer）  
- 关键图：Figure 1/2/3/…（列清单）  

---

## 5. 讨论（Discussion）

- Why it works：______  
- Failure cases：______  
- Limitations：______  

---

## 6. TODO（Roadmap / Checklist）

> 必须使用勾选清单；每个条目必须“可交付 + 可验收”。

### P0（本周）
- [ ] 任务A（交付物：__；验收：__）
- [ ] 任务B（交付物：__；验收：__）

### P1（两周）
- [ ] 任务C（交付物：__；验收：__）

---

## 7. 复现入口（Reproducibility）

### 7.1 最小复现命令
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir <your_config.yaml>
```

### 7.2 输出位置
- `results/`：______  
- `figures/`：______  

### 7.3 验收（Acceptance）
- [ ] 结果可复现：命令+配置+输出三件套齐全  
- [ ] 多seed统计：mean±std/95%CI  
- [ ] 解释评估：至少1个faithfulness+1个stability+1个efficiency  

