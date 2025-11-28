# 阶段3基线与集成实验调优计划（2025-11-27）

> 依据：`docs/11_26/results/experiment_summary_11_27.md`、`TSPN_detailed_results.md`、`unified_baseline_comparison.md`  
> 目标：先稳定 TSPN 与统一基线，再系统完成 7 个新方法 + 经典 baseline 的对比与组合实验。

---

## 一、本轮实验评审与主要问题

### 1.1 TSPN 基线评审

- 成功率：2/5（40%），成功时测试准确率 ≈ 99.93%，性能极佳。  
- 失败特征：  
  - 验证损失在 30+ epoch 后突然变为 NaN；  
  - 训练准确率从 100% 掉到 60% 左右；  
  - L1 损失同步变为 NaN。  
- 已定位原因（文档中）：L1 正则系数过大（`l1_norm=0.001`）+ 可能存在梯度爆炸/归一化数值不稳。

### 1.2 统一基线表进度

- 已有：TSPN 行内容较完整（含参数量、训练时间估计、不稳定性说明）。  
- 待填：Fusion1D-2D、MoE、OperatorAttention、FuzzyLogic、ResNet/SincNet/WKN/MCN/TFN 等仍为 “运行中/待运行”。  
- 当前瓶颈：  
  - TSPN 训练不稳定 → 无法放心用作“黄金基线”；  
  - 多个实验并行中，不宜一次性扩展到 12×5 全矩阵，需要明确优先顺序。

### 1.3 风险点总结

1. **数值稳定性风险**：TSPN 的 L1/归一化问题若不先解决，后续对比会混入“训练崩溃”的 noise。  
2. **资源瓶颈风险**：所有模型全量 5 次运行（≥60 runs）对单 GPU 压力较大，需要合理调度与早停策略。  
3. **结果一致性风险**：如果不同模型使用不同配置/指标，汇总表难以保证可比性。

---

## 二、调优与实验执行优先级（短期：接下来 3–5 天）

### 2.1 Step 1：先稳定 TSPN（高优先级）

**目的**：TSPN 是统一基线的核心，必须先解决其 NaN 问题，再扩展其他模型。

建议配置修改（在对应 `config_TSPN.yaml` 派生一个新配置，如 `config_TSPN_stable.yaml`）：  
- 降低 L1 正则：  
  ```yaml
  l1_norm: 0.0001  # 从 1e-3 降到 1e-4
  ```  
- 添加梯度裁剪：  
  ```yaml
  gradient_clip_val: 1.0
  gradient_clip_algorithm: "norm"
  ```  
- 若仍不稳定，可尝试：  
  - 将学习率从 `1e-3` 降到 `5e-4`；  
  - 检查归一化层（InstanceNorm/BatchNorm）设置，适当增大 epsilon。

**执行计划**：  
- [ ] 使用新配置重新运行 TSPN 3–5 次（不同种子）；  
- [ ] 若 5/5 次无 NaN，更新 `TSPN_detailed_results.md` 与 `unified_baseline_comparison.md`；  
- [ ] 若仍有 NaN，追加记录失败 log，并在 docs/11_27 下补充 “TSPN_Stability_Analysis_11_27.md” 进一步分析。

### 2.2 Step 2：经典基线模型优先跑完（中高优先级）

**目的**：在 TSPN 稳定后，尽快补齐 `ResNet / SincNet / WKN / MCN / TFN` 行，为新方法对比提供坚实基线。

建议顺序：  
1. ResNet（代表标准深度模型）  
2. SincNet（代表专用滤波器 CNN）  
3. WKN / MCN / TFN（信号处理型与变换型模型）

**执行任务**：  
- [ ] 确定所有 baseline 都使用 VBench 数据接口 + 统一 trainer；  
- [ ] 为每个 baseline 跑至少 3 次（如资源不足可先 3 次，后续补到 5 次）；  
- [ ] 将结果填入 `unified_baseline_comparison.md` 的 “基线方法对比” 表。

### 2.3 Step 3：7 个新方法的第一轮主实验（中优先级）

在 TSPN + baseline 稳定后，按以下顺序展开：  
1. Fusion1D-2D（📘）：多模态核心方法；  
2. MoE（🟠）：专家路由方法；  
3. OperatorAttention（🔴）：算子注意力方法；  
4. FuzzyLogic（🩷）：规则级解释方法。

**执行任务**：  
- [ ] 每个方法至少在 THU_018_basic 上完成 3 次主实验；  
- [ ] 将性能、训练稳定性、参数量等信息填入统一表格；  
- [ ] 对于 LLM Interface / NeSy，只记录其依赖实验（不参与数值对比）。

---

## 三、调试与监控建议（避免重复踩坑）

### 3.1 NaN/爆炸检测

- 在训练 loop 中增加简单的数值检查：  
  - 每 N step 检查 loss / 梯度是否包含 NaN/Inf；  
  - 一旦发现，记录当前 batch/配置，提前终止 run 并保存日志。  
- 对于使用 L1/L2 大正则的配置，尤需注意梯度规模。

### 3.2 早停与资源策略

- 为所有模型启用 EarlyStopping（基于验证损失/准确率），避免显然失败的 run 浪费资源。  
- 对同类实验采用 **分批执行策略**：先 3 次快速 run 检查稳定性，再决定是否扩展到 5 次。

### 3.3 结果记录规范

- 对每个模型建立独立的 `*_detailed_results.md`（TSPN 已有，可仿照）：  
  - 按种子/配置记录每次 run 的关键统计；  
  - 在 unified 表里仅放聚合结果（均值±方差）。  
- 确保实验结果文件中记录使用的配置文件名，便于追溯。

---

## 四、下一个阶段性目标（阶段3中段）：从“单模型对比”走向“组合实验”

在 TSPN + baseline + 单方法主实验完成后（预期接下来 1–2 周内），可以进入组合阶段：

### 4.1 组合实验优先级

1. TSPN + MoE：路径级物理专家路由对原始 TSPN 的增益；  
2. TSPN + OperatorAttention：算子级注意力对透明算子的影响；  
3. Fusion1D-2D + MoE/OpAtt：多模态 + 专家/算子的协同效果；  
4. Fuzzy-XFD 与深度模型：规则层对决策的修正与解释增强。

### 4.2 LLM 与 Toolkit 的联动实验

- 当 Explainable_FD_Toolkit 的统一接口与多模型解释结果稳定后：  
  - 由 LLM Interface 用统一解释结果生成自然语言解释；  
  - 对比传统解释（图 + 文本说明）与 LLM 解释的理解度/效率。

---

## 五、对后续 Codex/Agent 的使用建议（本轮之后）

1. 若任务是“立刻恢复稳定训练” → 优先按 **2.1 Step 1** 调整 TSPN 配置并重跑少量实验。  
2. 若任务是“补齐统一基线表” → 完成 **2.2 Step 2** 中的 ResNet/SincNet/WKN/MCN/TFN 实验。  
3. 若任务是“推进新方法主结果” → 在基线稳固后按 **2.3 Step 3** 顺序跑 1D-2D / MoE / OpAtt / Fuzzy 的主实验。  
4. 若任务是“做跨方法组合与 NeSy/LLM 集成” → 等上述工作基本完成后，再参考 `docs/11_26/codex/plan_stage3_integration_11_26.md` 的组合实验计划。  

---

## 六、文档与配置一致性修复子计划（针对 11_27 GLM 文档）

> 依据 Claude 反馈：`docs/11_27/glm/实验执行清单.md` 与 `统一基线实验完成计划.md` 中存在配置名与当前仓库状态不一致的问题。

### 6.1 修复目标概览

1. **配置文件名统一**  
   - 消除文档中对不存在文件 `config_TSPN_fixed.yaml` 的引用；  
   - 统一 ResNet 配置名为实际存在的 `config_Resnet.yaml`（小写 n）。  
2. **TSPN 状态同步**  
   - 文档中将 TSPN 配置从“需要修复”更新为“已降低 l1_norm 至 1e-4，待添加/验证梯度裁剪”；  
   - 与当前 `configs/unified_baseline/config_TSPN.yaml` 状态保持一致。  
3. **执行优先级与时间估算对齐 codex**  
   - GLM 文档中的执行顺序改为：先稳定 TSPN → 经典基线 → 新方法；  
   - 时间估算增加缓冲（可在 GLM 文档中标注“实际执行按 1.5× 预估预留时间”）。

### 6.2 具体修改步骤

#### Step A：更新 GLM 实验执行清单

文件：`docs/11_27/glm/实验执行清单.md`

- [ ] 将所有 `config_TSPN_fixed.yaml` 文本替换为 `config_TSPN.yaml` 或 `config_TSPN_stable.yaml`（二选一，推荐新建稳定版本并显式命名）。  
- [ ] 将 `config_ResNet.yaml` 全部改为 `config_Resnet.yaml`（与 `configs/unified_baseline/` 实际文件匹配）。  
- [ ] 在“配置文件状态”小节中，将 `config_TSPN.yaml` 标注从“需要修复”更新为“已修复 l1_norm=0.0001，待验证梯度裁剪”；  
- [ ] 在“立即行动项”中，将“修复 TSPN 配置”调整为“为 TSPN 添加梯度裁剪配置，并用新配置重跑 3 次验证实验”；  
- [ ] 在 TODO 区块中，将执行顺序描述调整为：“先 TSPN 稳定 → 经典基线 → 新方法”，与本 codex 计划一致。

#### Step B：更新 GLM 统一基线实验完成计划

文件：`docs/11_27/glm/统一基线实验完成计划.md`

- [ ] 将所有 `config_ResNet` 或“ResNet”配置文件名统一为 `config_Resnet`（只改文件名部分，不改模型名本身）。  
- [ ] 在“关键问题识别”与“第一阶段：稳定化与验证”中，说明 TSPN 当前已降低 `l1_norm`，本阶段重点是添加梯度裁剪并确认稳定性；  
- [ ] 在执行计划中显式写出与 codex 一致的三步顺序：  
  1. 调整/验证 TSPN 配置并确保稳定；  
  2. 补齐经典基线五个模型；  
  3. 补齐新方法与组合实验。  
- [ ] 在时间线表格处，将每阶段预估时间注明为“基础估算 ×1.5”或添加一句说明：“实际执行中建议预留约 1.5 倍时间缓冲”。

#### Step C：同步配置文件（按需）

文件：`configs/unified_baseline/config_TSPN.yaml`

- [ ] 若尚未添加梯度裁剪配置，可按建议加入：  
  ```yaml
  gradient_clip_val: 1.0
  gradient_clip_algorithm: "norm"
  ```  
- [ ] 若决定使用单一配置文件而非 `*_stable` 版本，应在 GLM 文档与本 codex 中统一说明使用 `config_TSPN.yaml` 作为“稳定版”。

### 6.3 验证清单

完成上述修改后，检查以下点：

- [ ] 仓库中不再出现 `config_TSPN_fixed.yaml` 的引用；  
- [ ] 所有 GLM 文档中的 ResNet 配置名均为 `config_Resnet.yaml`；  
- [ ] TSPN 在文档中的状态描述为“已降低 l1_norm 并添加/计划添加梯度裁剪”；  
- [ ] GLM 文档中的执行顺序与本 codex 计划（先 TSPN → 再基线 → 再新方法）一致。

