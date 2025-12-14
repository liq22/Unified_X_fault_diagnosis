# UXFD 可解释性评估协议（PHM-Vibench多数据集）

**更新日期**：2025-12-14  
**目的**：将“可解释性”从“可视化展示”升级为**可量化、可复现、可验收**的评估协议，适配7篇Paper共用。  
**数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU；可扩展 FEMTO/THU/MFPT/UNSW）。  

---

## 1. 统一输入输出约定

### 1.1 输入
- 数据样本：`x`（原始信号窗口，或模型接受的输入形式）
- 标签：`y`
- 模型：`f(·)` 输出类别概率或logits
- 解释：`E(x)`（可为时序归因、特征归因、算子权重、规则激活、专家路由权重等）

### 1.2 输出（每次评估必须保存）
- `metrics.json`：包含本协议所有指标的数值（逐样本 + 汇总）
- `figures/`：至少包含主图（稳定性/忠实度对比、案例可视化）
- `run_meta.yaml`：seed、数据集ID列表、模型版本、解释方法版本、代码commit（可选）

---

## 2. 指标集合（最小必做 + 可选增强）

> 至少满足“1个忠实度指标 + 1个稳定性指标 + 1个效率指标”。顶刊建议补齐一致性/人类一致性。

### 2.1 Faithfulness（忠实度，必做）

#### (A) Deletion / Occlusion Test（删除/遮挡）
对解释给出的 Top-k 重要位置（或Top-k特征/算子/规则）进行遮挡，观察预测置信度下降：
$$
\mathrm{Del}(x)=f_y(x)-f_y(\mathrm{mask}(x, M_{topk}))
$$

- `M_{topk}`：由解释 `E(x)` 排序得到的Top-k mask
- 输出：`Del@k`（k可取 1%、5%、10% 或固定数量）

**验收标准（最低）**：`Del@k` 显著大于随机mask（同k），并报告均值±标准差。  

#### (B) Insertion Test（可选）
从基线输入逐步插入Top-k重要部分，预测应更快恢复：
$$
\mathrm{Ins}(x)=f_y(\mathrm{insert}(x_0, x, M_{topk}))
$$

---

### 2.2 Stability（稳定性，必做）

对输入添加小扰动（噪声/小平移/轻微尺度变化），解释应保持一致：
$$
\mathrm{Stab}(x)=\mathbb{E}_{\delta \sim \mathcal{N}(0,\sigma^2)}\left[\mathrm{sim}(E(x),E(x+\delta))\right]
$$

- `sim(·)`：Spearman 相关/余弦相似度/JS距离的1-归一化
- 输出：`Stab@σ`（σ可固定为若干档）

**验收标准（最低）**：给出至少一档 σ 的稳定性均值±标准差；并与随机解释或基线解释对照。  

---

### 2.3 Consistency（跨样本一致性，可选但强烈建议）

对“相似输入”（同类别/同工况/同域）的解释一致性：
$$
\mathrm{Cons}=\mathbb{E}_{(x_i,x_j)\in \mathcal{P}}\left[\mathrm{sim}(E(x_i),E(x_j))\right]
$$

**验收**：同类一致性 > 异类一致性（报告差值与显著性）。  

---

### 2.4 Sparsity / Compactness（稀疏性，可选）

用于衡量解释是否“可审计/可阅读”：
$$
\mathrm{Sparsity}(E)=\\frac{\\lVert E \\rVert_0}{\\lVert E \\rVert_1 + \\epsilon}
$$

不同解释类型的替代定义：
- 算子注意力：激活算子数（α_i > τ）
- MoE：top-1/熵（路由熵越低越专一）
- Fuzzy：激活规则数、规则覆盖率

---

### 2.5 Efficiency（效率，必做）

记录解释生成耗时与资源开销：
$$
\mathrm{Time}=t(E(x))
$$

输出：
- `ms/sample`（均值、P95）
- GPU显存峰值（若可）

**验收**：至少给出 `ms/sample` 与 P95。  

---

### 2.6 Human Agreement（人类一致性，可选但顶刊强烈建议）

对解释的可理解性/可用性做小规模专家评估（最小10人或10任务）：
- 评分维度：理解度、可信度、行动建议可用性
- 输出：均值±标准差，及任务完成时间/错误率

**验收**：明确研究设计（对照组：无解释/仅可视化/文本解释），并可复现问卷与统计方法。  

---

## 3. 多数据集评估矩阵（PHM-Vibench）

### 3.1 训练/测试协议（建议）
- In-domain：每个数据集内部划分训练/测试
- Cross-domain：留一数据集（LODO）或 CWRU→XJTU 等迁移验证

### 3.2 最小必做组合
- 至少两数据集：CWRU + XJTU
- 至少两模型：一个高性能（Paper1/TSPN）+ 一个高透明（Paper5/7）
- 至少两解释类型：intrinsic + post-hoc（若适用）

---

## 4. 输出文件规范（建议）

```
results/
  <paper_id>/
    <dataset_id>/
      <model>/
        <explainer>/
          run_meta.yaml
          metrics.json
          tables/
            main_results.csv
            explain_metrics.csv
          figures/
            faithfulness_del.png
            stability.png
            case_study_01.png
```

---

## 5. 与7篇Paper的绑定关系（谁必须用哪些指标）

- Paper1（Fusion）：必做 Faithfulness + Stability + 跨模态一致性（Cons），并输出多数据集泛化表  
- Paper2（Toolkit）：必须覆盖全部指标定义与统一输出格式（作为协议载体）  
- Paper3（LLM）：必须补 Human Agreement（或至少用户研究指标）+ Efficiency（成本/延迟）  
- Paper4（MoE）：必须补 Stability（跨seed/跨域）+ Sparsity（路由熵/路径专一度）  
- Paper5（Fuzzy）：必须补 Sparsity（激活规则数）+ Safety failure cases（解释支持）  
- Paper6（NeSy）：必须补 Consistency（跨层一致性）+ 命题验证指标  
- Paper7（OperatorAttention）：必须补 Faithfulness（合成信号对照）+ 物理一致性评分（OSS/OCS等）

