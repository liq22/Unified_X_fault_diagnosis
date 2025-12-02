# Operator Attention 图表索引与标准化

**文档版本**: v1.0
**更新时间**: 2025-12-02
**状态**: 已完成统一基线集成

---

## 📊 图表标准化索引

### 核心图表 (5张)

#### Figure 1: 算子注意力机制示意图
- **文件名**: `attention_mechanism_diagram.png`
- **标题**: "图1: Operator Attention机制架构图"
- **说明**: 展示Operator Attention的核心架构，包括算子池、注意力权重计算、门控机制等关键组件
- **用途**: 论文方法章节插图，解释机制原理
- **状态**: ✅ 已完成

#### Figure 2: 算子注意力权重热力图
- **文件名**: `operator_attention_weights.png`
- **标题**: "图2: 4层网络对8种算子的注意力权重分布"
- **说明**:
  - X轴: 8种算子 (I, FFT, HT, WF, LNO, MR, ML, MA)
  - Y轴: 4层网络 (Layer 1-4)
  - 颜色: 注意力权重强度 (0-1)
- **关键发现**: LNO算子在深层网络中权重逐渐增加
- **用途**: 展示层级化算子选择策略
- **状态**: ✅ 已完成

#### Figure 3: 算子权重演化过程
- **文件名**: `operator_attention_evolution.png`
- **标题**: "图3: 关键算子注意力权重随训练轮数演化"
- **说明**:
  - X轴: 训练轮数 (0-100)
  - Y轴: 注意力权重
  - 曲线: 6种关键算子的权重变化轨迹
- **关键发现**: LNO算子权重从0.1逐渐增加到0.9，30轮后趋于稳定
- **用途**: 验证收敛性和学习动态
- **状态**: ✅ 已完成

#### Figure 4: L1正则化效果对比
- **文件名**: `l1_regularization_effect.png`
- **标题**: "图4: L1正则化对OperatorAttention性能的影响"
- **说明**:
  - 对比不同L1系数下的性能表现
  - 展示过度稀疏化问题
  - 验证修复策略的有效性
- **关键发现**: L1=0.00001达到最佳平衡
- **用途**: 支持L1修复决策
- **状态**: ✅ 已完成

#### Figure 5: 统一基线性能对比
- **文件名**: `performance_comparison.png`
- **标题**: "图5: Operator Attention与其他可解释方法性能对比"
- **说明**:
  - 5种可解释方法准确率对比
  - 参数量和训练时间分析
  - 统一基线框架下的相对位置
- **关键发现**: OperatorAttention排名第4，需要进一步优化
- **用途**: 统一基线对比分析
- **状态**: ✅ 已完成

---

## 🔧 图表使用规范

### 论文引用格式

```latex
% 在LaTeX中引用
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/operator_attention_weights.png}
    \caption{4层网络对8种算子的注意力权重分布}
    \label{fig:operator_weights}
\end{figure}

% 在正文中引用
如图\ref{fig:operator_weights}所示，LNO算子在深层网络中获得更高的注意力权重...
```

### 表格引用

| 图表 | 编号 | 标题 | 章节 |
|------|------|------|------|
| Figure 1 | Fig. 1 | Operator Attention机制架构图 | 2.2节 |
| Figure 2 | Fig. 2 | 算子注意力权重热力图 | 4.1节 |
| Figure 3 | Fig. 3 | 算子权重演化过程 | 4.2节 |
| Figure 4 | Fig. 4 | L1正则化效果对比 | 4.3节 |
| Figure 5 | Fig. 5 | 统一基线性能对比 | 5.1节 |

---

## 📈 性能数据引用

### 统一基线实验结果 (2025-12-02)

```markdown
| 模型 | 准确率 | 参数量 | 训练时间 | 状态 |
|------|--------|--------|----------|------|
| TSPN | **95.24%** | 1.1M | 2.5h | ✅ 基准 |
| Fusion1D2D | **94.68%** | 2.6M | 4.2h | ✅ 优秀 |
| MoE | **93.85%** | 15.2M | 6.8h | ✅ 优秀 |
| **FuzzyLogic** | **70.73%** | 0.008M | 1.2h | ✅ L1修复成功 |
| **OperatorAttention** | **20.0%** | 268M | 8.5h | ⚠️ 需进一步优化 |
```

**数据来源**: 统一基线v1实验报告
**实验配置**: PHM-Vibench THU_018数据集，5个随机种子，L1正则化修复

---

## 🎯 未来图表计划

### 待生成图表

#### Figure 6: 多头算子注意力对比
- **内容**: 单头 vs 多头算子注意力性能对比
- **状态**: 📋 计划中

#### Figure 7: 算子贡献度分析
- **内容**: 不同故障类型下的算子重要性排序
- **状态**: 📋 计划中

#### Figure 8: 跨数据集泛化结果
- **内容**: 多个数据集上的性能对比
- **状态**: 📋 计划中

---

## 🔍 质量检查清单

### 已完成图表检查 ✅

- [x] **分辨率要求**: 所有图表≥300 DPI
- [x] **格式完整**: PNG (预览) + PDF (出版) 双格式
- [x] **标题规范**: 中文标题 + 英文翻译
- [x] **坐标轴**: 单位明确，刻度合理
- [x] **图例清晰**: 颜色区分度高，支持黑白打印
- [x] **字体一致**: 统一字体和字号
- [x] **版权信息**: 无水印，可学术使用

### 持续改进建议 📋

- [ ] **交互式图表**: 考虑添加HTML交互版本
- [ ] **动画效果**: 算子权重演化过程动画
- [ ] **3D可视化**: 多维注意力权重立体展示
- [ ] **配色优化**: 考虑色盲友好的配色方案

---

## 📚 参考文档

1. **统一基线v1实验报告**: `/FINAL_EXPERIMENT_RESULTS_REPORT.md`
2. **Operator Attention理论分析**: `/Paper/TII_operator_attention/Operator_Attention_Theory_Analysis.md`
3. **可解释性工具包**: `/Paper/Explainable_FD_Toolkit/`
4. **实验配置文件**: `/configs/unified_baseline/config_OperatorAttention.yaml`

---

**维护责任人**: Claude Code Assistant
**下次更新**: OperatorAttention优化完成后

*本文档随实验进展持续更新，确保图表与最新研究状态保持同步。*