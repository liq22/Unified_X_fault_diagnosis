# Paper 1 当前TODO清单

> **更新时间**：2024-12-14
> **优先级**：P0 > P1 > P2
> **状态**：Phase 0 已完成，进入 Phase 1

## 🚨 P0任务（24-72小时，最高优先级）

### 已完成 ✅
- [x] 论文初稿框架搭建（Abstract, Introduction, Related Work）
- [x] 实验复现清单（experiments.md）
- [x] 参考文献整理（references.bib，69篇）

### 进行中 🔄
- [ ] 3-seed稳定性测试执行
  - [ ] 创建seed=20/42/2024配置文件
  - [ ] 启动THU_018实验（3个GPU并行）
  - [ ] 监控实验进度
  - [ ] 收集结果并统计分析
  - [ ] 生成three_seed_report.pdf

### 待办 ☐
- [ ] 投稿材料预准备
  - [ ] 确认目标期刊格式要求
  - [ ] 检查图表分辨率（300/600 dpi）
  - [ ] 整理补充材料清单

---

## 📋 P1任务（1-2周）

### 实验相关
- [ ] **多数据集泛化验证**
  - [ ] CWRU数据集实验
    - [ ] 创建配置文件 `config_CWRU.yaml`
    - [ ] 执行3-seed测试
    - [ ] 性能对比分析
  - [ ] XJTU数据集实验
    - [ ] 创建配置文件 `config_XJTU.yaml`
    - [ ] 执行3-seed测试
  - [ ] THU_006数据集实验
    - [ ] 创建配置文件 `config_THU_006.yaml`
    - [ ] 执行基础验证
  - [ ] 生成跨数据集性能对比表

- [ ] **可解释性评估实验**
  - [ ] Faithfulness（Deletion/Occlusion Test）
    - [ ] 实现Del@k评估
    - [ ] 与随机mask对比
  - [ ] Stability（扰动一致性）
    - [ ] 高斯噪声测试
    - [ ] 计算Spearman相关
  - [ ] Efficiency（解释耗时）
    - [ ] 时间测量
    - [ ] GPU内存使用记录

### 论文撰写
- [ ] **Methodology章节**
  - [ ] 3.1 Problem Formulation（300字）
    - [ ] 数学符号定义
    - [ ] 优化目标函数
  - [ ] 3.2 Tri-Level Alignment Framework（800字）
    - [ ] 物理对齐约束公式
    - [ ] 语义对齐损失函数
    - [ ] 几何对齐实现
  - [ ] 3.3 Progressive Fusion Network（600字）
    - [ ] 网络架构图
    - [ ] 融合机制说明
  - [ ] 3.4 Explainability Protocol（300字）
    - [ ] 评估指标定义

### 图表准备
- [ ] **Figure 1**: 整体框架与三层对齐示意
  - [ ] 架构图设计
  - [ ] 三层对齐可视化
  - [ ] 输出为PDF（双栏，600dpi）

- [ ] **Figure 2**: 解释机制图
  - [ ] 跨模态贡献分析
  - [ ] 对齐一致性展示
  - [ ] 输出为PDF（双栏，600dpi）

---

## 📝 P2任务（1个月）

### 论文完善
- [ ] **Section 4: Experiments**
  - [ ] 4.1 Datasets详细描述
  - [ ] 4.2 Baseline方法对比
  - [ ] 4.3 Evaluation Metrics说明

- [ ] **Section 5: Results**
  - [ ] 5.1 Performance Comparison
    - [ ] 主结果表（含CI）
    - [ ] 统计显著性检验
  - [ ] 5.2 Explainability Analysis
    - [ ] 三层对齐效果展示
    - [ ] 跨模态贡献热力图
  - [ ] 5.3 Ablation Study
    - [ ] 融合策略对比
    - [ ] 对齐模块消融
  - [ ] 5.4 Case Studies
    - [ ] 成功案例分析（2个）
    - [ ] 失败案例分析（1个）

- [ ] **Section 6: Discussion**
  - [ ] 主要发现总结
  - [ ] 机制解释
  - [ ] 与SOTA对比
  - [ ] 局限性分析
  - [ ] 未来工作方向

- [ ] **Section 7: Conclusion**
  - [ ] 工作总结
  - [ ] 贡献重申
  - [ ] 实际应用价值

### 消融实验深化
- [ ] 融合策略对比：early vs mid vs late
- [ ] 对齐模块贡献：w/ vs w/o alignment
- [ ] 统计特征重要性分析
- [ ] 超参数敏感性研究

### 投稿准备
- [ ] LaTeX格式转换
- [ ] 图表格式检查
- [ ] 参考文献格式化
- [ ] Cover Letter撰写
- [ ] 补充材料准备

---

## 📊 进度跟踪

### 本周重点
1. **优先1**: 3-seed稳定性测试（P0）
2. **优先2**: Methodology章节撰写（P1）
3. **优先3**: CWRU数据集验证（P1）

### 里程碑
- **Day 1-3**: 完成3-seed测试
- **Day 4-7**: 完成Methodology
- **Week 2**: 完成多数据集验证
- **Week 3**: 完成Results和Discussion
- **Week 4**: 完成Conclusion和投稿准备

---

## 🔧 工具与脚本

### 已准备
- [x] 实验监控脚本：`scripts/monitor_three_seed.py`
- [x] 结果分析脚本：`scripts/analyze_three_seed_results.py`
- [x] 配置验证脚本：`scripts/validate_configs.py`

### 待开发
- [ ] 多数据集批量运行脚本
- [ ] 图表自动生成脚本
- [ ] LaTeX格式检查脚本

---

## 📞 需要支持

1. **实验资源**
   - 确保至少3块GPU可用于并行实验
   - 验证数据集路径可访问

2. **文献支持**
   - TSPN论文完整引用信息（当前标记为TO-VERIFY）
   - THU_018/PHM-Vibench官方引用

3. **格式要求**
   - 确认目标投稿期刊（IEEE Transactions?）

---

## ✅ 完成标准

### P0完成标准
- [ ] 3-seed实验全部完成
- [ ] 统计报告生成
- [ ] 论文Methodology章节初稿

### P1完成标准
- [ ] 多数据集验证完成
- [ ] 所有实验图表生成
- [ ] Results章节完成

### P2完成标准
- [ ] 论文全文完成
- [ ] 格式符合投稿要求
- [ ] 投稿材料准备就绪