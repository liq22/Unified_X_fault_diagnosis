# 🎯 统一故障诊断项目下一步执行计划

**时间**: 2025年11月28日
**基于**: GLM视角的实验状态分析 + Codex规范的技术路线
**目标**: 系统化推进统一基线实验与Paper开发

---

## 📋 执行顺序与具体任务

### Phase 1: 实验稳定与问题解决（1–2 天）

#### 任务1: 监控并收集当前实验数据（立即执行）
- **MoE 和 OperatorAttention 训练**：继续训练至配置中设定的 epoch（如 100），视收敛情况决定是否延长  
- **实时监控**：使用`experiment_monitor_dashboard.py`跟踪进展
- **数据收集**：
  - 完整训练曲线（loss, accuracy, L1 regularization）
  - 模型检查点备份（`save/`目录）
  - WandB训练日志（`wandb/`目录）
  - 性能指标时间戳记录

#### 任务2: 修复 Fusion1D2D 的 shape 问题（并行进行）
- **临时方案**：
  - 立即停止当前失败实验
  - 如需重新启动实验，继续使用当前统一基线配置 `configs/unified_baseline/config_Fusion1D2D.yaml`（`main.py` 中已使用简化版 Fusion1D2D）
  - 使用已验证的当前版本收集初步数据
- **完整方案**：
  - 创建专门的shape诊断脚本
  - 分析`shape '[64, 3, -1]' is invalid for input of size 524288`错误
  - 定位SignalProcessingLayer到CNN分支的tensor flow问题
  - 确保与VBenchDataset的shape兼容性

#### 任务3: 调整 OperatorAttention 的 L1 正则化系数
- **问题诊断**：当前 L1 值过高（40k–150k），可能影响收敛速度
- **调整策略**：
  - 将当前 L1 系数降低 10–100 倍（例如从 `1e-3` 调整到 `1e-4` 或 `1e-5`）  
  - 观察收敛速度和稳定性变化，记录调整前后的性能对比曲线  
- **配置更新**：在 `configs/unified_baseline/config_OperatorAttention.yaml` 中更新对应正则化参数（如 `l1_norm`），并在注释中记录修改原因  

### Phase 2: 统一基线文档建设（2–3 天）

> 原则：以“结果快照 + 明确标注”为主，所有数值在 Codex 侧视为当前运行状态，后续 baseline 版本需单独确认。

#### 任务1: 创建统一基线结果表框架
- **文件创建**：`Paper/doc/11_28/codex/unified_baseline_results_codex_11_28.md`
- **表头设计**：
  ```
  | 模型 | 数据集 | 准确率 | F1分数 | 参数量 | 训练时间 | 稳定性备注 | 状态 |
  |------|--------|--------|--------|--------|----------|------------|------|
  ```
- **指标标准化**：
  - 准确率：验证集最佳精度
  - F1分数：多类别平均
  - 参数量：可训练参数总数
  - 稳定性：多次运行的方差分析
  - 状态：快照/需复现/已验证

#### 任务2: 写入三个核心 Paper 的实验快照（标注为“快照”）
- **1D-2D Fusion**：
  - 当前训练结果（含shape问题分析）
  - 99.57%历史结果的复现验证
  - 模态贡献分析数据
- **MoE**：
  - 63%基线结果和专家激活分析
  - 物理约束专家系统性能
  - 路径稀疏性统计
- **OperatorAttention**：
  - 当前20%精度基线
  - 算子权重分布分析
  - 计算复杂度对比（vs TSPN）

#### 任务3: 建立 7 个 Paper 的基线引用段落
- **统一引用格式**：
  ```markdown
  统一基线结果见 unified_baseline_results_codex_11_28.md 中的第 X 行，
  本章节只讨论本方法相对基线的差异和优势。
  ```
- **具体实施**：
  - 1D-2D Fusion Paper：结果章节添加基线对比
  - MoE Paper：专家系统章节引用基线
  - OperatorAttention Paper：算子分析章节对比
  - FuzzyLogic Paper：规则推理章节对比
  - Toolkit Paper：解释工具章节对比
  - LLM Paper：对话生成章节对比
  - NeSy Paper：理论框架章节对比

### Phase 3: Paper 深度开发（3–5 天）

#### 任务1: 实验驱动型 Paper 收紧
- **1D-2D Fusion Paper**：
  - 完善结果+讨论章节
  - 准备性能对比图表（准确率、收敛曲线、模态贡献）
  - 物理对齐、语义对齐、几何对齐的可视化
  - 与现有SOTA方法的对比分析
- **MoE Paper**：
  - 专家激活模式分析（热力图）
  - 物理约束专家的解释性分析
  - 专家选择路由的可视化
  - 路径稀疏性对性能的影响分析
- **OperatorAttention Paper**：
  - 算子权重分布热力图
  - 注意力机制的可视化分析
  - 与TSPN、ResNet等基线的理论对比
  - 计算效率vs解释性权衡分析

#### 任务2: 工具 / 理论型 Paper 支撑
- **Toolkit Paper**：
  - 解释结果的benchmark对比表
  - 多模型解释一致性评估
  - 用户研究设计（可选）
- **LLM Paper**：
  - 离线对话样例收集（50+）
  - 主观评分标准和结果
  - 对话质量的多维度评估
- **Fuzzy-XFD Paper**：
  - Rules/NN/Hybrid对比表
  - 模糊规则的可解释性分析
  - 隶属函数优化结果
- **NeSy Paper**：
  - 7个子项目的NeSy映射表
  - 四层架构的理论完善
  - 形式化定义和证明框架

---

## 🔧 技术实施细节

### Fusion1D2D 修复策略
1. **立即行动流程（示意命令，请根据实际进程号调整）**：
   ```bash
   # 停止当前失败实验
   kill -9 <Fusion1D2D_PID>

   # 如需重启统一基线实验（当前 main.py 已使用简化版 Fusion1D2D）
   CUDA_VISIBLE_DEVICES=3 python main.py --config_dir configs/unified_baseline/config_Fusion1D2D.yaml
   ```
2. **并行调试方案**：
   - 创建`debug_fusion1d2d_shape.py`诊断脚本
   - 分析input_dim=4096, in_channels=2, out_channels=3的tensor flow
   - 逐层验证shape变化：`input → signal_processing → CNN reshape → classifier`

### 实验监控机制
1. **实时监控设置**：
   ```bash
   # 启动监控仪表板
   python experiment_monitor_dashboard.py
   ```
2. **数据收集频率**：
   - 每10分钟记录关键指标
   - 每个epoch结束后完整备份
   - 异常情况立即警报

3. **监控指标**：
   - 训练/验证损失曲线
   - 准确率变化趋势
   - L1正则化效果
   - GPU内存使用率
   - 训练速度（samples/sec）

### 文档标准化流程
1. **统一格式规范**：
   ```markdown
   ## 基线对比
   统一基线结果见 [unified_baseline_results_codex_11_28.md] 的第 X 行。

   ### 性能提升
   - 相对基线提升：+X%
   - 收敛速度提升：Y倍
   - 参数效率：Z params/M accuracy
   ```

2. **版本控制策略**：
   - 每次独立提交：`git commit -m "docs: update baseline results"`
   - 重要里程碑打tag：`git tag -a v1.0-baseline -m "Unified baseline v1.0"`
   - 分支管理：`feature/baseline-collection`

3. **质量保证检查清单**：
   - [ ] 数据来源明确标注
   - [ ] 统计显著性检验
   - [ ] 多次运行结果一致性
   - [ ] 图表坐标轴和标签完整
   - [ ] 引用格式统一

---

## 📊 预期成果与交付物

### 技术成果（Phase 1 完成时的预期）
- [ ] 3 个核心模型在当前配置下稳定运行若干 Epoch（如 50–100），日志与检查点完整保存  
- [ ] Fusion1D2D 的 shape 问题在验证阶段完全解决  
- [ ] 实验监控和数据收集体系可复用，关键脚本有文档说明  
- [ ] 超参数有初步优化记录（特别是 OperatorAttention 的 L1 正则化）  

### 文档成果（Phase 2 完成时的预期）
- [ ] `unified_baseline_results_codex_11_28.md` 形成完整结果表（每行带“快照 / 待复现 / 已验证”标注）  
- [ ] 7 个 Paper 的基线引用段落完成，并与统一表格链接  
- [ ] 标准化的结果展示格式（表头、图列、说明段落）  
- [ ] 关键实验配置文件在文档中有引用与说明  

### 学术成果（Phase 3 完成时的预期）
- [ ] 3 篇实验驱动型 Paper 的核心结果和主要图表初稿  
- [ ] 4 篇工具 / 理论型 Paper 的支撑材料初稿（映射表、示例图、接口说明等）  
- [ ] 可用于投稿的完整实验结果集（至少在 THU_018 上统一）  
- [ ] 统一的学术写作模板与引用规范草案  

---

## ⚠️ 风险控制与应急预案

### 实验连续性风险
1. **预防措施**：
   - 确保至少一个简化版本始终可用
   - 关键配置文件的多重备份
   - 实验环境容器化（Docker）准备

2. **应急方案**：
   - GPU故障：快速切换到备用GPU
   - 数据损坏：从WandB恢复训练状态
   - 内存不足：降低batch size或梯度累积

### 数据安全风险
1. **多重备份机制**：
   ```bash
   # 本地备份
   cp -r save/ backup/save_$(date +%Y%m%d_%H%M%S)/

   # 远程备份（如有）
   rsync -av save/ user@backup-server:/backup/unified_x_fault/
   ```

2. **版本管理**：
   - 每个重大修改独立分支
   - 完整的commit历史
   - 标签化重要里程碑

### 时间管理风险
1. **里程碑设置**：
   - Day 1: Fusion1D2D修复 + 监控系统
   - Day 3: 统一基线表初稿
   - Day 5: 3个核心Paper结果
   - Day 10: 完整Paper套装

2. **并行优化**：
   - 文档撰写与实验运行并行
   - 问题诊断与数据收集并行
   - Paper准备与超参数调优并行

---

## 🚀 立即行动清单（今日）

### 高优先级
1. **修复 Fusion1D2D**：
   - [ ] 停止当前失败实验
   - [ ] 启动Fusion1D2D_simple.py版本
   - [ ] 创建shape问题诊断脚本

2. **监控当前实验**：
   - [ ] 运行`python experiment_monitor_dashboard.py`
   - [ ] 调整OperatorAttention的L1系数到0.00001
   - [ ] 记录MoE和OperatorAttention的完整训练曲线

3. **创建文档框架**：
   - [ ] 建立`unified_baseline_results_codex_11_28.md`
   - [ ] 设计标准化的结果对比表头
   - [ ] 为7个Paper添加基线引用段落模板

### 中优先级
1. **数据收集**：
   - [ ] 备份所有模型检查点
   - [ ] 收集WandB训练日志
   - [ ] 创建性能指标时间戳记录

2. **长期准备**：
   - [ ] 为FuzzyLogic准备统一baseline配置
   - [ ] 规划Paper的图表和可视化需求
   - [ ] 准备交叉验证实验设计

---

**总结**: 本计划采用循序渐进、风险可控的方式，确保每个阶段都有明确的产出和交付物。通过系统化的实验管理、问题解决和文档建设，为后续的学术发表和项目推广奠定坚实基础。

**执行原则**: 数据安全第一、实验连续性优先、文档标准化并重、学术质量至上。
