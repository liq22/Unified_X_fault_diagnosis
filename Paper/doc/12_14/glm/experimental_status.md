# 实验执行状态报告

> **更新时间**：2024-12-14
> **项目**：1D-2D Fusion Explainable Fault Diagnosis
> **目标**：验证模型性能和可解释性评估

---

## 📊 实验概览

### 核心指标目标
- **诊断准确率**：≥95%（已达成95.7%）
- **可解释性指标**：
  - Faithfulness: ≥0.85
  - Stability: ≥0.90
  - Efficiency: ≤20ms/sample

---

## ✅ 已配置完成

### 1. 基础配置文件
- [x] `configs/unified_baseline/config_Fusion1D2D_seed20.yaml`
- [x] `configs/unified_baseline/config_Fusion1D2D_seed42.yaml`
- [x] `configs/unified_baseline/config_Fusion1D2D_seed2024.yaml`

### 2. 监控脚本
- [x] `scripts/monitor_three_seed.py` - 实时监控3个seed的实验进度
- [x] `scripts/analyze_three_seed_results.py` - 分析结果并生成统计报告
- [x] `scripts/validate_configs.py` - 验证配置文件一致性

### 3. 实验目录结构
```
results/stability/
├── seed_20/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
├── seed_42/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
├── seed_2024/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
├── logs/
└── figures/
```

---

## 🔄 当前实验状态

### THU_018数据集（99.57%基准）
- **状态**：待执行
- **配置**：3个seed已就绪
- **GPU分配**：
  - Seed 20 → GPU 1
  - Seed 42 → GPU 2
  - Seed 2024 → GPU 3
- **预估时间**：24小时
- **输出**：
  - 准确率：mean±std
  - 95%置信区间
  - 训练曲线对比图

### 多数据集验证

#### CWRU数据集
- **状态**：配置待创建
- **路径**：`/home/user/data/CWRU/`
- **任务**：滚动轴承故障诊断
- **类别**：Normal, Inner Race, Ball, Outer Race
- **配置文件**：`configs/config_CWRU.yaml`（待创建）

#### XJTU数据集
- **状态**：配置待创建
- **路径**：`/home/user/data/XJTU/`
- **任务**：轴承寿命预测与故障诊断
- **配置文件**：`configs/config_XJTU.yaml`（待创建）

#### THU_006数据集
- **状态**：配置待创建
- **路径**：`/home/user/data/PHMbenchdata/PHM-Vibench/`
- **任务**：齿轮箱故障诊断
- **配置文件**：`configs/config_THU_006.yaml`（待创建）

---

## 📈 可解释性评估实验

### Faithfulness (Del@k)
- **目标**：验证模型解释与实际决策的一致性
- **方法**：
  1. 提取特征重要性
  2. 迭代删除top-k重要特征
  3. 评估性能下降
- **预期**：忠实模型应显示显著性能下降

### Stability (Stab@σ)
- **目标**：测试解释对输入扰动的稳定性
- **方法**：
  1. 添加高斯噪声（σ=0.01, 0.02, 0.05）
  2. 计算解释相似度
  3. 评估稳定性指标
- **预期**：相似度应保持>0.9

### Efficiency
- **目标**：测量解释生成的时间成本
- **方法**：
  1. 记录解释生成时间
  2. GPU内存使用峰值
  3. 平均每样本耗时
- **目标**：<20ms/sample

---

## 🔧 实验执行命令

### 单个seed测试
```bash
# Seed 20
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config_file configs/unified_baseline/config_Fusion1D2D_seed20.yaml

# Seed 42
CUDA_VISIBLE_DEVICES=2 python main.py \
  --config_file configs/unified_baseline/config_Fusion1D2D_seed42.yaml

# Seed 2024
CUDA_VISIBLE_DEVICES=3 python main.py \
  --config_file configs/unified_baseline/config_Fusion1D2D_seed2024.yaml
```

### 并行执行（推荐）
```bash
# 使用提供的脚本
./scripts/run_three_seed_test.sh
```

### 监控进度
```bash
# 实时监控
python scripts/monitor_three_seed.py

# 检查GPU使用
watch -n 30 nvidia-smi
```

### 分析结果
```bash
# 生成统计报告
python scripts/analyze_three_seed_results.py

# 生成可视化
python scripts/generate_training_curves.py
```

---

## 📋 实验检查清单

### 实验前检查
- [ ] GPU可用性确认（nvidia-smi）
- [ ] 数据路径可访问
- [ ] 配置文件语法正确
- [ ] Python环境一致
- [ ] 磁盘空间充足（>10GB）

### 实验中检查
- [ ] Loss正常下降
- [ ] 无NaN或Inf
- [ ] GPU内存使用合理
- [ ] 日志正常记录

### 实验后检查
- [ ] Checkpoint保存成功
- [ ] 指标计算正确
- [ ] 可解释性评估完成
- [ ] 结果备份完整

---

## 🎯 成功标准

### 性能标准
- [ ] 平均准确率 ≥ 95%
- [ ] 标准差 ≤ 2%
- [ ] 95%置信区间合理

### 可解释性标准
- [ ] Faithfulness显著高于随机
- [ ] Stability ≥ 0.9
- [ ] Efficiency ≤ 20ms

### 可复现性标准
- [ ] 3个seed结果可重现
- [ ] 统计显著性p<0.05
- [ ] 配置文件版本控制

---

## 🚨 问题预案

### 常见问题与解决方案

1. **OOM错误**
   - 减小batch_size（64→32→16）
   - 使用gradient accumulation
   - 检查GPU内存泄漏

2. **不收敛**
   - 检查学习率（尝试1e-4）
   - 增加warmup epochs
   - 验证数据预处理

3. **结果差异大**
   - 检查随机种子设置
   - 验证数据划分一致性
   - 确认超参数完全相同

4. **可解释性评估失败**
   - 检查梯度计算
   - 验证模型是否requires_grad
   - 调试特征提取过程

---

## 📊 实验时间线

### Day 1-2
- 00:00 启动3-seed实验
- 06:00 检查初始进度
- 12:00 中期检查
- 18:00 继续监控
- 24:00 状态确认

### Day 3
- 06:00 收集结果
- 12:00 统计分析
- 18:00 生成报告
- 24:00 准备下一阶段

---

## 📝 实验记录模板

### 实验 Log 示例
```
=== 2024-12-14 14:30:00 ===
Experiment: Fusion1D2D (seed=20)
Dataset: THU_018
Epoch: 45/100
Train Loss: 0.0234
Val Loss: 0.0456
Val Acc: 0.9421
GPU Mem: 6.2GB/24.0GB
Status: Normal
```

### 结果记录示例
```json
{
  "seed": 20,
  "final_accuracy": 0.957,
  "final_f1": 0.9568,
  "best_epoch": 87,
  "training_time": 2.3,
  "convergence_epoch": 45,
  "explainability": {
    "faithfulness": 0.89,
    "stability": 0.92,
    "efficiency": 15.3
  }
}
```

---

**备注**：所有实验结果将自动保存到`results/stability/`目录，并定期备份到云存储。