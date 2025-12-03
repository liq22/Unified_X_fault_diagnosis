# Vbench数据集使用指南

## 快速开始

### 1. 环境准备

确保已安装所有依赖：
```bash
pip install pandas openpyxl h5py torch scikit-learn
```

### 2. 数据准备

Vbench数据目录结构：
```
/home/user/data/PHMbenchdata/PHM-Vibench/
├── metadata_6_11.xlsx  # 元数据（包含Dataset_id 1-20）
├── data.h5            # 信号数据（使用Id作为键）
└── corpus.xlsx           # 语料库（可选）
```

### 3. 核心功能说明

#### VbenchDataset类
- **数据加载**：支持HDF5内存映射，高效处理大规模数据
- **智能采样**：通过SmartSampler控制每类样本数量
- **滑动窗口**：可配置窗口长度和步长
- **域自适应**：支持留一法跨域评估

#### SmartSampler类
- **ID池机制**：默认50个ID，自动处理不足情况
- **均衡采样**：确保每类样本数一致
- **灵活配置**：支持随机/顺序/分层采样策略

## 使用方式

### 单数据集实验

```bash
# 使用CWRU数据集（Dataset_id=1）
python main.py --config_file configs/vbench/config_vbench_diagnosis.yaml \
    --dataset_task VBENCH_1 \
    --num_epochs 100

# 使用XJTU数据集（Dataset_id=2）
python main.py --config_file configs/vbench/config_vbench_diagnosis.yaml \
    --dataset_task VBENCH_2
```

### 批量测试所有数据集

#### 遍历所有20个数据集

```bash
# 快速测试模式（每个数据集10个epoch）
./scripts/test_all_vbench_datasets.py --all-datasets --epochs 10

# 测试指定范围（如数据集1-5）
./scripts/test_all_vbench_datasets.py --id-range 1-5 --epochs 50

# 测试单个数据集
./scripts/test_all_vbench_datasets.py --single-dataset 3

# 完整训练模式（每个数据集100个epoch）
./scripts/test_all_vbench_datasets.py --all-datasets --epochs 100
```

### 批量并行训练

```bash
# 使用4个GPU并行测试
./scripts/test_all_vbench_datasets.py --all-datasets --parallel 4

# 结合resum使用（从上次中断处继续）
./scripts/test_all_vbench_datasets.py --all-datasets --resume
```

## 配置说明

### 数据集ID列表

Vbench支持以下Dataset_id：

| Dataset_id | 数据集名称 | 说明 |
|-----------|------------|------|
| 1 | CWRU | Case Western Reserve University |
| 2 | XJTU | Xi'an Jiaotong University |
| 3 | FEMTO | FEMTO-STT bearing dataset |
| 4 | MUST | Mitsubishi University |
| 5 | PU | Paderborn University |
| ... | ... | ... |
| 20 | HUST23 | Huazhong University |

### 配置文件示例

#### 基础配置
```yaml
vbench_config:
  dataset_ids: [1]  # 直接使用Dataset ID数字
  sampling_config:
    target_per_class: 1000  # 每类样本数
    ids_cap: 50           # ID池大小
```

#### 快速测试配置
```yaml
vbench_config:
  dataset_ids: [1, 2, 3]  # 测试前3个数据集
  sampling_config:
    target_per_class: 100    # 小规模测试
    ids_cap: 20
args:
  num_epochs: 10  # 快速验证
```

## 结果查看

### 1. WandB可视化
访问 [https://wandb.ai](https://wandb.ai) 查看实时训练曲线

### 2. 本地结果
```bash
# 查看测试汇总
cat results/vbench_test/test_summary.yaml

# 查看CSV结果
cat results/vbench_test/test_results.csv

# 查看特定实验日志
ls -la logs/
```

## 常见问题

### Q1: 如何指定特定Dataset_id？
A: 在配置文件中设置：
```yaml
vbench_config:
  dataset_ids: [1, 5, 10]  # 选择特定数据集
```

### Q2: 如何调整每类样本数？
A: 修改sampling_config：
```yaml
sampling_config:
  target_per_class: 500  # 从1000减少到500
```

### Q3: 如何启用数据增强？
A: 设置：
```yaml
augmentation:
  enable: true
  noise_std: 0.02
```

### Q4: 内存不足怎么办？
A: 减少批量大小或worker数量：
```yaml
args:
  batch_size: 16  # 从64减少到16
  num_workers: 4  # 从8减少到4
```

## 最佳实践

1. **先快速测试**：使用小数据集和小epoch验证配置
2. **逐步扩展**：成功后逐步增加数据集规模
3. **记录实验**：使用WandB跟踪所有超参数和结果
4. **版本控制**：保存每个实验的配置文件
5. **资源监控**：使用`nvidia-smi`监控GPU使用

## 下一步

1. 运行单个数据集测试
2. 分析结果，选择最佳数据集
3. 进行超参数搜索
4. 扩展到所有数据集
5. 撰写研究报告

---

现在您可以开始使用Vbench数据集进行故障诊断实验了！