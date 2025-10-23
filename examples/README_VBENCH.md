# Vbench数据集使用指南

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install pandas openpyxl h5py torch
```

### 2. 数据准备

将Vbench数据下载或解压到：
```
/home/user/data/PHMbenchdata/PHM-Vibench/
├── metadata_6_11.xlsx  # 元数据
├── data.h5             # 信号数据
└── corpus.xlsx           # 语料库（可选）
```

### 3. 运行示例

#### 方式1：使用演示脚本
```bash
# 交互式运行
./scripts/run_vbench_demo.sh
```

#### 方式2：命令行直接运行
```bash
# 使用默认配置
python main.py --config_file configs/vbench/config_vbench_diagnosis.yaml

# 使用CWRU数据集
python main.py --config_file configs/vbench/RM_001_CWRU/config_TSPN.yaml

# 使用自定义参数
python main.py \
    --config_file configs/vbench/config_vbench_diagnosis.yaml \
    --model TSPN \
    --batch_size 32 \
    --num_epochs 100
```

#### 方式3：Python代码调用
```python
from data.vbench_dataset import VbenchDataset
from data.vbench_utils import SmartSampler

# 创建数据集
dataset = VbenchDataset(args, flag='train')

# 使用数据加载器
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    data, labels = batch
    # 训练代码...
```

## 配置说明

### 核心配置项

| 配置项 | 说明 | 默认值 |
|---------|------|--------|
| `data_dir` | 数据目录 | /home/user/data/PHMbenchdata/PHM-Vibench |
| `metadata_file` | 元数据文件名 | metadata_6_11.xlsx |
| `data_file` | HDF5数据文件名 | data.h5 |
| `task_type` | 任务类型 | fault_diagnosis |
| `target_column` | 目标列 | Label |

### 采样配置

```yaml
sampling_config:
  method: "smart"           # 采样方法
  ids_cap: 50               # ID池大小
  target_per_class: 1000     # 每类样本数
  window_strategy: "random"   # 窗口策略
```

### 域自适应配置

```yaml
domain_config:
  leave_one_domain_out: true  # 留一法
  target_domain: null         # null=自动选择
```

## 常见问题

### Q: 如何修改每类样本数？
A: 修改配置文件中的 `target_per_class` 值

### Q: 如何使用特定数据集？
A: 修改 `dataset_ids` 列表，例如：
```yaml
dataset_ids: ["RM_001_CWRU"]  # 仅CWRU
```

### Q: 如何调整窗口大小？
A: 修改 `window_config`：
```yaml
window_config:
  window_length: 2048  # 窗口长度
  stride: 512         # 滑动步长
```

### Q: 如何启用数据增强？
A: 设置 `augmentation.enable: true`

## 性能优化建议

1. **使用小数据集调试**：设置 `target_per_class: 100`
2. **调整批大小**：根据显存调整 `batch_size`
3. **多GPU训练**：设置 `gpus: N` 并行
4. **启用缓存**：设置 `use_cache: true`

## 扩展指南

### 添加新数据集

1. 将数据转换为HDF5+Excel格式
2. 在metadata.xlsx中添加数据集信息
3. 更新配置文件的 `dataset_ids`

### 自定义采样器

继承 `SmartSampler` 类并重写方法：
```python
class CustomSampler(SmartSampler):
    def sample_class(self, label_id_pool, label_name):
        # 自定义采样逻辑
        return super().sample_class(label_id_pool, label_name)
```

## 实验模板

### 标准故障诊断实验

```bash
# 1. 准备配置
cp configs/vbench/config_vbench_diagnosis.yaml my_exp.yaml

# 2. 修改参数
# 编辑 my_exp.yaml

# 3. 运行实验
python main.py --config_file my_exp.yaml

# 4. 查看结果
# WandB: https://wandb.ai/
# 本地: save/my_exp/
```

### 超参数搜索

创建批量脚本：
```bash
#!/bin/bash
for lr in 0.001 0.0005 0.0001; do
    python main.py \
        --config_file configs/vbench/config_vbench_diagnosis.yaml \
        --learning_rate $lr \
        --experiment_name "lr_$lr"
done
```

## 下一步

1. 查看 `examples/vbench_usage_example.py` 了解完整实现
2. 运行 `scripts/run_vbench_demo.sh` 快速开始
3. 根据需要修改配置文件进行实验
4. 使用WandB跟踪实验结果