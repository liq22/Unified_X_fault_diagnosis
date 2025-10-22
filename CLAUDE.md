# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个统一的、可解释的故障诊断方法框架，用于信号处理和故障诊断研究。项目基于 PyTorch 和 PyTorch Lightning 构建，实现了多种透明信号处理网络（TSPN）和对比模型。

## 环境配置

### 创建环境
```shell
conda env create -f environment.yml
conda activate UXFD
```

### 主要依赖
- Python 3.9
- PyTorch 2.1.2+cu121
- PyTorch Lightning 2.1.3
- Weights & Biases 0.17.3（实验跟踪）
- ptwt 0.1.7（小波变换）

## 常用命令

### 基本运行
```shell
# 运行主要模型（TSPN, TFON, NNSPN, TKAN）
python main.py --config_file configs/THU_018/config_TSPN.yaml

# 运行对比模型（ResNet, WKN, SincNet, MCN 等）
python main_com.py --config_file configs/THU_018/config_com.yaml

# 快速演示
./script/demo.sh
```

### 特殊实验
```shell
# K-shot 少样本学习实验
python main_kshotexp.py --config_file configs/THU_018/config_TSPN_kshot.yaml

# 消融实验
python main_ablation_exp.py --config_file configs/THU_018/config_TSPN.yaml

# K-shot 对比实验
python main_com_kshotexp.py --config_file configs/THU_018/config_com_kshot.yaml
```

### 批量实验脚本
```shell
# TSPN 消融实验
./script/run_TSPN_ablation.sh

# K-shot 学习实验
./script/run_kshot_exp.sh

# TFON 论文实验
./script/TFON_paper/run_com.sh
```

## 项目架构

### 核心组件

1. **信号处理层** (`model/Signal_processing.py`)
   - FFT：快速傅里叶变换
   - HT：希尔伯特变换
   - WF：小波滤波
   - I：恒等变换
   - LNO：拉普拉斯神经算子

2. **特征提取器** (`model/Feature_extract.py`)
   - 13 种统计特征：均值、标准差、方差、熵、最大/最小值、绝对均值、峰度、均方根、峰值因子、偏度、间隙因子、形状因子

3. **主要模型**
   - `TSPN.py`：透明信号处理网络
   - `TFON.py`：时频算子网络（需添加）
   - `NNSPN.py`：神经信号处理网络
   - `TKAN.py`：时间 Kolmogorov-Arnold 网络

4. **训练框架** (`trainer/`)
   - `trainer_basic.py`：基础训练循环
   - `trainer_set.py`：训练配置（日志、检查点、剪枝）
   - `utils.py`：损失函数和回调类

### 配置系统

配置文件位于 `configs/` 目录，按数据集组织：
- `config_basic.yaml`：基础配置模板
- `config_com.yaml`：对比模型配置
- 数据集特定配置：`THU_006/`、`THU_018/`、`DIRG/`

### 数据集任务映射

```python
DATASET_TASK_CLASS = {
    'THU_006_basic': THU_006or018_basic,
    'THU_018_basic': THU_006or018_basic,
    'THU_018_few_shot': THU_006or018_few_shot,
    'THU_006_few_shot': THU_006or018_few_shot,
    'THU_006_generalization': THU_006_generalization
}
```

## 关键参数说明

### 信号处理配置
- `layer1-layer4`：四层信号处理模块配置，每层可选择 ['I', 'WF', 'HT', 'FFT', 'LNO']

### 模型参数
- `in_dim/out_dim`：输入/输出维度（通常为 4096）
- `in_channels/out_channels`：输入/输出通道数
- `scale`：缩放比例（默认为 4）
- `skip_connection`：是否使用残差连接

### 训练参数
- `monitor`：监控指标（默认 'val_loss'）
- `patience`：早停等待轮数
- `l1_norm`：L1 正则化系数
- `pruning`：剪枝比例（可选）
- `snr`：信噪比

## 实验跟踪

项目使用 Weights & Biases 进行实验跟踪，自动记录：
- 训练/验证损失和准确率
- 模型检查点
- 超参数配置
- 可视化结果

## 开发注意事项

1. **数据路径**：确保在配置文件中正确设置 `data_dir`
2. **目标设置**：`target` 参数决定目标任务（如 'IF' 表示内圈故障）
3. **GPU 使用**：通过 `gpus` 参数设置使用的 GPU 数量
4. **随机种子**：使用 `seed` 参数确保实验可重复性
5. **K-shot 学习**：设置 `k_shot` 参数控制每类样本数量

## 模型对比

支持的对比模型在 `model_collection/` 中：
- 传统深度学习：ResNet、SincNet
- 专业信号处理：WKN、MCN、TFN
- 极限学习机：EELM
- 符号学习：F_EQL、EQL

## 结果分析

- 模型检查点保存在 `save/` 目录
- 使用 `post/` 中的工具进行后处理和分析
- `post_analysis.ipynb` 提供交互式分析界面

## 其他

见 @AGENTS.md 获取对照信息