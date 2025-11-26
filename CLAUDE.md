# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个统一的、可解释的故障诊断方法框架（UXFD），集成了完整的学术研究辅助系统。项目基于 PyTorch 和 PyTorch Lightning 构建，实现了多种透明信号处理网络（TSPN）和对比模型，并提供从文献调研到论文发表的全流程学术研究支持。

### 🎯 双重定位
1. **机器学习研究平台**：专注于故障诊断领域的深度学习方法研究
2. **学术写作助手**：集成18个专业AI代理，支持Nature级别论文写作

### ✨ 核心特性
- **透明信号处理网络**：TSPN、NNSPN、TKAN等可解释模型
- **智能缓存系统**：自动管理研究过程和思路
- **18个专业代理**：覆盖研究、写作、代码全链条
- **学术工作流**：文献综述→实验设计→论文写作→投稿

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

## 🤖 学术研究系统

### 18个专业AI代理

#### 研究类代理（7个）
- **research-literature**: 文献搜索与分析专家
- **research-knowledge-graph**: 知识图谱构建专家
- **research-hypothesis**: AI驱动假设生成
- **research-gap-identifier**: 研究空白识别
- **research-trends**: 研究趋势分析
- **research-academic**: 学术文献专家
- **research-semantic-scholar**: Semantic Scholar集成

#### 写作类代理（8个）
- **写作集群**: intro, method, results, discussion, format
- **质量控制**: cache-manager, quality-controller, style-formatter

#### 技术类代理（3个）
- **coder-industrial-ai**: 工业AI部署专家
- **coder-reviewer**: 代码审查专家
- **coder-debugger**: 调试与错误诊断

### 智能工作流

#### 1. 文献管理
```bash
# 自动文献搜索
/research-literature --topic "fault diagnosis attention mechanism"
# 生成知识图谱
/research-knowledge-graph --domain "signal processing"
```

#### 2. 实验设计
```bash
# 创建实验计划
./.claude/scripts/lq_save_plan.sh --slug experiment_design --apply
# 审查实验进展
./.claude/scripts/lq_review_recent.sh --days 7
```

#### 3. 论文写作
```bash
# 生成论文引言
/research-intro --topic "transparent fault diagnosis"
# 优化论文格式
/writer-format --venue "Nature Machine Intelligence"
```

#### 4. 版本控制
```bash
# 智能Git提交
./.claude/scripts/lq_git_commit.sh --apply
# Bug跟踪
/bug-create --title "Model convergence issue"
```

### Hook系统
- **pre-execute**: 执行前自动捕获上下文
- **post-execute**: 执行后自动保存结果
- **auto-cache**: 智能缓存研究过程

### 模板系统
- **设计模板**: 9个专业模板覆盖研究全流程
- **任务模板**: 结构化任务管理
- **输出样式**: Nature/Science/IEEE等多种格式

## 💡 使用技巧

### 学术研究最佳实践
1. **开始新研究**
   - 使用 `research-literature` 进行文献调研
   - 用 `lq_save_plan` 创建研究计划
   - 通过 `research-gap-identifier` 找到创新点

2. **实验管理**
   - 使用 `spec-create` 定义实验规格
   - 通过 `bug-track` 管理实验问题
   - 用 `lq_review_recent` 定期检查进展

3. **论文写作**
   - 采用分模块写作策略
   - 利用质量控制代理提升质量
   - 自动适配目标期刊格式

## 其他

见 @AGENTS.md 获取对照信息