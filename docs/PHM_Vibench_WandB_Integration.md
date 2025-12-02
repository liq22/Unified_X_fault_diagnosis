# PHM-Vibench W&B实验跟踪集成指南

本文档介绍如何将PHM-Vibench数据集实验与Weights & Biases (W&B)集成，实现实验的自动化跟踪和可视化。

## 1. W&B项目配置

### 1.1 项目结构
PHM-Vibench实验使用以下W&B项目结构：

- **主项目**: `PHM-Vibench-Unified-Baseline`
  - **TSPN组**: TSPN模型的所有实验
  - **TKAN组**: TKAN模型的所有实验
  - **NNSPN组**: NNSPN模型的所有实验
  - **OperatorAttention组**: 算子注意力模型实验
  - **FuzzyLogic组**: 模糊逻辑模型实验
  - **Baseline-Models组**: 对比模型实验

- **测试项目**: `PHM-Vibench-Test`
  - 用于调试和验证的小规模实验

- **专项实验项目**:
  - `PHM-Domain-Adaptation`: 域自适应实验
  - `PHM-Few-Shot`: 少样本学习实验

### 1.2 实验命名规范

W&B实验命名遵循以下规范：
```
{Model}_{Dataset}_{ExperimentType}_{Date}_{ConfigHash}
```

示例：
- `TSPN_CWRU_baseline_20251201_abc123`
- `TKAN_PHM_fewshot_20251201_def456`

## 2. 配置文件集成

### 2.1 基础W&B配置
每个配置文件都包含W&B设置：

```yaml
args:
  # W&B项目配置
  wandb_project: 'PHM-Vibench-Unified-Baseline'
  wandb_group: 'TSPN'
  wandb_job_type: 'train'
  wandb_tags: ['PHM', 'fault_diagnosis', 'baseline']
```

### 2.2 自动日志记录
配置文件自动记录：
- 训练/验证损失和准确率
- 模型超参数
- 数据集信息
- 系统资源使用情况

## 3. 实验跟踪内容

### 3.1 自动跟踪的指标
- **训练指标**: loss, accuracy, learning_rate
- **验证指标**: val_loss, val_accuracy
- **测试指标**: test_loss, test_accuracy
- **模型信息**: 参数数量, 模型大小
- **数据信息**: 数据集大小, 类别分布
- **系统信息**: GPU使用率, 内存使用情况

### 3.2 手动跟踪内容
- 混淆矩阵
- 类别级别的准确率
- 训练时间统计
- 特殊实验结果

## 4. 使用方法

### 4.1 设置W&B API密钥
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

### 4.2 运行实验
```bash
# 使用配置文件运行
python main.py --config_file configs/PHM_Vibench/config_TSPN.yaml

# 使用批量脚本运行
./script/run_PHM_baseline.sh
```

### 4.3 离线模式
```bash
export WANDB_MODE="offline"
python main.py --config_file configs/PHM_Vibench/config_TSPN.yaml
```

## 5. 实验分析

### 5.1 W&B Dashboard访问
训练完成后，可在以下链接访问实验结果：
- 主项目: https://wandb.ai/your_username/PHM-Vibench-Unified-Baseline
- 测试项目: https://wandb.ai/your_username/PHM-Vibench-Test

### 5.2 常用分析视图
1. **并排对比**: 选择多个实验进行参数和性能对比
2. **超参数重要性**: 分析超参数对性能的影响
3. **混淆矩阵**: 可视化分类结果
4. **学习曲线**: 监控训练过程

### 5.3 导出结果
```python
import wandb

# 连接到项目
api = wandb.Api()

# 获取项目运行
runs = api.runs("your_username/PHM-Vibench-Unified-Baseline")

# 导出到CSV
summary_df = wandb.runs_to_dataframe(runs)
summary_df.to_csv("experiment_results.csv")
```

## 6. 最佳实践

### 6.1 实验组织
- 使用有意义的group和tag
- 在description中详细记录实验目的
- 定期清理过期实验

### 6.2 版本控制
- 记录代码版本 (git commit hash)
- 保存数据集版本信息
- 标注实验环境的依赖版本

### 6.3 资源管理
- 监控GPU使用情况
- 设置实验预算限制
- 使用优先级队列管理实验

## 7. 故障排除

### 7.1 常见问题
1. **API密钥错误**: 检查WANDB_API_KEY环境变量
2. **项目权限**: 确保对项目有写入权限
3. **网络连接**: 检查与W&B服务器的连接
4. **磁盘空间**: 确保有足够空间保存日志

### 7.2 调试方法
```bash
# 详细日志
export WANDB_VERBOSE=true

# 调试模式
export WANDB_DEBUG=true

# 同步状态检查
wandb status
```

## 8. 示例配置

### 8.1 完整的W&B集成配置
```yaml
args:
  # 实验标识
  experiment_type: 'PHM_unified_baseline'
  description: 'TSPN透明信号处理网络在PHM-Vibench数据集上的统一基线实验'

  # W&B项目配置
  wandb_project: 'PHM-Vibench-Unified-Baseline'
  wandb_group: 'TSPN'
  wandb_job_type: 'train'
  wandb_tags:
    - 'PHM'
    - 'fault_diagnosis'
    - 'TSPN'
    - 'baseline'
    - 'signal_processing'

  # W&B高级配置
  wandb_notes: |
    实验目标：验证TSPN在PHM-Vibench数据集上的性能
    数据集：CWRU, XJTU, FEMTO, THU, MFPT, UNSW
    主要改进：使用智能采样和动态窗口

  wandb_config:
    log_model: true  # 保存模型到W&B
    log_graph: true  # 保存计算图
  ```

通过以上配置，PHM-Vibench实验可以完全集成到W&B平台，实现高效的实验管理和结果分析。