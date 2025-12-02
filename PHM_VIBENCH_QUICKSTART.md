# PHM-Vibench 快速开始指南

## 🚀 快速开始

PHM-Vibench数据集已成功集成到统一基线框架！以下是快速使用方法：

### 1. 环境准备
```bash
# 激活环境
source activate LQ_signal

# 设置W&B密钥（可选）
export WANDB_API_KEY="your_wandb_api_key"
```

### 2. 快速验证
```bash
# 验证集成是否成功
python verify_phm_integration.py

# 运行一个简单测试
python main.py --config_file configs/PHM_Vibench/config_TSPN_test.yaml
```

### 3. 运行完整实验

#### 单模型实验
```bash
# TSPN模型
python main.py --config_file configs/PHM_Vibench/config_TSPN.yaml

# TKAN模型
python main.py --config_file configs/PHM_Vibench/config_TKAN.yaml

# 对比模型
python main_com.py --config_file configs/PHM_Vibench/config_com.yaml
```

#### 批量实验
```bash
# 运行所有基线模型
./script/run_PHM_baseline.sh

# 域自适应实验
./script/run_PHM_domain_adaptation.sh

# 少样本学习实验
./script/run_PHM_few_shot.sh
```

### 4. 监控实验
```bash
# 一次性状态检查
python script/monitor_phm_experiments.py --once

# 实时监控（60秒间隔）
python script/monitor_phm_experiments.py
```

## 📊 实验场景

### 场景1: 基础故障诊断
测试不同模型在PHM数据集上的性能
```bash
# 运行5个主要模型
./script/run_PHM_baseline.sh
```

### 场景2: 跨数据集泛化
验证模型的泛化能力
```bash
# 留一法域自适应
./script/run_PHM_domain_adaptation.sh
```

### 场景3: 少样本学习
测试在样本有限条件下的性能
```bash
# 1/5/10/20/50-shot实验
./script/run_PHM_few_shot.sh
```

## ⚙️ 自定义配置

### 选择特定数据集
编辑配置文件中的`dataset_ids`：
```yaml
vbench_config:
  dataset_ids: [1]      # 仅CWRU
  # dataset_ids: [1,2] # CWRU + XJTU
  # dataset_ids: [1,2,3,6,7,8] # 全部数据集
```

### 调整采样策略
```yaml
vbench_config:
  sampling_config:
    method: "smart"      # 智能/随机/平衡
    target_per_class: 500 # 每类样本数
    window_length: 4096  # 窗口长度
```

### 修改训练参数
```yaml
args:
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 100
  gpus: 1  # GPU数量
```

## 📈 查看结果

### W&B Dashboard
- 访问 https://wandb.ai/your_username/PHM-Vibench-Unified-Baseline
- 查看实时训练曲线和对比结果

### 本地日志
```bash
# 查看实验日志
ls logs/PHM_*

# 查看特定实验结果
tail -f logs/PHM_*/TSPN_*.log
```

## 🔧 故障排除

### 常见问题
1. **内存不足**: 减少`target_per_class`或`batch_size`
2. **CUDA错误**: 检查GPU可用性 `nvidia-smi`
3. **采样错误**: 使用简化配置`config_TSPN_test.yaml`

### 获取帮助
```bash
# 查看完整文档
cat PHM_VIBENCH_INTEGRATION_SUMMARY.md

# W&B集成指南
cat docs/PHM_Vibench_WandB_Integration.md

# 验证集成状态
python verify_phm_integration.py
```

## 🎯 下一步

1. **运行你的第一个实验**:
   ```bash
   python main.py --config_file configs/PHM_Vibench/config_TSPN_test.yaml
   ```

2. **扩展到更多数据集**:
   编辑配置文件中的`dataset_ids`

3. **尝试高级功能**:
   - 域自适应实验
   - 少样本学习
   - 对比模型评估

4. **分析结果**:
   - 使用W&B进行可视化分析
   - 导出结果进行论文写作

---

🎉 **恭喜！PHM-Vibench数据集已准备就绪，开始你的故障诊断研究吧！**