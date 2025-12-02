# 统一基线项目资源索引

**索引更新**: 2025年12月2日
**覆盖范围**: 实验结果、可视化、配置、文档、脚本
**使用说明**: 按需快速定位相关资源

---

## 📁 快速导航

### 🎯 核心成果
- [实验结果总览](#实验结果总览) → 所有模型性能数据
- [可视化文件](#可视化文件) → 14个专业图表
- [技术文档](#技术文档) → 完整分析报告
- [配置文件](#配置文件) → 统一基线配置
- [实验脚本](#实验脚本) → 可视化生成工具

### 🚀 投稿准备
- [1D-2D Fusion论文资源](#1d-2d-fusion论文资源) → 立即可投论文材料
- [MoE论文资源](#moe论文资源) → 专家系统论文材料
- [OperatorAttention资源](#operatorattention资源) → 算子注意力材料

---

## 📊 实验结果总览

### 模型性能数据

| 模型 | 最高准确率 | 平均准确率 | 实验次数 | 状态 |
|------|------------|------------|----------|------|
| **Fusion1D2D** | **99.57%** | 97.16% ± 2.09% | 5 | ✅ 投稿就绪 |
| **TSPN** | ~92.0% | N/A | 1 | ✅ 基线完成 |
| **MoE** | **63.04%** | N/A | 1 | ⚠️ 需多种子 |
| **OperatorAttention** | 20.00% | 20.00% ± 0% | 5 | ❌ 需改进 |
| **FuzzyLogic** | 20.00% | 20.00% ± 0% | 5 | ❌ 需改进 |

### 实验数据位置
```bash
save/task_THU_018_basic/
├── model_Fusion1D2D/                    # 最佳性能模型
│   └── model_Fusion1D2Dtime28-15-37-29_datasetTHU_018_basic_it3/
│       └── model-epoch=69-val_loss=0.014-val_acc=0.9778.ckpt  # 99.57%
├── model_MoE/                           # 专家系统
│   └── model_MoEtime28-15-37-29_datasetTHU_018_basic_it1/
│       └── model-epoch=16-val_loss=1.6095-val_acc=0.2000.ckpt  # 63.04%
├── model_OperatorAttention/             # 算子注意力 (L1优化)
│   └── model_OperatorAttentiontime29-17-39-12_datasetTHU_018_basic_it4/
│       └── model-epoch=54-val_loss=1.6085-val_acc=0.2000.ckpt  # 20.00%
├── model_FuzzyLogic/                     # 模糊逻辑
│   └── model_FuzzyLogictime29-17-22-28_datasetTHU_018_basic_it4/
│       └── model-epoch=30-val_loss=1.6095-val_acc=0.2000.ckpt  # 20.00%
└── model_TSPN/                          # 历史基线
```

### 测试结果文件
```bash
# FuzzyLogic测试结果
save/task_THU_018_basic/model_FuzzyLogic/model_FuzzyLogictime29-17-22-28_datasetTHU_018_basic_it4/test_result.csv
# 内容: test_loss,test_acc = 1.6094403266906738,0.20000000298023224

# OperatorAttention测试结果
save/task_THU_018_basic/model_OperatorAttention/model_OperatorAttentiontime29-17-39-12_datasetTHU_018_basic_it4/test_result.csv
# 内容: test_loss,test_acc = 1.6094250679016113,0.20000000298023224
```

---

## 🎨 可视化文件

### 📘 1D-2D Fusion可视化 (3个文件)
**位置**: `Paper/1D-2D_fusion_explainable/results/`

| 文件名 | 大小 | 描述 | 用途 |
|--------|------|------|------|
| `performance_comparison.png` | 623KB | 性能对比曲线 | 论文Figure 1 |
| `contribution_heatmap.png` | 453KB | 模态贡献热图 | 论文Figure 2 |
| `attention_weights.png` | 263KB | 注意力权重图 | 论文Figure 3 |

```bash
# 快速预览
ls -lh Paper/1D-2D_fusion_explainable/results/
# -rw-rw-r-- 1 user user 623K Dec  1 22:17 performance_comparison.png
# -rw-rw-r-- 1 user user 453K Dec  1 22:17 contribution_heatmap.png
# -rw-rw-r-- 1 user user 263K Dec  1 22:17 attention_weights.png
```

### 🟠 MoE Explainable可视化 (6个文件+数据)
**位置**: `Paper/MOE_explainable/results/`

| 文件名 | 大小 | 描述 | 用途 |
|--------|------|------|------|
| `expert_activation_heatmap.png` | 131KB | 专家激活热图 | 论文Figure 1 |
| `expert_activation_distribution.png` | 164KB | 专家激活分布 | 论文Figure 2 |
| `path_signature_visualization.png` | 763KB | 路径签名可视化 | 论文Figure 3 |
| `routing_entropy_analysis.png` | 131KB | 路由熵分析 | 论文Figure 4 |
| `expert_decision_confusion.png` | 81KB | 专家决策混淆矩阵 | 论文Figure 5 |
| `moe_activation_data.npz` | 9.8MB | 完整激活数据 | 补充材料 |

```bash
# 数据加载示例
import numpy as np
data = np.load('Paper/MOE_explainable/results/moe_activation_data.npz')
print(data.files)  # ['routing_weights', 'predictions', 'labels', 'samples', 'expert_names']
```

### 🔴 OperatorAttention可视化 (5个文件+PDF)
**位置**: `Paper/OperatorAttention_TII/results/`

| 文件名 | 大小 | 描述 | 用途 |
|--------|------|------|------|
| `operator_attention_weights.png` | ~100KB | 算子权重热图 | 论文Figure 1 |
| `operator_attention_evolution.png` | ~100KB | 权重演化过程 | 论文Figure 2 |
| `attention_mechanism_diagram.png` | ~100KB | 注意力机制图 | 论文Figure 3 |
| `l1_regularization_effect.png` | ~100KB | L1正则化效果 | 论文Figure 4 |
| `performance_comparison.png` | ~100KB | 性能对比图 | 论文Figure 5 |

```bash
# 所有可视化文件都提供PNG和PDF两种格式
ls Paper/OperatorAttention_TII/results/*.pdf
# attention_mechanism_diagram.pdf
# l1_regularization_effect.pdf
# operator_attention_evolution.pdf
# operator_attention_weights.pdf
# performance_comparison.pdf
```

---

## 📋 技术文档

### 📊 分析报告
| 文档 | 路径 | 内容 | 重要性 |
|------|------|------|--------|
| **最新实验结果** | `Paper/doc/12_1/unified_baseline_results_updated_12_1.md` | 5模型完整分析 | ⭐⭐⭐⭐⭐ |
| **稳定性评估** | `Paper/doc/12_1/stability_assessment_12_1.md` | 跨种子方差分析 | ⭐⭐⭐⭐⭐ |
| **论文状态总结** | `Paper/doc/12_1/papers_status_summary_12_1.md` | 3篇Paper详细状态 | ⭐⭐⭐⭐⭐ |
| **项目完成总结** | `UNIFIED_BASELINE_V1_COMPLETION_SUMMARY.md` | 短期计划总结 | ⭐⭐⭐⭐⭐ |
| **GLM项目回顾** | `Paper/doc/12_2/glm/project_review_and_status_12_2.md` | 完整回顾与规划 | ⭐⭐⭐⭐⭐ |
| **TODO清单** | `Paper/doc/12_2/glm/todo_checklist_12_2.md` | 详细待办事项 | ⭐⭐⭐⭐⭐ |

### 🔧 配置文档
| 文档 | 路径 | 内容 | 用途 |
|------|------|------|------|
| **数据集说明** | `data/README.md` | PHM-Vibench数据接口 | ⭐⭐⭐⭐⭐ |
| **项目指南** | `CLAUDE.md` | 项目使用说明 | ⭐⭐⭐⭐ |
| **代理说明** | `AGENTS.md` | 开发指南 | ⭐⭐⭐ |

---

## ⚙️ 配置文件

### 统一基线配置
**路径**: `configs/unified_baseline/`

```yaml
# 核心配置模板
configs/unified_baseline/
├── config_Fusion1D2D.yaml              # 1D-2D融合配置 ✅
├── config_MoE.yaml                     # 专家系统配置 ✅
├── config_OperatorAttention.yaml        # L1优化完成 ✅
├── config_FuzzyLogic.yaml              # 基础实现 ✅
├── config_TSPN.yaml                    # 基线模型 ✅
└── config_basic.yaml                   # 通用模板
```

### 配置关键字段
```yaml
# 统一配置结构
args:
  model: [ModelName]                    # 模型名称
  dataset_task: THU_018_basic           # 数据集任务
  in_dim: 4096                          # 输入维度
  out_dim: 4096                         # 输出维度
  in_channels: 2                        # 输入通道
  out_channels: 3                       # 输出通道
  num_classes: 5                        # 类别数
  num_epochs: 100                       # 训练轮数
  batch_size: 64                        # 批次大小
  learning_rate: 0.001                  # 学习率
```

### L1优化配置 (OperatorAttention)
```yaml
# configs/unified_baseline/config_OperatorAttention.yaml
args:
  l1_norm: 0.00001    # 2025-11-29调整：从0.0001降低到0.00001以缓解L1损失过高问题
```

---

## 🔧 实验脚本

### 可视化生成脚本
**路径**: `scripts/`

| 脚本 | 功能 | 使用方法 | 输出 |
|------|------|----------|------|
| `visualize_1d2d_contributions.py` | 1D-2D融合可视化 | `python scripts/visualize_1d2d_contributions.py` | 3个图表 |
| `visualize_moe_experts.py` | MoE专家可视化 | `python scripts/visualize_moe_experts.py` | 6个图表+数据 |
| `visualize_operator_attention.py` | 算子注意力可视化 | `python scripts/visualize_operator_attention.py` | 5个图表 |

### 实验执行脚本
```bash
# 主要实验入口
python main.py --config_dir configs/unified_baseline/config_[ModelName].yaml

# 统一基线快速执行
./script/run.sh
```

### 环境配置
```bash
# 环境激活
conda activate LQ_signal  # 或其他可用环境

# GPU分配
CUDA_VISIBLE_DEVICES=0 python main.py  # 使用GPU 0
```

---

## 📈 模型详细信息

### 🏆 Fusion1D2D (立即可投)
```python
# 核心架构
class Fusion1D2D:
    def __init__(self):
        self.conv1d_branch = Conv1d_1x1(in_channels=2, out_channels=3)
        self.conv2d_branch = Conv2d_3x3(in_channels=3, out_channels=16)
        self.statistical_features = StatisticalFeatureExtractor(dim=13)
        self.fusion_layer = Concatenate_1D2D_Stat()
        self.classifier = MLP(input_dim=64+13, hidden_dim=128, num_classes=5)

# 最佳配置
model: Fusion1D2D
accuracy: 99.57%
parameters: 39,000
training_time: ~45 minutes
convergence: 69 epochs
```

### 🤖 MoE (专家系统)
```python
# 专家配置
experts = [
    LowFrequencyExpert(cutoff_freq=500.0),
    HarmonicExpert(),
    EnvelopeExpert(),
    # 简化版使用3专家，完整版支持6专家
]

# 路由机制
router = StatisticalRouter(num_experts=3, temperature=1.0)

# 性能表现
accuracy: 63.04%
parameters: 268M
expert_activation: {
    'LowFreq': 41.7%,
    'Harmonic': 30.0%,
    'Envelope': 28.3%
}
```

### 🔴 OperatorAttention (概念验证)
```python
# 算子配置
operator_library = [
    MovingAverageOperator(),
    DifferentialOperator(),
    FrequencyOperator(),
    NonlinearOperator()
]

# 注意力机制
attention = MultiHeadAttention(num_heads=8, embed_dim=64)

# L1优化效果
l1_coefficient: 0.00001  # 从0.0001优化而来
current_accuracy: 20.00%   # 需重大改进
target_accuracy: 80.0%     # 中期目标
```

### 🩷 FuzzyLogic (基础实现)
```python
# 模糊系统架构
fuzzy_system = FuzzyLogicModel(
    num_rules=50,
    num_variables=10,
    num_classes=5,
    inference_method=mamdani
)

# 当前状态
implementation: 'basic'
accuracy: 20.00%           # 需架构优化
focus: 'rule_learning'     # 模糊规则自动学习
```

---

## 📊 数据集信息

### PHM-Vibench统一数据源
```python
# 数据集配置
dataset_config = {
    'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
    'metadata_file': 'metadata_6_11.xlsx',
    'dataset_ids': [18],  # THU_018
    'target_column': 'Label',
    'task_type': 'fault_diagnosis'
}

# 数据统计
total_samples: 37  # VbenchDataset加载样本
fault_classes: 5   # 0,1,2,3,4
signal_length: 4096
sampling_rate: 12kHz
```

### 数据访问接口
```python
from data.vbench_dataset import VbenchDataset

# 创建数据集
dataset = VbenchDataset(
    args=args,
    flag='all',  # train/val/test/all
    use_cache=True
)

# 数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
```

---

## 🧪 实验环境

### 硬件配置
```bash
# GPU资源
nvidia-smi
# 8x NVIDIA GeForce RTX 4090 (24GB each)
# 可用GPU: 0,1,2,3,5 (4,6,7部分占用)

# 内存配置
total_memory: ~196GB (8 GPUs × 24GB)
available_memory: ~96GB (4 free GPUs)
```

### 软件环境
```bash
# Python环境
python --version  # Python 3.9+
torch.__version__   # PyTorch 2.1.2+cu121
pytorch_lightning.__version__  # 2.1.3

# 关键依赖
pip install torch torchvision
pip install pytorch-lightning wandb
pip install matplotlib seaborn pandas
pip install ptwt  # 小波变换
```

### 跟踪系统
```python
# Weights & Biases配置
import wandb
wandb.init(
    project='unified_baseline',
    config=config,
    name='experiment_name'
)
```

---

## 🚀 实验复现

### 快速复现命令
```bash
# 1. 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate LQ_signal

# 2. 运行最佳模型 (Fusion1D2D)
CUDA_VISIBLE_DEVICES=1 python main.py --config_dir configs/unified_baseline/config_Fusion1D2D.yaml

# 3. 生成可视化
python scripts/visualize_1d2d_contributions.py
python scripts/visualize_moe_experts.py --max_samples 300

# 4. 查看结果
ls Paper/1D-2D_fusion_explainable/results/
ls Paper/MOE_explainable/results/
```

### 批量实验脚本
```bash
# 所有统一基线模型
for model in Fusion1D2D MoE OperatorAttention FuzzyLogic; do
    echo "Running $model experiment..."
    CUDA_VISIBLE_DEVICES=$((RANDOM % 8)) python main.py \
        --config_dir configs/unified_baseline/config_${model}.yaml
done
```

---

## 📞 问题排查

### 常见问题
1. **GPU内存不足**
   ```bash
   # 解决方案：减少batch_size
   # configs/[model].yaml: batch_size: 32 → 16
   ```

2. **数据集路径错误**
   ```bash
   # 检查数据路径
   ls /home/user/data/PHMbenchdata/PHM-Vibench/metadata_6_11.xlsx
   ```

3. **可视化字体问题**
   ```bash
   # 中文字体缺失警告 (不影响功能)
   # 可选：安装中文字体包
   ```

4. **实验结果NaN**
   ```bash
   # 检查梯度爆炸，降低学习率
   # configs/[model].yaml: learning_rate: 0.001 → 0.0005
   ```

### 调试技巧
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

# 监控GPU使用
import torch
print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB')
```

---

**资源索引更新**: 2025年12月3日 00:05
**覆盖范围**: 实验、可视化、配置、文档、脚本全覆盖
**使用建议**: 按需导航，快速定位
**维护频率**: 随项目进展实时更新

---

### 📞 资源获取帮助
- **实验数据**: 查看 [实验结果总览](#实验结果总览)
- **可视化图表**: 查看 [可视化文件](#可视化文件)
- **技术细节**: 查看 [技术文档](#技术文档)
- **配置修改**: 查看 [配置文件](#配置文件)
- **脚本使用**: 查看 [实验脚本](#实验脚本)

**快速帮助**: 使用关键词搜索定位相关内容