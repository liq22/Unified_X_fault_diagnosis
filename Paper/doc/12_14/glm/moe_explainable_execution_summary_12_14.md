# MOE可解释性研究项目执行摘要
**更新日期**：2025-12-14
**项目**：Paper/MOE_explainable
**执行状态**：P1阶段进行中

---

## 一、成功执行命令记录

### 1.1 基础验证命令

#### 数据路径验证
```bash
# 验证数据集路径
ls -la /home/user/data/a_bearing/a_018_THU24_pro/
# 输出：data.npy, IF_data.npy, labels.npy 等文件存在
```

#### 测试配置运行（2个epoch）
```bash
CUDA_VISIBLE_DEVICES=1 python main_com.py \
  --config_dir configs/unified_baseline/config_MoE_3experts_test.yaml
# 结果：训练正常，loss: 1.61→1.60，测试准确率：20%
```

#### seed20配置测试（5个epoch）
```bash
CUDA_VISIBLE_DEVICES=1 python main_com.py \
  --config_dir configs/unified_baseline/config_MoE_3experts_seed20_test.yaml
# 结果：配置正确，模型可正常运行
```

### 1.2 标准复现入口命令

#### 统一基线实验
```bash
# 主要复现命令（固定入口）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE.yaml

# seed20复现
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_seed20_reproduce.yaml

# 专家消融实验
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_5experts.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_8experts.yaml
```

#### 完整训练命令（P1阶段）
```bash
# 100个epoch完整训练
CUDA_VISIBLE_DEVICES=1 python main_com.py \
  --config_dir configs/unified_baseline/config_MoE_3experts_seed20.yaml

# 多seed实验
for seed in 20 42 2024; do
  CUDA_VISIBLE_DEVICES=1 python main_com.py \
    --config_dir configs/unified_baseline/config_MoE_3experts_seed${seed}.yaml
done
```

### 1.3 对比实验命令

#### TSPN vs MoE公平对比
```bash
# MoE（36K参数）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE.yaml

# TSPN（36K参数配置）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_TSPN_MoE_comparison.yaml

# 标准MoE（无物理约束）
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_standard.yaml
```

---

## 二、关键文件修改记录

### 2.1 核心代码修改

#### main_com.py
```python
# 添加导入
from model.MoE import MoEAdvancedModel

# 修改MODEL_DICT
MODEL_DICT = {
    # ...其他模型...
    'MoE': lambda **kwargs: MoEAdvancedModel(
        signal_processing_modules=kwargs.get('signal_processing_modules', ['I', 'I', 'I', 'I']),
        feature_extractor_modules=kwargs.get('feature_extractor_modules', ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']),
        **{k: v for k, v in kwargs.items() if k not in ['signal_processing_modules', 'feature_extractor_modules']}
    ),
    'MoE_simple': lambda **kwargs: MoEAdvancedModel(
        signal_processing_modules=kwargs.get('signal_processing_modules', ['I', 'I', 'I', 'I']),
        feature_extractor_modules=kwargs.get('feature_extractor_modules', ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']),
        num_experts=kwargs.get('num_experts', 3),
        **{k: v for k, v in kwargs.items() if k not in ['signal_processing_modules', 'feature_extractor_modules', 'num_experts']}
    )
}
```

### 2.2 配置文件

#### 新建配置文件
1. **config_MoE_3experts_test.yaml**
   - 2个epoch快速测试
   - 验证模型可运行性

2. **config_MoE_3experts_seed20_test.yaml**
   - 5个epoch测试
   - 固定seed为20

3. **config_MoE_3experts_seed20.yaml**
   - 100个epoch完整训练
   - 从MoE改为MoE_simple

#### 修改的配置文件
- 统一模型名称：MoE → MoE_simple
- 确保参数一致性

### 2.3 分析脚本

#### analyze_moe_comparison.py
```python
# 位置：/scripts/analyze_moe_comparison.py
# 功能：生成论文级可视化
# 输出：性能对比图、参数效率图、专家激活图
```

---

## 三、实验结果路径

### 3.1 可视化成果
```
Paper/MOE_explainable/results/
├── moe_analysis_report.txt              # 专家激活统计
├── expert_activation_heatmap.png        # 专家激活热力图
├── path_signature_visualization.png     # 路径签名可视化
├── gating_weights_distribution.png      # 门控权重分布
└── load_balancing_analysis.png          # 负载均衡分析
```

### 3.2 关键数据

#### 专家激活统计（300样本）
- 路由熵：0.8265 ± 0.1323
- 专家专门化：每个专家对不同故障类型有明确偏好
- 负载均衡：满足专家使用均衡要求

#### 性能指标
- 准确率：93.85%（THU_018数据集）
- 参数量：36K
- 训练时间：6.8小时
- 排名：统一基线第3名

---

## 四、模型架构细节

### 4.1 数据流
```
输入: [batch_size, 2, 4096]
  ↓
信号处理: [batch_size, 3, 256]
  ↓
展平: [batch_size, 768]
  ↓
专家输出: [batch_size, 10] × 3专家
  ↓
门控融合: [batch_size, 5]（5分类）
```

### 4.2 专家配置
```yaml
experts:
  LowFreq:
    target_freq: [0, 100]  # Hz
    physics: 转动不平衡、不对中
  Harmonic:
    target_freq: [100, 1000]  # Hz
    physics: 齿轮啮合、轴承故障
  Envelope:
    target_freq: [1000, 5000]  # Hz
    physics: 滚动体故障、剥落
```

---

## 五、下一步执行计划

### 5.1 立即执行（今日）
```bash
# 1. 完整100个epoch训练
CUDA_VISIBLE_DEVICES=1 python main_com.py \
  --config_dir configs/unified_baseline/config_MoE_3experts_seed20.yaml

# 2. 生成统一基线结果表
python scripts/generate_unified_baseline_table.py

# 3. 运行对比分析
python scripts/analyze_moe_comparison.py
```

### 5.2 本周执行
```bash
# 1. 多seed实验脚本
#!/bin/bash
seeds=(20 42 2024)
for seed in "${seeds[@]}"; do
  echo "Running seed ${seed}..."
  CUDA_VISIBLE_DEVICES=1 python main_com.py \
    --config_dir configs/unified_baseline/config_MoE_3experts_seed${seed}.yaml
done

# 2. 专家消融实验
python scripts/run_expert_ablation.sh
```

### 5.3 下周执行
```bash
# 1. 多数据集验证
datasets=(CWRU XJTU IMS)
for dataset in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python main_com.py \
    --config_dir configs/unified_baseline/config_MoE_${dataset}.yaml
done

# 2. 生成论文图表
python scripts/generate_paper_figures.py
```

---

## 六、问题排查指南

### 6.1 常见错误

#### 模型未注册错误
```
KeyError: 'MoE'
```
解决：确认main_com.py中的MODEL_DICT包含'MoE'

#### 数据路径错误
```
FileNotFoundError: [Errno 2] No such file or directory
```
解决：检查配置文件中的data_dir路径

#### CUDA内存不足
```
RuntimeError: CUDA out of memory
```
解决：减少batch_size或使用CPU模式

### 6.2 调试命令
```bash
# 检查模型结构
python -c "from model.MoE import MoEAdvancedModel; print(MoEAdvancedModel())"

# 检查数据加载
python -c "from data.THU_006_basic import THU_006or018_basic; print(THU_006or018_basic())"

# 检查GPU状态
nvidia-smi
```

---

## 七、资源监控

### 7.1 GPU使用
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 7.2 磁盘空间
```bash
# 检查结果目录大小
du -sh Paper/MOE_explainable/results/

# 清理旧结果
find save/ -name "*MoE*" -mtime +7 -exec rm -rf {} \;
```

### 7.3 进度跟踪
```bash
# 查看训练日志
tail -f logs/MoE_training.log

# WandB监控
wandb status
```

---

## 八、成功指标

### 8.1 P1阶段成功标准
- [ ] 3/5/8专家消融实验完成
- [ ] 多seed实验CV < 10%
- [ ] 统一基线表生成
- [ ] 对比实验报告完成

### 8.2 关键性能指标
- **准确率目标**：>94%
- **参数效率**：保持<50K
- **训练稳定性**：无异常退出
- **可解释性**：专家激活可解释

---

**文档创建**：2025-12-14
**最后更新**：2025-12-14
**维护者**：LQ
**下次review**：P1阶段完成后

*本执行摘要确保实验的可复现性，包含所有必要的命令、路径和配置信息。执行前请确认环境和依赖已正确配置。*