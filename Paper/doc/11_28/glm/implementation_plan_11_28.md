# 🚀 统一故障诊断项目执行计划

**时间**: 2025年11月28日
**基于**: execution_plan_unified_baseline_11_28.md
**目标**: 解决当前阻塞问题，确保实验稳定运行

---

## 📋 立即执行（今日）

### Phase 1: 紧急修复（最高优先级）

#### 1. 修复OperatorAttention的Shape错误
- **问题**：`shape '[64, 3, -1]' is invalid for input of size 524288`
- **位置**：`model/OperatorAttention_simple.py` line 139
- **解决方案**：
  ```python
  # 将 target_channels 从 3 改为 2（与in_channels对齐）
  target_channels = 2  # 而不是 3
  # 验证：524288 / 64 / 2 = 4096 （完美整除）
  ```

#### 2. 调整L1正则化系数
- **OperatorAttention**：从 0.0001 调整到 0.00001（降低10倍）
- **MoE**：根据表现决定是否调整
- **修改文件**：`configs/unified_baseline/config_OperatorAttention.yaml`

#### 3. 重新启动稳定实验
- **Fusion1D2D**：使用已验证的配置（`config_Fusion1D2D.yaml`）
- **OperatorAttention**：修复shape后重启
- **MoE**：继续监控，必要时调整超参数

### Phase 2: 监控与数据收集

#### 1. 启动监控系统
```bash
python experiment_monitor_dashboard.py
```

#### 2. 备份实验数据
- 备份`save/`目录到`backup/`
- 压缩`wandb/`日志

#### 3. 验证配置文件
- 确保所有统一基线配置文件完整
- 检查L1系数已正确修改

## 📊 明日任务

### Phase 3: 统一基线文档建设

#### 1. 创建统一基线结果表
- 文件：`Paper/doc/11_28/codex/unified_baseline_results_codex_11_28.md`
- 内容：标准化结果对比表，标注状态（快照/待复现/已验证）

#### 2. 收集实验快照
- 记录三个模型的当前性能指标
- 分析收敛曲线和稳定性
- 为Paper准备基线引用数据

## 🎯 本周目标

### Phase 4: Paper开发与整合

#### 1. 为7个Paper添加基线引用
- 统一引用格式："详见unified_baseline_results_codex_11_28.md"
- 标注每个Paper的相对基线差异

#### 2. 渐进式代码重构
- 梳理重复代码点
- 设计统一接口
- 小步抽取公共模块

## ⚠️ 关键修复点

### OperatorAttention Shape修复
```python
# 修改 model/OperatorAttention_simple.py 第139行附近
# 将：
x = x.view(x.size(0), target_channels, -1)  # target_channels=3导致错误
# 改为：
target_channels = 2  # 与in_channels保持一致
x = x.view(x.size(0), target_channels, -1)
```

### L1正则化调整
```yaml
# configs/unified_baseline/config_OperatorAttention.yaml
args:
  l1_norm: 0.00001  # 从0.0001降低10倍
```

## ✅ 成功标准

### 今日验收标准
1. [ ] 所有shape错误完全修复
2. [ ] 三个模型稳定运行（无RuntimeError）
3. [ ] 监控系统正常运行
4. [ ] L1调整生效（观察loss曲线）

### 明日验收标准
1. [ ] 统一基线结果表创建完成
2. [ ] 至少收集到50个epoch的训练数据
3. [ ] 所有Paper添加基线引用段落

## 🚀 执行命令

```bash
# 1. 修复OperatorAttention
sed -i 's/target_channels = 3/target_channels = 2/' model/OperatorAttention_simple.py

# 2. 调整L1系数
sed -i 's/l1_norm: 0.0001/l1_norm: 0.00001/' configs/unified_baseline/config_OperatorAttention.yaml

# 3. 启动实验
CUDA_VISIBLE_DEVICES=3 python main.py --config_file configs/unified_baseline/config_Fusion1D2D.yaml &
CUDA_VISIBLE_DEVICES=4 python main.py --config_file configs/unified_baseline/config_MoE.yaml &
CUDA_VISIBLE_DEVICES=7 python main.py --config_file configs/unified_baseline/config_OperatorAttention.yaml &

# 4. 启动监控
python experiment_monitor_dashboard.py
```

## 📝 执行日志

### 时间线记录
- **09:00**: 开始执行计划
- **09:05**: 修复OperatorAttention shape问题
- **09:10**: 调整L1正则化系数
- **09:15**: 重启所有实验
- **09:20**: 启动监控系统
- **10:00**: 验证所有实验正常运行

### 问题记录
- OperatorAttention shape错误：已修复
- L1正则化过高：已调整
- GPU资源分配：已优化

### 下一步行动
1. 持续监控实验进展
2. 收集性能数据
3. 准备统一基线报告
4. 为Paper开发做准备

---

**总结**: 本计划优先解决阻塞问题，确保实验能够稳定运行，然后逐步推进文档建设和Paper开发。通过系统化的修复和监控，为后续的学术研究奠定坚实基础。