# data 目录说明（中文）

本目录包含 **数据加载与预处理** 相关代码，是所有实验共享的数据层基础设施。当前版本默认围绕 **PHM‑Vibench 振动信号基准库** 组织数据接口。

---

## 1. 数据来源概览：PHM‑Vibench

本项目使用的数据来自上游工程：`PHM‑Vibench`（PHM‑Vibench 振动信号基准数据库），详细说明可参考：

- 上游仓库数据文档：  
  `/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/README.md`

### 关键特性（简要）

- 多个开源轴承/齿轮箱等故障诊断数据集统一整合（CWRU, XJTU, THU, DIRG 等）；  
- 数据统一整理为 HDF5 + Excel 元数据（`metadata_*.xlsx` / `cache*.h5`）；  
- 支持不同转速、负载、故障类型（正常、内圈/外圈/滚动体/复合故障等）；  
- 面向机械故障诊断与预测性维护任务。  

在本仓库中，我们通过 `vbench_dataset.py` + `vbench_utils.py` 对这些数据做了统一封装。

---

## 2. 核心文件与职责

- `vbench_dataset.py`  
  - 提供统一的 `VbenchDataset(Dataset)` 封装，是推荐使用的主数据入口；  
  - 支持根据 `args.vbench_config` 配置，自动加载 HDF5 数据与 Excel 元数据；  
  - 内置训练/验证/测试划分、滑动窗口采样、按类别均衡采样等逻辑。  

- `vbench_utils.py`  
  - VBench 相关辅助函数：  
    - 元数据读取与检查（如 Dataset_id, Label 等字段）；  
    - ID 采样、滑动窗口位置计算；  
    - 数据切分与统计工具。  

- `data_provider.py` / `datasets.py`  
  - 旧版数据接口与数据集定义，部分历史实验还在使用；  
  - 新实验建议逐步迁移到 `VbenchDataset` 体系。  

- `utils.py`  
  - 通用数据工具函数（路径处理、缓存、简单转换等）。  

> 建议：**新方法 / 新 Paper 优先复用 `VbenchDataset` + `vbench_utils`，不要在子目录重复实现 Dataset。**

---

## 3. VbenchDataset 配置说明（`vbench_dataset.py`）

`VbenchDataset` 通过 `args.vbench_config`（或 `args.config['vbench_config']`）配置数据来源与采样策略，常用字段如下：

```yaml
vbench_config:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"  # 数据根目录
  metadata_file: "metadata_6_11.xlsx"                   # 元数据 Excel 文件
  # 可选：备用 HDF5 文件（fallback 使用）
  data_file: "cache.h5"
  dataset_ids: [1]                                      # 使用哪些 Dataset_id
  task_type: "fault_diagnosis"                          # 任务类型
  target_column: "Label"                                # 标签列名

  sampling_config:
    target_per_class: 4096         # 每类目标样本数
    ids_cap: 64                    # 每类 ID 采样上限
    min_samples_per_id: 1          # 每个 ID 至少采样次数
    window_length: 4096            # 滑动窗口长度
    window_strategy: "random"      # "random" / "sequential"
    resample_strategy: "repeat"    # 不足时如何重采样
```

### 初始化与调用示例（伪代码）

```python
from data.vbench_dataset import VbenchDataset

args.vbench_config = {
    "data_dir": "/home/user/data/PHMbenchdata/PHM-Vibench",
    "metadata_file": "metadata_6_11.xlsx",
    "dataset_ids": [6],  # 例如使用 RM_006_THU 数据集
    "task_type": "fault_diagnosis",
    "target_column": "Label",
    "sampling_config": {
        "target_per_class": 4096,
        "window_length": 4096,
        "window_strategy": "random",
    },
}

train_dataset = VbenchDataset(args, flag="train")
val_dataset   = VbenchDataset(args, flag="val")
test_dataset  = VbenchDataset(args, flag="test")
```

内部会自动完成：
- 读取 `metadata_file` 并过滤指定 `dataset_ids`；  
- 过滤掉标签为空的样本；  
- 按类别采样 ID，基于 `window_length` 与 `window_strategy` 做滑动窗口截取；  
- 构建 PyTorch `Dataset` 对象供 `DataLoader` 使用。

---

## 4. 常见使用建议

1. **统一用 VbenchDataset 做数据入口**  
   - 在新的 config 中添加 `vbench_config` 而不是手写 Dataset；  
   - 统一通过 `data_dir + metadata_file + dataset_ids` 控制数据来源。  

2. **保持与上游 VBench README 一致**  
   - 若需要详细了解某个 Dataset_id 对应的数据集（CWRU/XJTU/THU 等），请参考：  
     `/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/README.md`。  

3. **Paper 级别的数据处理逻辑**  
   - 如果某篇论文需要特殊划分/过滤规则，建议在对应 `Paper/<Project>/code` 下实现，  
   - 不要修改 `VbenchDataset` 的通用行为，只通过配置和下游处理实现差异。  

---  

如需扩展新的数据集或任务类型，推荐先在 VBench 仓库中整理元数据和 HDF5 文件，再在本仓库的 `vbench_dataset.py` / `vbench_utils.py` 中增加对应的配置/工具函数。  
