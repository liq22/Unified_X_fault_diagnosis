# Vbench数据集集成计划

## 项目概述
设计一个面向发论文的快捷Vbench数据集的dataset和dataloader，完全兼容现有配置文件系统，支持故障诊断、RUL预测和异常检测三种任务类型。

## 数据集信息
- 数据路径: `/home/user/data/PHMbenchdata/PHM-Vibench`
- 元数据文件: `metadata_6_11.xlsx`
- 数据格式: HDF5 (data.h5) + Excel (metadata.xlsx)
- 核心字段: Id, Label, RUL_label, Domain_id, Sample_rate, Sample_length, Channel等

## Vbench Data Components Guide

This guide provides a clear overview of the core data files used in Vbench and how to access them.

### Core Components

The dataset is organized into two main files, linked by a common `Id`. An optional `corpus.xlsx` may be provided in future versions.

1.  **`metadata.xlsx` (Excel File)**
    - Purpose: The central index of the dataset. It contains all descriptive information, labels, and parameters for each data sample.
    - Primary Key: The `Id` column uniquely identifies each sample and links the three files.

2.  **`data.h5` (HDF5 File)**
    - Purpose: Stores the raw time-series signal data.
    - Access: Data is retrieved using the `Id` from the metadata file as the key.
    - Shape: The data for each `Id` is a 2D array of shape `(L, C)`, where `L` (Sample_lenth) and `C` (Channel) are specified in `metadata.xlsx`.

Optional:  **`corpus.xlsx` (Excel File)**
    - Purpose: Supplementary text descriptions and natural language annotations.
    - Availability: Not included in the current version; plan for graceful opt-in when present.

### `metadata.xlsx`: Column Descriptions

The first row of the metadata file contains the following headers:

- `Id`: (Primary Key) Unique identifier for the sample.
- `Dataset_id`: Source dataset identifier.
- `Name`: Human-readable name.
- `Description`: Brief description of the sample.
- `TYPE`: Type of data (e.g., vibration, acoustic).
- `File`: Source file name.
- `Visiable`: Visibility or usage flag.
- `Label`: The primary fault class or label.
- `Label_Description`: Textual description of the `Label`.
- `Fault_level`: Severity or stage of the fault.
- `RUL_label`: Remaining Useful Life value.
- `RUL_label_description`: Description for the RUL value.
- `Domain_id`: Identifier for the operational condition.
- `Domain_description`: Textual description of the domain.
- `Sample_rate`: Signal sampling rate (Hz).
- `Sample_lenth (L)`: Number of data points in the sample.
- `Channel (C)`: Number of channels in the sample.
- `Fault_Diagnosis`: Flag for fault diagnosis task suitability.
- `Anomaly_Detection`: Flag for anomaly detection task suitability.
- `Remaining_Life`: Flag for RUL prediction task suitability.

## Supported Industrial Datasets

### Major Datasets
- RM_001_CWRU: Case Western Reserve University bearing data
- RM_002_XJTU: Xi'an Jiaotong University bearing data
- RM_003_FEMTO: FEMTO bearing degradation data
- RM_006_THU: Tsinghua University bearing data
- RM_026_HUST23: HUST bearing dataset
- And 25+ more industrial datasets

## 1. 文件结构设计

### 1.1 新增文件
```
data/
├── vbench_dataset.py          # Vbench专用Dataset类
├── vbench_utils.py           # Vbench数据处理工具
└── data_provider.py          # 更新添加Vbench支持

configs/vbench/
├── config_vbench_basic.yaml     # 基础配置模板
├── RM_001_CWRU/                 # 各数据集子目录
│   ├── config_TSPN.yaml
│   └── config_TFON.yaml
├── RM_002_XJTU/
│   └── ...
└── ...

scripts/
└── run_vbench_experiments.sh   # Vbench实验脚本
```

## 2. 核心功能设计

### 2.1 VbenchDataset类
```python
class VbenchDataset(Dataset):
    """Vbench数据集加载器，支持多任务和域自适应"""

    支持功能：
    - HDF5数据文件读取
    - Excel元数据解析
    - 滑动窗口采样
    - 多任务标签（分类/回归/异常检测）
    - 留一法域自适应
    - 多通道融合策略
```

### 2.2 任务类型支持
1. **故障诊断 (Fault Diagnosis)**
   - 使用 `Label` 字段作为分类标签
   - 支持多分类任务
   - 自动处理类别不平衡

<!-- 2. **剩余寿命预测 (RUL Prediction)**
   - 使用 `RUL_label` 字段作为回归目标
   - 支持时序预测
   - 可配置预测窗口

3. **异常检测 (Anomaly Detection)**
   - 基于 `Anomaly_Detection` 标志
   - 支持无监督和监督学习
   - 可配置异常类型 -->

### 2.3 域自适应策略（留一法）
- 将其中一个Domain_id作为测试集
- 其余Domain_id作为训练集
- 支持交叉验证评估
- 可指定特定domain作为目标域, 默认最后一个, 不同

### 2.4 数据处理功能

#### 1. 滑动窗口采样
- window_length: 窗口长度（如4096）
- stride: 滑动步长（如1024）
- padding_mode: 填充方式

#### 2. 智能采样策略
为了高效控制数据集规模，实现分层采样机制：

**核心规则**：
- **构造ID池**：为每个类别构建固定大小的ID池（默认50个ID）
  - 若该类真实ID数 ≥ 50：等概率无放回抽取50个ID
  - 若该类真实ID数 < 50：循环重复这些ID直到池长=50
  - 可通过配置`ids_cap`修改默认上限

- **样本抽取**：
  - 从ID池中等概率抽取1个ID（即1个数据文件）
  - 在该文件上执行随机滑窗得到m个样本
  - m = ceil(target_per_class / ids_pool_size)
  - 重复直到达到每类目标样本数

**算法流程**：
```
1. 按Label分组：获得每类的所有ID列表
2. 构建ID池：每类最多保留ids_cap个ID
3. 计算采样密度：samples_per_id = ceil(target_per_class / ids_cap)
4. 循环采样：
   while 样本数 < target_per_class:
     - 随机选ID
     - 提取samples_per_id个窗口
```

**优势**：
- **规模可控**：精确控制每类样本数量
- **多样性保证**：ID池机制确保样本多样性
- **内存友好**：避免一次性加载所有数据
- **可重现性**：固定随机种子确保结果一致


## 3. 配置文件设计

### 3.1 基础配置模板 (config_vbench_basic.yaml)
```yaml
############### model_config ####################
signal_processing_configs:
  layer1: ['I', 'WF', 'HT', 'FFT']
  layer2: ['I', 'WF', 'HT', 'FFT']
  layer3: ['I', 'WF', 'HT', 'FFT']
  layer4: ['I', 'WF', 'HT', 'FFT']

feature_extractor_configs: ['Mean', 'Std', 'Var', 'Entropy', 'Max', 'Min']

############### Vbench配置 ####################
vbench_config:
  # 数据路径
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata_6_11.xlsx"
  data_file: "data.h5"
  # corpus_file: "corpus.xlsx"  # 当前版本缺省，无则留空或删除此项



  # 数据集筛选
  dataset_ids: ["RM_001_CWRU", "RM_002_XJTU"]  # 可指定特定数据集

  # 智能采样配置
  sampling_config:
    # 采样方法：smart（智能分层）、random（随机）、balanced（平衡）
    method: "smart"

    # ID池配置
    ids_cap: 50                # ID池大小上限，默认50
    ensure_unique_ids: true      # 确保ID的唯一性（当数量足够时）

    # 样本数量控制
    target_per_class: 1000      # 每类目标样本数
    min_samples_per_id: 1       # 每个ID最少提取样本数

    # 滑窗参数
    window_strategy: "random"   # random/sequential/stratified
    windows_per_file: null       # 每个文件提取的窗口数（null表示自动计算）

    # 重采样策略（当ID不足时）
    resample_strategy: "repeat"  # repeat/circular/oversample

  # 窗口配置
  window_config:
    window_length: 4096
    stride: 1024
    padding_mode: "reflect"
    min_window_ratio: 0.5

  # 域自适应（留一法）
  domain_config:
    leave_one_domain_out: true # 自动选取最后1个domain
    



  # 数据分割
  split_config:
    train_ratio: 0.7
    val_ratio: 0.1
    test_ratio: 0.2
    stratify_by: "Label"  # Label/Domain_id/Fault_level

  # 数据增强
  augmentation:
    enable: false
    noise_std: 0.01
    time_scale_range: [0.9, 1.1]
    amp_scale_range: [0.9, 1.1]

############### 通用配置 ####################
args:
  device: cuda
  model: TSPN
  skip_connection: true
  num_classes: 5  # 根据实际数据自动调整
  in_dim: 4096
  out_dim: 4096
  in_channels: 2
  out_channels: 3
  scale: 4

  # 超参数
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 300
  weight_decay: 0.0001
  num_workers: 8
  seed: 17

  # 训练配置
  monitor: 'val_loss'
  patience: 200
  l1_norm: 0.01
  pruning: None
  snr: 1
```

### 3.2 任务特定配置

#### 故障诊断配置 (config_vbench_diagnosis.yaml)
```yaml
# 继承基础配置，覆盖任务特定部分
defaults:
  - config_vbench_basic

vbench_config:
  task_type: "fault_diagnosis"
  target_column: "Label"
  split_config:
    stratify_by: "Label"

args:
  num_classes: null  # 自动从数据中获取
  model: TSPN
```

#### RUL预测配置 (config_vbench_rul.yaml)
```yaml
defaults:
  - config_vbench_basic

vbench_config:
  task_type: "rul_prediction"
  target_column: "RUL_label"

args:
  model: TFON  # 时频模型更适合RUL预测
  loss_function: "mse"
  metrics: ["mse", "mae", "r2"]
```

#### 异常检测配置 (config_vbench_anomaly.yaml)
```yaml
defaults:
  - config_vbench_basic

vbench_config:
  task_type: "anomaly_detection"
  target_column: "Anomaly_Detection"

args:
  num_classes: 2  # 正常/异常
  model: NNSPN  # 注意力机制适合异常检测
```

## 4. 实施步骤

### Phase 1: 核心数据加载器（第1-2天）
1. 创建 `data/vbench_utils.py`
   - HDF5数据读取工具
   - Excel元数据解析工具
   - 留一法域分割函数
   - 滑动窗口采样函数

2. 创建 `data/vbench_dataset.py`
   - VbenchDataset基础类
   - 实现三种任务类型支持
   - 多通道融合实现
   - 数据增强实现

### Phase 2: 配置系统集成（第3天）
1. 创建配置文件模板
2. 更新 `data_provider.py`
   - 添加Vbench任务映射
   - 实现get_vbench_data函数
3. 更新配置解析器支持嵌套配置

### Phase 3: 测试和优化（第4天）
1. 创建测试脚本
2. 性能优化（内存映射、并行加载）
3. 错误处理和日志完善

### Phase 4: 文档和示例（第5天）
1. 更新README.md Vbench使用说明
2. 创建示例脚本和教程
3. 添加API文档

## 5. 使用示例

### 5.1 基础使用
```bash
# 故障诊断任务
python main.py --config_file configs/vbench/config_vbench_diagnosis.yaml

# RUL预测任务
python main.py --config_file configs/vbench/config_vbench_rul.yaml

# 异常检测任务
python main.py --config_file configs/vbench/config_vbench_anomaly.yaml
```

### 5.2 特定数据集实验
```bash
# CWRU数据集实验
python main.py --config_file configs/vbench/RM_001_CWRU/config_TSPN.yaml

# 跨域泛化实验（留一法）
python main.py --config_file configs/vbench/config_vbench_generalization.yaml
```

## 6. 技术要点

### 6.1 内存优化
- 使用HDF5内存映射避免加载全部数据
- 实现按需加载和缓存机制
- 支持分布式训练

### 6.2 兼容性保证
- 完全兼容现有trainer和模型
- 保持配置文件格式一致性
- 支持现有的数据处理流水线

### 6.3 扩展性设计
- 模块化设计便于添加新任务
- 插件式数据增强策略
- 灵活的域适配接口

## 7. 预期成果

1. **高效的Vbench数据加载器**：支持GB级数据快速加载
2. **灵活的任务配置**：一个数据集支持三种研究任务
3. **强大的域自适应能力**：留一法评估模型泛化性能
4. **完整的示例代码**：开箱即用的配置和脚本

这个方案将使研究人员能够快速在Vbench数据集上进行实验验证，大大提高论文实验效率。

---

## 8. 通用读取与MoE准备（高级扩展功能）

### 8.1 统一数据读取管线（单一Dataset）
- **前提条件**：数据结构统一（metadata.xlsx + data.h5；corpus.xlsx 可选）。
- **标准样本结构**：`{"x": Float32[C,L], "y": target, "meta": {...}`。
- **实现要点**：
  - 仅实现一个 `VbenchDataset`，内部使用 `pandas` 读取 `metadata.xlsx`，使用 `h5py` 读取 `data.h5`（惰性加载/分块加载）
  - `__getitem__(idx)`：通过 `Id` 定位到 HDF5 键，返回 `[C,L]`（注意转置），附带元信息与门控特征。
  - 支持滑动窗口（length/stride）、多通道选择/融合、任务标签派生（Label/RUL/Anomaly）。

示例代码：
```python
class VbenchDataset(Dataset):
    """Vbench统一数据集加载器"""

    def __init__(self, root, task, window, channel, use_corpus=False):
        """
        初始化数据集
        Args:
            root: 数据根目录
            task: 任务类型 (fault_diagnosis/rul_prediction/anomaly_detection)
            window: 窗口配置 (length, stride)
            channel: 通道配置
            use_corpus: 是否使用语料库
        """
        # 读取元数据
        self.meta = pd.read_excel(root/"metadata.xlsx")
        # 打开HDF5文件（惰性加载）
        self.h5 = h5py.File(root/"data.h5", "r")
        self.corpus = None
        if use_corpus and (root/"corpus.xlsx").exists():
            self.corpus = pd.read_excel(root/"corpus.xlsx")
        # 预构建索引、窗口映射...

    def __getitem__(self, i):
        """获取单个样本"""
        row = self.meta.iloc[i]
        # 从HDF5读取原始数据，形状为(L,C)
        x = self.h5[str(row.Id)][...]
        # 转置为(C,L)以符合PyTorch约定
        x = torch.from_numpy(x).T.contiguous()
        # 根据任务类型构建目标标签
        y = self._build_target(row, task)
        # 提取门控特征（用于MoE路由）
        g = self._extract_gate_features(x, row)
        return {"x": x, "y": y, "meta": dict(row), "gate_feat": g}

    def _build_target(self, row, task):
        """根据任务类型构建目标标签"""
        if task == "fault_diagnosis":
            return row.Label
        elif task == "rul_prediction":
            return row.RUL_label
        elif task == "anomaly_detection":
            return row.Anomaly_Detection

    def _extract_gate_features(self, x, row):
        """提取门控特征"""
        # 实现特征提取逻辑
        pass


class SmartSampler:
    """智能采样器，实现分层采样和ID池机制"""

    def __init__(self, metadata, config):
        """
        初始化智能采样器
        Args:
            metadata: 元数据DataFrame
            config: 采样配置字典
        """
        self.metadata = metadata
        self.config = config
        self.ids_cap = config.get('ids_cap', 50)
        self.target_per_class = config.get('target_per_class', 1000)
        self.seed = config.get('seed', 42)

        # 设置随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)

    def build_id_pool(self, label_ids):
        """
        构建ID池
        Args:
            label_ids: 某个类别的所有ID列表
        Returns:
            id_pool: 构建好的ID池
        """
        if len(label_ids) >= self.ids_cap:
            # 等概率无放回抽取
            return random.sample(label_ids, self.ids_cap)
        else:
            # 循环重复直到达到池大小
            pool = []
            while len(pool) < self.ids_cap:
                remaining = self.ids_cap - len(pool)
                pool.extend(label_ids[:min(remaining, len(label_ids))])
            return pool[:self.ids_cap]

    def sample_class(self, label_id_pool, label_name):
        """
        为单个类别采样
        Args:
            label_id_pool: 该类别的ID池
            label_name: 类别名称
        Returns:
            samples: 采样结果列表，每个元素为(id, start_pos, label)
        """
        samples = []
        samples_per_id = max(1,
                          math.ceil(self.target_per_class / len(label_id_pool)))

        while len(samples) < self.target_per_class:
            # 从ID池中等概率抽取一个ID
            selected_id = random.choice(label_id_pool)

            # 加载该ID的数据文件
            data = self._load_h5_data(selected_id)
            data_length = data.shape[0]  # L维度长度

            # 随机滑窗采样
            for _ in range(min(samples_per_id,
                           self.target_per_class - len(samples))):
                # 随机选择起始位置
                max_start = max(0, data_length - self.config['window_length'])
                if max_start > 0:
                    start_pos = random.randint(0, max_start)
                else:
                    start_pos = 0

                samples.append({
                    'id': selected_id,
                    'start_pos': start_pos,
                    'label': label_name,
                    'window_length': self.config['window_length']
                })

        return samples[:self.target_per_class]  # 确保不超过目标数

    def sample_all_classes(self):
        """
        为所有类别采样
        Returns:
            all_samples: 所有类别的采样结果
        """
        # 1. 按Label分组
        class_groups = self.metadata.groupby('Label')

        all_samples = []
        for label_name, group in class_groups:
            # 2. 获取该类别的所有ID
            label_ids = group['Id'].unique().tolist()

            # 3. 构建ID池
            id_pool = self.build_id_pool(label_ids)
            print(f"Label {label_name}: "
                  f"{len(label_ids)} IDs → {len(id_pool)} ID pool")

            # 4. 为该类采样
            samples = self.sample_class(id_pool, label_name)
            all_samples.extend(samples)

        return all_samples

    def _load_h5_data(self, id):
        """加载HDF5数据（惰性加载）"""
        # 实现HDF5数据加载逻辑
        pass


# 使用示例
if __name__ == "__main__":
    # 配置参数
    sampling_config = {
        'ids_cap': 50,              # ID池大小
        'target_per_class': 1000,   # 每类目标样本数
        'window_length': 4096,      # 窗口长度
        'seed': 42                  # 随机种子
    }

    # 读取元数据
    metadata = pd.read_excel("metadata.xlsx")

    # 创建采样器
    sampler = SmartSampler(metadata, sampling_config)

    # 执行采样
    samples = sampler.sample_all_classes()

    print(f"总采样数: {len(samples)}")
    print(f"类别分布: {pd.Series([s['label'] for s in samples]).value_counts()}")
```

### 8.2 MoE（混合专家）门控特征与路由
- **轻量门控特征**：
  - 时域特征：`rms`（均方根）、`kurtosis`（峰度）、`skewness`（偏度）、`crest_factor`（峰值因子）
  - 频域特征：`spec_centroid`（频谱重心）、`bandpower`（频带功率：低/中/高）、`spectral_entropy`（频谱熵）

- **归一化处理**：
  - 使用 `z-score` 按通道归一化
  - 可缓存至 `data/.cache/`（按 `dataset_id/hash` 存储）

- **路由策略**：
  - **无监督路由**：门控网络 `g(x or gate_feat) -> experts_logits`
  - **半监督路由**：若元数据中含传感器/频段/工况信息，作为弱标签监督门控

- **DataLoader `collate_fn`**：
  - 确保 `gate_feat` 与 `meta` 正确打包
  - 保留元信息用于专家激活调试

### 8.3 采样与混合策略（为MoE稳定训练）
- **采样器类型**：
  - `ClassBalancedSampler`：类别均衡采样
  - `DomainBalancedSampler`：域均衡采样
  - `CurriculumSampler`：按频带/信噪比递进采样

- **多数据集混合**：
  - 跨 `Vbench + 现有THU/HUST` 数据集混合
  - 支持 `mix.weights` 加权策略或 `balanced` 均衡策略

### 8.4 配置扩展（向后兼容）
```yaml
# Vbench数据配置
data:
  root: /data/PHM-Vibench
  subset:
    dataset_ids: [RM_001_CWRU, RM_002_XJTU]  # 指定使用的数据集
  window:
    length: 4096
    stride: 1024

# MoE门控配置
gating:
  # 门控特征列表
  features: [rms, kurtosis, spec_centroid, bandpower_low, spectral_entropy]
  # 归一化方式
  normalize: zscore
  # 是否使用监督
  supervise: false
```

### 8.5 实施里程碑（版本2.0）
1. **第一阶段**：实现 `data/vbench_dataset.py`（惰性读取、窗口索引、任务派生）
2. **第二阶段**：提供 `fast_gate_features` 与 `collate_fn`；打通 `trainer_set` 承接 `gate_feat`
3. **第三阶段**：加入 `BalancedSampler` 与 `DomainBalancedSampler`；支持按 `Dataset_id/Domain_id/Label` 均衡采样
4. **第四阶段**：文档与基准：门控特征可视化、专家路由占比、跨域稳健性报告

---

## 9. 性能基准与技术细节

### 9.1 性能目标与基准

#### 数据加载性能
- **单文件加载时间**：< 1秒（10万样本）
- **内存占用**：< 2GB（惰性加载模式下）
- **并行加载速度**：支持32个worker，吞吐量 > 10000 samples/second

#### 训练性能
- **GPU利用率**：> 90%（RTX 4090，batch_size=64）
- **训练速度**：< 30分钟/epoch（Vbench完整数据集）
- **收敛速度**：100 epoch内达到95%准确率

#### 域自适应性能
- **留一法准确率**：CWRU数据集上 > 95%
- **跨数据集泛化**：CWRU → XJTU 迁移准确率 > 85%
- **小样本学习**：K-shot（K=64）准确率 > 90%

#### 智能采样效率基准
- **ID池构建时间**：< 0.1秒（每类1000个ID）
- **采样执行时间**：< 5秒（生成10000个样本）
- **内存占用增量**：< 100MB（采样索引存储）
- **采样均衡性**：类别间样本数差异 < 1%
- **ID覆盖度**：50个ID池覆盖 > 95%的数据多样性

### 9.2 HDF5数据访问优化

#### 内存映射策略
```python
# 优化前：全部加载到内存
data = h5_file[key][...]  # 消耗大量内存

# 优化后：内存映射访问
import h5py
f = h5py.File('data.h5', 'r')
dset = f[key]
# 按需读取
sample = dset[start:end, channels]  # 仅读取所需数据
```

#### 数据缓存机制
- **LRU缓存**：最近访问的数据块保留在内存中
- **预取策略**：异步预加载下一个batch的数据
- **压缩存储**：启用HDF5压缩减少磁盘I/O

### 9.3 数学公式与算法细节

#### 滑动窗口采样
对于长度为N的信号，窗口长度为W，步长为S：

```
窗口数量 = floor((N - W) / S) + 1
窗口_i = signal[i*S : i*S + W]
```

#### 多通道融合策略

1. **拼接融合（Concat）**：
   ```
   X_fused = concat(X_1, X_2, ..., X_C)
   shape: (C, L) → (C, L)
   ```

2. **平均融合（Mean）**：
   ```
   X_fused = (1/C) * Σ_{i=1}^{C} X_i
   shape: (C, L) → (1, L)
   ```

3. **注意力融合（Attention）**：
   ```
   α_i = softmax(W_a · X_i)
   X_fused = Σ_{i=1}^{C} α_i · X_i
   ```

#### 门控特征计算

1. **均方根（RMS）**：
   ```
   RMS = √(1/N * Σ_{i=1}^{N} x_i²)
   ```

2. **峰度（Kurtosis）**：
   ```
   Kurtosis = (1/N * Σ_{i=1}^{N} (x_i - μ)⁴) / σ⁴
   ```

3. **频谱重心（Spectral Centroid）**：
   ```
   SC = Σ_{k} k·|X(k)| / Σ_{k} |X(k)|
   ```

### 9.4 内存优化技术

#### 数据类型优化
- **float16训练**：在保持精度的同时减少50%显存占用
- **梯度累积**：大batch_size模拟
- **模型并行**：多GPU负载均衡

#### 批处理优化
```python
# 动态batch_size
batch_size = compute_optimal_batch_size(
    model, available_memory, input_size
)

# 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```



## 10. 常见问题与解决方案（FAQ）



### Q2: 内存不足怎么办？
**A**: 多种优化策略：
1. 使用HDF5内存映射（默认启用）
2. 减少num_workers数量
3. 使用更小的batch_size
4. 启用梯度累积

### Q3: 如何添加新的数据集？
**A**: 简单两步：
1. 将数据转换为HDF5+Excel格式
2. 在metadata.xlsx中添加相应条目
3. 更新配置文件的dataset_ids列表



### Q5: 如何进行数据增强？
**A**: 配置灵活的增强策略：
```yaml
augmentation:
  enable: true
  methods:
    - type: "noise"
      params: {std: 0.01}
    - type: "time_warp"
      params: {scale: 0.1}
    - type: "amp_scale"
      params: {range: [0.9, 1.1]}
```

### Q6: 如何监控训练过程？
**A**: 多维度监控：
- **WandB集成**：自动记录loss、accuracy等指标
- **梯度监控**：检查梯度范数，防止梯度消失/爆炸
- **数据分布**：监控batch内数据分布变化

### Q7: 如何调试模型？
**A**: 调试工具：
1. **可视化**：信号的时频图、注意力权重
2. **特征分析**：t-SNE/UMAP可视化特征空间
3. **错误分析**：混淆矩阵、错误样本可视化

### Q8: 如何提高模型泛化能力？
**A**: 泛化策略：
1. **域对抗训练**：添加域分类器
2. **元学习**：MAML原型网络
3. **数据增强**：多样化的增强策略
4. **正则化**：Dropout、BatchNorm、Weight Decay

### Q9: 如何部署到生产环境？
**A**: 部署方案：
1. **模型导出**：ONNX格式
2. **服务封装**：Flask/FastAPI
3. **容器化**：Docker
4. **批处理**：支持实时和批量推理

### Q10: 如何贡献代码？
**A**: 贡献流程：
1. Fork项目
2. 创建feature分支
3. 添加测试用例
4. 提交Pull Request
5. 代码审查通过后合并

---

## 11. 总结与展望

### 11.1 项目价值
本方案为故障诊断领域提供了一个**统一、高效、可扩展**的数据处理框架，能够：
- 大幅提升实验效率，减少重复工作
- 支持多种研究任务，促进方法创新
- 保证实验可复现性，增强成果可信度

### 11.2 未来扩展方向
1. **更多数据集支持**：持续集成新的工业数据集
2. **自动化机器学习**：AutoML集成，自动超参数搜索
3. **联邦学习**：支持分布式、隐私保护的联合训练
4. **在线学习**：支持流式数据的增量学习

### 11.3 技术演进路径
- **短期目标**（3个月）：完成核心功能，支持主要数据集
- **中期目标**（6个月）：增加MoE支持，提升模型性能
- **长期目标**（1年）：形成完整的工业AI生态系统
