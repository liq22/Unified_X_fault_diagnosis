import numpy as np
import torch
import pandas as pd
import h5py
import random
import math
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split


class VbenchDataset(Dataset):
    """
    Vbench数据集加载器
    支持HDF5数据文件和Excel元数据，实现智能采样和滑动窗口
    """

    def __init__(
        self,
        args,
        flag: str = 'train',
        transform=None,
        use_cache: bool = True
    ):
        """
        初始化Vbench数据集

        Args:
            args: 配置参数对象，包含vbench_config
            flag: 数据集类型 ('train', 'val', 'test')
            transform: 数据变换函数
            use_cache: 是否使用缓存加速
        """
        self.args = args
        self.flag = flag
        self.transform = transform
        self.use_cache = use_cache

        # 提取Vbench配置 - 支持不同层级的配置
        if hasattr(args, 'vbench_config'):
            # 如果有 vbench_config 属性
            self.vb_config = args.vbench_config
        elif hasattr(args, 'config') and 'vbench_config' in args.config:
            # 如果配置在 config.vbench_config 下
            self.vb_config = args.config.get('vbench_config', {})
        else:
            # 创建默认配置
            self.vb_config = {
                'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
                'metadata_file': 'metadata_6_11.xlsx',
                'data_file': 'cache.h5',
                'dataset_ids': [1],  # 默认测试数据集1
                'task_type': 'fault_diagnosis',
                'target_column': 'Label'
            }

        self.sampling_config = self.vb_config.get('sampling_config', {})

        # 数据路径
        # 支持不同的路径格式
        if 'data_dir' in self.vb_config:
            self.data_dir = Path(self.vb_config['data_dir'])
        else:
            # 默认路径
            self.data_dir = Path('/home/user/data/PHMbenchdata/PHM-Vibench')

        if 'metadata_file' in self.vb_config:
            self.metadata_file = self.data_dir / self.vb_config['metadata_file']
        else:
            # 默认文件名
            self.metadata_file = self.data_dir / 'metadata_6_11.xlsx'

        # 移除固定的data_file配置，改为动态选择
        # 保留可选的fallback支持
        self.use_fallback_cache = self.vb_config.get('fallback_cache', False)
        if self.use_fallback_cache and 'data_file' in self.vb_config:
            self.fallback_file = self.data_dir / self.vb_config['data_file']
        else:
            self.fallback_file = None

        # 任务类型
        self.task_type = self.vb_config.get('task_type', 'fault_diagnosis')
        self.target_column = self.vb_config.get('target_column', 'Label')

        # H5文件缓存
        self.h5_cache = {}  # 缓存已打开的H5文件

        # 加载元数据
        print(f"Loading metadata from: {self.metadata_file}")
        self.metadata = pd.read_excel(self.metadata_file)

        # 创建Dataset_id到Name的映射
        self.dataset_mapping = self.metadata[['Dataset_id', 'Name']].drop_duplicates()
        self.dataset_mapping = self.dataset_mapping.set_index('Dataset_id')['Name'].to_dict()

        # 数据集筛选
        if 'dataset_ids' in self.vb_config:
            self.metadata = self.metadata[
                self.metadata['Dataset_id'].isin(self.vb_config['dataset_ids'])
            ].reset_index(drop=True)

        # 过滤掉标签为NaN的样本
        if self.target_column in self.metadata.columns:
            initial_count = len(self.metadata)
            self.metadata = self.metadata.dropna(subset=[self.target_column]).reset_index(drop=True)
            filtered_count = initial_count - len(self.metadata)
            if filtered_count > 0:
                print(f"过滤掉 {filtered_count} 个标签为NaN的样本")

        # 不再打开单一的HDF5文件，改为按需加载
        if self.use_fallback_cache and self.fallback_file and self.fallback_file.exists():
            print(f"Opening fallback HDF5 file: {self.fallback_file}")
            self.h5_file = h5py.File(self.fallback_file, 'r')
        else:
            self.h5_file = None  # 将使用动态H5文件加载

        # 初始化采样器
        self._init_sampler()

        # 生成样本索引
        self._build_sample_indices()

        print(f"Dataset initialized: {len(self.samples)} samples")
        print(f"Class distribution: {self.get_class_distribution()}")

    def _init_sampler(self):
        """初始化智能采样器"""
        # 获取当前数据集的标签列表（根据flag）
        self.train_data, self.val_data, self.test_data = self._split_data()

        if self.flag == 'train':
            self.data_for_sampling = self.train_data
        elif self.flag == 'val':
            self.data_for_sampling = self.val_data
        else:
            self.data_for_sampling = self.test_data

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据配置分割数据

        Returns:
            train_data, val_data, test_data
        """
        split_config = self.vb_config.get('split_config', {})

        # 域自适应（留一法）
        domain_config = self.vb_config.get('domain_config', {})
        if domain_config.get('leave_one_domain_out', False):
            return self._leave_one_domain_out_split(domain_config)

        # 常规分割
        train_ratio = split_config.get('train_ratio', 0.7)
        val_ratio = split_config.get('val_ratio', 0.1)
        test_ratio = 1 - train_ratio - val_ratio

        stratify_by = split_config.get('stratify_by', 'Label')

        # 检查是否可以进行分层采样
        can_stratify_initial = False
        can_stratify_secondary = False

        if stratify_by in self.metadata.columns:
            label_counts = self.metadata[stratify_by].value_counts()
            min_samples_per_class = min(label_counts)
            # 如果最小类别的样本数至少为2，则可以进行分层采样
            can_stratify_initial = min_samples_per_class >= 2

        if can_stratify_initial:
            train_data, temp_data = train_test_split(
                self.metadata,
                test_size=1-train_ratio,
                stratify=self.metadata[stratify_by],
                random_state=getattr(self.args, 'seed', 42)
            )

            # 检查temp_data是否可以继续分层采样
            temp_label_counts = temp_data[stratify_by].value_counts()
            min_temp_samples = min(temp_label_counts)
            can_stratify_secondary = min_temp_samples >= 2

            if can_stratify_secondary:
                val_data, test_data = train_test_split(
                    temp_data,
                    test_size=test_ratio/(test_ratio+val_ratio),
                    stratify=temp_data[stratify_by],
                    random_state=getattr(self.args, 'seed', 42)
                )
            else:
                print(f"警告: temp_data中类别样本不足，使用随机分割")
                val_data, test_data = train_test_split(
                    temp_data,
                    test_size=test_ratio/(test_ratio+val_ratio),
                    random_state=getattr(self.args, 'seed', 42)
                )
        else:
            print(f"警告: 类别样本不足，使用随机分割而非分层采样")
            train_data, temp_data = train_test_split(
                self.metadata,
                test_size=1-train_ratio,
                random_state=getattr(self.args, 'seed', 42)
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio/(test_ratio+val_ratio),
                random_state=getattr(self.args, 'seed', 42)
            )

        return train_data, val_data, test_data

    def _leave_one_domain_out_split(self, domain_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        留一法域自适应分割

        Args:
            domain_config: 域配置

        Returns:
            train_data, val_data, test_data
        """
        target_domain = domain_config.get('target_domain')
        unique_domains = sorted(self.metadata['Domain_id'].unique())

        if target_domain is None:
            # 默认使用最后一个domain作为测试
            target_domain = unique_domains[-1]

        print(f"Leave-one-domain-out: target domain = {target_domain}")

        # 分割数据
        test_data = self.metadata[self.metadata['Domain_id'] == target_domain]
        train_val_data = self.metadata[self.metadata['Domain_id'] != target_domain]

        # 训练验证集分割
        split_config = self.vb_config.get('split_config', {})
        val_ratio = split_config.get('val_ratio', 0.1)

        if len(train_val_data) > 0:
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_ratio,
                random_state=getattr(self.args, 'seed', 42)
            )
        else:
            train_data = train_val_data
            val_data = pd.DataFrame()  # 空验证集

        return train_data, val_data, test_data

    def _build_sample_indices(self):
        """构建样本索引列表"""
        # 如果启用了智能采样
        if self.sampling_config.get('method') == 'smart':
            self.samples = self._smart_sampling()
        else:
            # 常规采样（每个样本一个窗口）
            self.samples = []
            for idx, row in self.data_for_sampling.iterrows():
                self.samples.append({
                    'id': str(row['Id']),
                    'index': idx,
                    'start_pos': 0,
                    'label': row[self.target_column],
                    'window_length': getattr(self.args, 'in_dim', 4096)
                })

    def _smart_sampling(self) -> List[Dict]:
        """
        智能采样实现

        Returns:
            samples: 采样结果列表
        """
        from .vbench_utils import SmartSampler

        # 创建采样器
        sampler = SmartSampler(self.data_for_sampling, self.sampling_config, getattr(self.args, 'seed', 42))

        # 执行采样
        samples_info = sampler.sample_all_classes()

        # 转换为内部格式
        samples = []
        for sample in samples_info:
            samples.append({
                'id': sample['id'],
                'index': -1,  # 将在__getitem__中动态获取
                'start_pos': sample['start_pos'],
                'label': sample['label'],
                'window_length': sample['window_length']
            })

        return samples

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def _get_h5_file(self, sample_id: str) -> h5py.File:
        """
        根据样本ID获取对应的HDF5文件

        Args:
            sample_id: 样本ID

        Returns:
            h5py.File: 对应的HDF5文件对象
        """
        # 从metadata中获取该样本对应的Dataset_id
        sample_meta = self.metadata[self.metadata['Id'] == sample_id]
        if sample_meta.empty:
            raise ValueError(f"Sample ID {sample_id} not found in metadata")

        dataset_id = int(sample_meta.iloc[0]['Dataset_id'])

        # 使用fallback cache（如果启用）
        if self.use_fallback_cache and self.h5_file is not None:
            return self.h5_file

        # 获取对应的数据集名称
        if dataset_id not in self.dataset_mapping:
            raise ValueError(f"Dataset_id {dataset_id} not found in dataset mapping")

        dataset_name = self.dataset_mapping[dataset_id]
        h5_path = self.data_dir / f"{dataset_name}.h5"

        # 缓存机制：如果文件已打开则直接返回
        if str(h5_path) in self.h5_cache:
            return self.h5_cache[str(h5_path)]

        # 检查文件是否存在
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        # 打开并缓存文件
        print(f"Opening HDF5 file for dataset {dataset_id}: {h5_path}")
        h5_file = h5py.File(h5_path, 'r')
        self.h5_cache[str(h5_path)] = h5_file

        return h5_file

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            data: 信号数据 [C, L]
            label: 标签
        """
        sample_info = self.samples[idx]

        # 动态获取HDF5文件
        h5_file = self._get_h5_file(sample_info['id'])

        # 从HDF5读取数据
        sample_id_str = str(sample_info['id'])  # 转换为字符串
        if sample_id_str not in h5_file:
            raise KeyError(f"ID {sample_id_str} not found in HDF5 file {h5_file.filename}")

        # 读取原始数据 [L, C]
        raw_data = h5_file[sample_id_str][...]

        # 滑动窗口提取
        start_pos = sample_info['start_pos']
        end_pos = start_pos + sample_info['window_length']
        window_data = raw_data[start_pos:end_pos]

        # 统一返回 [L, C]（与仓库其他数据集/模型期望一致：输入为 (batch, seq_len, channels)）
        if window_data.ndim == 2:
            data = torch.from_numpy(window_data).float()
        else:
            # 处理单通道情况：[L] -> [L, 1]
            data = torch.from_numpy(window_data).float().unsqueeze(-1)

        # 数据增强
        if self.transform:
            data = self.transform(data)

        # 获取标签
        label = sample_info['label']

        # 转换标签为tensor
        if isinstance(label, (int, np.integer)):
            label = torch.tensor(label, dtype=torch.long)
        elif isinstance(label, (float, np.floating)):
            label = torch.tensor(label, dtype=torch.float)

        return data, label

    def get_class_distribution(self) -> Dict:
        """获取类别分布"""
        labels = [s['label'] for s in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    def get_num_classes(self) -> int:
        """获取类别数量"""
        unique_labels = self.metadata[self.target_column].unique()
        return len(unique_labels)

    def close(self):
        """关闭所有HDF5文件"""
        # 关闭缓存的HDF5文件
        for h5_path, h5_file in self.h5_cache.items():
            try:
                h5_file.close()
                print(f"Closed HDF5 file: {h5_path}")
            except:
                pass
        self.h5_cache.clear()

        # 关闭fallback文件
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
                print("Closed fallback HDF5 file")
            except:
                pass

    def __del__(self):
        """析构函数"""
        self.close()


class VbenchSubset(Dataset):
    """
    Vbench数据集子集，用于创建特定的子集（如验证集）
    """

    def __init__(self, dataset: VbenchDataset, indices: List[int]):
        """
        初始化子集

        Args:
            dataset: 原始数据集
            indices: 要包含的索引列表
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.indices[idx]
        return self.dataset[original_idx]
