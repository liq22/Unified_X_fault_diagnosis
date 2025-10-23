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

        # 提取Vbench配置
        self.vb_config = getattr(args, 'vbench_config', {})
        self.sampling_config = self.vb_config.get('sampling_config', {})

        # 数据路径
        self.data_dir = Path(self.vb_config['data_dir'])
        self.metadata_file = self.data_dir / self.vb_config['metadata_file']
        self.data_file = self.data_dir / self.vb_config['data_file']

        # 加载元数据
        print(f"Loading metadata from: {self.metadata_file}")
        self.metadata = pd.read_excel(self.metadata_file)

        # 数据集筛选
        if 'dataset_ids' in self.vb_config:
            self.metadata = self.metadata[
                self.metadata['Dataset_id'].isin(self.vb_config['dataset_ids'])
            ].reset_index(drop=True)

        # 打开HDF5文件
        print(f"Opening HDF5 file: {self.data_file}")
        self.h5_file = h5py.File(self.data_file, 'r')

        # 任务类型
        self.task_type = self.vb_config.get('task_type', 'fault_diagnosis')
        self.target_column = self.vb_config.get('target_column', 'Label')

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

        if stratify_by in self.metadata.columns:
            train_data, temp_data = train_test_split(
                self.metadata,
                test_size=1-train_ratio,
                stratify=self.metadata[stratify_by],
                random_state=self.args.seed
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio/(test_ratio+val_ratio),
                stratify=temp_data[stratify_by],
                random_state=self.args.seed
            )
        else:
            train_data, temp_data = train_test_split(
                self.metadata,
                test_size=1-train_ratio,
                random_state=self.args.seed
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio/(test_ratio+val_ratio),
                random_state=self.args.seed
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
                random_state=self.args.seed
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
                    'window_length': self.args.in_dim
                })

    def _smart_sampling(self) -> List[Dict]:
        """
        智能采样实现

        Returns:
            samples: 采样结果列表
        """
        from .vbench_utils import SmartSampler

        # 创建采样器
        sampler = SmartSampler(self.data_for_sampling, self.sampling_config, self.args.seed)

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

        # 从HDF5读取数据
        if sample_info['id'] not in self.h5_file:
            raise KeyError(f"ID {sample_info['id']} not found in HDF5 file")

        # 读取原始数据 [L, C]
        raw_data = self.h5_file[sample_info['id']][...]

        # 滑动窗口提取
        start_pos = sample_info['start_pos']
        end_pos = start_pos + sample_info['window_length']
        window_data = raw_data[start_pos:end_pos]

        # 转换维度为 [C, L]
        if window_data.ndim == 2:
            data = torch.from_numpy(window_data.T).float()
        else:
            # 处理单通道情况
            data = torch.from_numpy(window_data).float().unsqueeze(0)

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
        """关闭HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

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