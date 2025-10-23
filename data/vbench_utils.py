import numpy as np
import pandas as pd
import random
import math
import h5py
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split


def set_seed(seed: int):
    """设置随机种子"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


class SmartSampler:
    """
    智能采样器，实现分层采样和ID池机制
    支持精确控制每类样本数量，确保数据多样性
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        config: Dict,
        seed: Optional[int] = 42
    ):
        """
        初始化智能采样器

        Args:
            metadata: 元数据DataFrame
            config: 采样配置字典
            seed: 随机种子
        """
        self.metadata = metadata
        self.config = config
        self.seed = seed

        # 设置随机种子
        set_seed(seed)

        # 配置参数
        self.ids_cap = config.get('ids_cap', 50)
        self.target_per_class = config.get('target_per_class', 1000)
        self.min_samples_per_id = config.get('min_samples_per_id', 1)
        self.window_length = config.get('window_length', 4096)
        self.window_strategy = config.get('window_strategy', 'random')
        self.resample_strategy = config.get('resample_strategy', 'repeat')
        self.target_column = config.get('target_column', 'Label')

        # 日志配置
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def build_id_pool(self, label_ids: List[str]) -> List[str]:
        """
        构建ID池

        Args:
            label_ids: 某个类别的所有ID列表

        Returns:
            id_pool: 构建好的ID池
        """
        num_ids = len(label_ids)

        if num_ids >= self.ids_cap:
            # 等概率无放回抽取
            id_pool = random.sample(label_ids, self.ids_cap)
            self.logger.info(
                f"Label pool: {num_ids} IDs → {self.ids_cap} IDs (sampled)"
            )
        else:
            # 循环重复直到达到池大小
            pool = []
            full_cycles = self.ids_cap // num_ids
            remainder = self.ids_cap % num_ids

            # 完整循环
            for _ in range(full_cycles):
                pool.extend(label_ids)

            # 剩余部分
            pool.extend(label_ids[:remainder])
            id_pool = pool

            self.logger.info(
                f"Label pool: {num_ids} IDs → {self.ids_cap} IDs "
                f"(repeated {full_cycles} times)"
            )

        return id_pool

    def _get_valid_windows(self, data_length: int) -> List[int]:
        """
        获取有效的窗口起始位置

        Args:
            data_length: 数据长度

        Returns:
            valid_positions: 有效起始位置列表
        """
        if self.window_strategy == 'sequential':
            # 顺序采样
            positions = list(range(0, data_length - self.window_length + 1,
                               self.window_length))
        elif self.window_strategy == 'random':
            # 随机采样，允许重叠
            max_start = max(0, data_length - self.window_length)
            if max_start > 0:
                positions = [random.randint(0, max_start)]
            else:
                positions = [0]
        else:
            # 默认使用顺序采样
            positions = list(range(0, data_length - self.window_length + 1,
                               self.window_length))

        return positions

    def sample_class(self, label_id_pool: List[str], label_name: Union[str, int, float]) -> List[Dict]:
        """
        为单个类别采样

        Args:
            label_id_pool: 该类别的ID池
            label_name: 类别名称

        Returns:
            samples: 采样结果列表
        """
        samples = []
        samples_per_id = max(
            self.min_samples_per_id,
            math.ceil(self.target_per_class / len(label_id_pool))
        )

        # 统计每个ID被选中的次数
        id_counts = {}
        while len(samples) < self.target_per_class:
            # 从ID池中等概率抽取一个ID
            selected_id = random.choice(label_id_pool)

            # 更新计数
            id_counts[selected_id] = id_counts.get(selected_id, 0) + 1

            # 加载该ID的数据文件长度信息
            data_length = self._get_data_length(selected_id)

            if data_length < self.window_length:
                self.logger.warning(
                    f"ID {selected_id} length {data_length} < window length {self.window_length}"
                )
                continue

            # 获取有效的窗口位置
            valid_positions = self._get_valid_windows(data_length)
            if not valid_positions:
                continue

            # 选择窗口起始位置
            start_pos = random.choice(valid_positions)

            samples.append({
                'id': selected_id,
                'start_pos': start_pos,
                'label': label_name,
                'window_length': self.window_length
            })

        # 打印采样统计
        self.logger.info(f"Sampled {len(samples)} samples for label {label_name}")
        self.logger.info(f"ID usage: {id_counts}")

        return samples[:self.target_per_class]  # 确保不超过目标数

    def _get_data_length(self, id: str) -> int:
        """
        获取数据长度（从metadata或实际加载）

        Args:
            id: 数据ID

        Returns:
            length: 数据长度
        """
        # 首先尝试从metadata获取
        row = self.metadata[self.metadata['Id'] == id]
        if not row.empty:
            sample_length = row.iloc[0].get('Sample_lenth', None)
            if sample_length is not None:
                return int(sample_length)

        # 如果metadata中没有，返回默认值
        self.logger.warning(f"Cannot find length for ID {id}, using default")
        return self.window_length * 2  # 假设是窗口长度的2倍

    def sample_all_classes(self) -> List[Dict]:
        """
        为所有类别采样

        Returns:
            all_samples: 所有类别的采样结果
        """
        # 按Label分组
        class_groups = self.metadata.groupby(self.target_column)
        self.logger.info(f"Found {len(class_groups)} classes")

        all_samples = []
        total_target = 0

        for label_name, group in class_groups:
            # 获取该类别的所有ID
            label_ids = group['Id'].unique().tolist()
            class_target = self.target_per_class
            total_target += class_target

            self.logger.info(f"\nProcessing label {label_name}:")
            self.logger.info(f"  Total IDs: {len(label_ids)}")

            # 构建ID池
            id_pool = self.build_id_pool(label_ids)

            # 为该类采样
            samples = self.sample_class(id_pool, label_name)
            all_samples.extend(samples)

        self.logger.info(f"\nTotal samples: {len(all_samples)}")
        self.logger.info(f"Target total: {total_target}")

        # 验证采样结果
        self._validate_sampling(all_samples)

        return all_samples

    def _validate_sampling(self, samples: List[Dict]):
        """
        验证采样结果

        Args:
            samples: 采样结果
        """
        # 统计每类样本数
        label_counts = {}
        for sample in samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        self.logger.info("\nSampling validation:")
        for label, count in sorted(label_counts.items()):
            target = self.target_per_class
            diff = count - target
            self.logger.info(
                f"  Label {label}: {count} samples "
                f"(target: {target}, diff: {diff:+d})"
            )

        # 统计ID使用情况
        id_usage = {}
        for sample in samples:
            id = sample['id']
            id_usage[id] = id_usage.get(id, 0) + 1

        unique_ids = len(id_usage)
        total_ids = sum(
            len(self.metadata[self.metadata[self.target_column] == label]['Id'].unique())
            for label in self.metadata[self.target_column].unique()
        )

        self.logger.info(f"\nID usage statistics:")
        self.logger.info(f"  Unique IDs used: {unique_ids}")
        self.logger.info(f"  Total unique IDs available: {total_ids}")
        self.logger.info(f"  Coverage: {100*unique_ids/total_ids:.1f}%")


class VbenchDataLoader:
    """
    Vbench专用数据加载器，支持批量加载和预处理
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn=None
    ):
        """
        初始化数据加载器

        Args:
            dataset: Vbench数据集
            batch_size: 批量大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后不完整的batch
            collate_fn: 自定义批处理函数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self):
        """迭代器"""
        # 这里可以添加自定义的批处理逻辑
        # 暂时返回基础实现
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )
        return iter(loader)


def create_balanced_sampler(
    dataset,
    sampling_mode: str = 'class',
    replacement: bool = False
):
    """
    创建平衡采样器

    Args:
        dataset: 数据集
        sampling_mode: 'class'或'group'
        replacement: 是否有放回采样

    Returns:
        sampler: 平衡采样器
    """
    from torch.utils.data import WeightedRandomSampler

    if sampling_mode == 'class':
        # 获取所有标签
        labels = [dataset[i][1] for i in range(len(dataset))]
        unique_labels = np.unique(labels)

        # 计算每个样本的权重（样本数少的类别权重高）
        class_counts = np.array([np.sum(labels == label) for label in unique_labels])
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]

        # 创建加权采样器
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=replacement
        )
    else:
        sampler = None

    return sampler


def load_vbench_metadata(
    data_dir: Union[str, Path],
    metadata_file: str = 'metadata.xlsx'
) -> pd.DataFrame:
    """
    加载Vbench元数据

    Args:
        data_dir: 数据目录
        metadata_file: 元数据文件名

    Returns:
        metadata: 元数据DataFrame
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_excel(metadata_path)
    return metadata