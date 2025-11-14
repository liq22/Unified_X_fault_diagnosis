#!/usr/bin/env python3
"""
测试解耦后的DataLoader流程
验证从不同的H5文件动态加载数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torch.utils.data import DataLoader
from data.vbench_dataset import VbenchDataset
import argparse


class Args:
    """模拟参数对象"""
    def __init__(self):
        self.vbench_config = {
            'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
            'metadata_file': 'metadata_6_11.xlsx',
            'dataset_ids': [1],  # 测试数据集1
            'task_type': 'fault_diagnosis',
            'target_column': 'Label',
            'sampling_config': {
                'method': 'smart',
                'target_per_class': 5,  # 进一步减少样本数以避免小数据集问题
                'ids_cap': 20
            },
            'window_config': {
                'window_length': 4096,
                'stride': 1024
            },
            'split_config': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'stratify_by': 'Label'
            },
            'domain_config': {
                'leave_one_domain_out': False
            },
            'augmentation': {
                'enable': False
            },
            'fallback_cache': False  # 禁用fallback，强制使用解耦的H5文件
        }


def test_single_dataset(dataset_id, batch_size=32, num_workers=4):
    """测试单个数据集的解耦加载"""
    print(f"\n{'='*60}")
    print(f"测试数据集 {dataset_id}")
    print(f"{'='*60}")

    # 创建args对象
    args = Args()
    args.vbench_config['dataset_ids'] = [dataset_id]

    try:
        # 创建数据集
        dataset = VbenchDataset(args, flag='train')
        print(f"✓ 数据集创建成功: {len(dataset)} 样本")
        print(f"✓ 类别数: {dataset.get_num_classes()}")
        print(f"✓ 类别分布: {dataset.get_class_distribution()}")

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"✓ DataLoader创建成功: {len(dataloader)} 批次")

        # 测试数据加载
        start_time = time.time()
        for i, (data, labels) in enumerate(dataloader):
            if i >= 3:  # 只测试前3个批次
                break

            print(f"  Batch {i+1}/{len(dataloader)}: data={data.shape}, labels={labels.shape}, time={time.time()-start_time:.3f}s")

            # 验证数据类型
            if not isinstance(data, torch.Tensor):
                print(f"  ⚠️ 警告: data类型为 {type(data)}")
            if labels.dtype != torch.long:
                print(f"  ⚠️ 警告: label类型为 {labels.dtype}，应为torch.long")

        print(f"✓ 数据集 {dataset_id} 测试成功")
        return True

    except Exception as e:
        print(f"✗ 数据集 {dataset_id} 测试失败: {str(e)}")
        return False
    finally:
        if 'dataset' in locals():
            dataset.close()


def test_multiple_datasets(dataset_ids, batch_size=32, num_workers=4):
    """测试多个数据集的混合加载"""
    print(f"\n{'='*60}")
    print(f"测试多数据集混合加载: {dataset_ids}")
    print(f"{'='*60}")

    # 创建args对象
    args = Args()
    args.vbench_config['dataset_ids'] = dataset_ids

    try:
        # 创建数据集
        dataset = VbenchDataset(args, flag='train')
        print(f"✓ 数据集创建成功: {len(dataset)} 样本")
        print(f"✓ 类别数: {dataset.get_num_classes()}")
        print(f"✓ 类别分布: {dataset.get_class_distribution()}")

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"✓ DataLoader创建成功: {len(dataloader)} 批次")

        # 测试数据加载，验证来自不同数据集的样本
        for i, (data, labels) in enumerate(dataloader):
            if i >= 10:  # 测试前10个批次
                break

            print(f"  Batch {i+1}: data={data.shape}, labels={labels.shape}")

        print(f"✓ 多数据集混合加载测试成功")
        return True

    except Exception as e:
        print(f"✗ 多数据集混合加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'dataset' in locals():
            dataset.close()


def main():
    parser = argparse.ArgumentParser(description='测试解耦后的DataLoader')
    parser.add_argument('--single-dataset', type=int, help='测试单个数据集')
    parser.add_argument('--id-range', type=str, help='测试ID范围，格式：1-3')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--all', action='store_true', help='测试所有数据集')

    args = parser.parse_args()

    print("Vbench 解耦数据集加载测试")
    print("=" * 60)
    print("测试说明：")
    print("- 每个数据集使用独立的H5文件")
    print("- 根据metadata中的Dataset_id动态选择对应的H5文件")
    print("- 支持多数据集混合加载")
    print("=" * 60)

    if args.single_dataset:
        # 测试单个数据集
        test_single_dataset(args.single_dataset, args.batch_size, args.num_workers)

    elif args.id_range:
        # 测试ID范围
        start, end = map(int, args.id_range.split('-'))
        for dataset_id in range(start, end + 1):
            test_single_dataset(dataset_id, args.batch_size, args.num_workers)

    elif args.all:
        # 测试所有数据集
        dataset_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for dataset_id in dataset_ids:
            test_single_dataset(dataset_id, args.batch_size, args.num_workers)

    else:
        # 默认测试几个数据集
        test_datasets = [1, 6]  # CWRU和THU
        for dataset_id in test_datasets:
            test_single_dataset(dataset_id, args.batch_size, args.num_workers)

        # 测试多数据集混合
        print("\n")
        test_multiple_datasets([1, 6], args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
