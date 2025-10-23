#!/usr/bin/env python3
"""
Vbench数据集使用示例
演示如何使用VbenchDataset进行故障诊断实验
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data.vbench_dataset import VbenchDataset
from data.vbench_utils import SmartSampler, create_balanced_sampler
from trainer.trainer_set import PL_Trainer
from model.TSPN import TSPN


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vbench故障诊断示例')

    # 配置文件
    parser.add_argument(
        '--config_file',
        type=str,
        default='configs/vbench/config_vbench_diagnosis.yaml',
        help='配置文件路径'
    )

    # 数据集选项
    parser.add_argument(
        '--dataset',
        type=str,
        default='VBENCH_CWRU',
        choices=['VBENCH_CWRU', 'VBENCH_XJTU', 'VBENCH_all'],
        help='数据集选择'
    )

    # 快速模式（小数据集）
    parser.add_argument(
        '--quick',
        action='store_true',
        help='使用小数据集进行快速测试'
    )

    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """
    加载YAML配置文件
    简化实现，实际项目可使用hydra或omegaconf
    """
    import yaml

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_model(args):
    """创建模型"""
    if args.model == 'TSPN':
        return TSPN(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    print(f"Loading config from: {args.config_file}")
    config = load_config(args.config_file)

    # 合并命令行参数
    args_dict = vars(args)
    for key, value in config.get('args', {}).items():
        if key not in args_dict or args_dict[key] is None:
            setattr(args, key, value)

    # 添加数据集任务
    args.dataset_task = args.dataset

    # 快速模式配置
    if args.quick:
        print("Quick mode: using reduced dataset")
        if 'vbench_config' in config:
            if 'sampling_config' in config['vbench_config']:
                config['vbench_config']['sampling_config']['target_per_class'] = 100
                config['vbench_config']['sampling_config']['ids_cap'] = 20
            args.num_epochs = 10
            args.batch_size = 16

    # 更新配置到args
    for key, value in config.get('vbench_config', {}).items():
        setattr(args, key, value)

    print(f"\nDataset task: {args.dataset_task}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")

    # 创建数据集
    print("\nCreating datasets...")

    # 训练集
    train_dataset = VbenchDataset(args, flag='train')
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Classes: {train_dataset.get_num_classes()}")
    print(f"Class distribution: {train_dataset.get_class_distribution()}")

    # 验证集
    val_dataset = VbenchDataset(args, flag='val')
    print(f"Validation set: {len(val_dataset)} samples")

    # 测试集
    test_dataset = VbenchDataset(args, flag='test')
    print(f"Test set: {len(test_dataset)} samples")

    # 创建数据加载器
    print("\nCreating data loaders...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    print("\nCreating model...")
    model = create_model(args)

    # 创建训练器
    print("Setting up trainer...")
    trainer = PL_Trainer(args, model)

    # 开始训练
    print("\nStarting training...")
    try:
        trainer.train(train_loader, val_loader)

        # 测试
        print("\nEvaluating on test set...")
        test_acc = trainer.test(test_loader)
        print(f"Test accuracy: {test_acc:.2f}%")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\nCleaning up...")
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()

    print("\nDone!")


if __name__ == "__main__":
    main()