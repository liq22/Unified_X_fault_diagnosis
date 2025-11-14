#!/usr/bin/env python3
"""
使用DataLoader测试Vbench数据集数据流
模拟真实训练场景，验证数据加载、批次处理和性能
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from collections import defaultdict
import gc
import argparse

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/dataloader_test.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vbench DataLoader测试')

    # 数据集选择
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all-datasets',
        action='store_true',
        help='测试所有数据集'
    )
    group.add_argument(
        '--id-range',
        type=str,
        help='ID范围，格式：1-5'
    )
    group.add_argument(
        '--single-dataset',
        type=int,
        help='测试单个数据集'
    )

    # DataLoader参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='数据加载器工作进程数'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='遍历次数（epoch数）'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试（只测试3个批次）'
    )

    # 输出控制
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    parser.add_argument(
        '--save-stats',
        action='store_true',
        help='保存统计信息到文件'
    )

    return parser.parse_args()


def test_single_dataset(dataset_id, config, args):
    """
    测试单个数据集的DataLoader
    """
    import torch
    from torch.utils.data import DataLoader
    from data.vbench_dataset import VbenchDataset

    logger.info(f"\n{'='*50}")
    logger.info(f"测试数据集 {dataset_id}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"工作进程: {args.num_workers}")

    try:
        # 创建数据集
        dataset = VbenchDataset(config, flag='train', use_cache=True)
        logger.info(f"✓ 数据集创建成功: {len(dataset)} 样本")
        logger.info(f"✓ 类别数: {dataset.get_num_classes()}")
        logger.info(f"✓ 类别分布: {dataset.get_class_distribution()}")

        # 检查数据有效性
        if len(dataset) == 0:
            logger.warning("⚠ 数据集为空，跳过测试")
            return {
                'dataset_id': dataset_id,
                'success': False,
                'error': 'Empty dataset'
            }

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"✓ DataLoader创建成功: {len(dataloader)} 批次")

        # 测试数据流
        stats = {
            'dataset_id': dataset_id,
            'dataset_size': len(dataset),
            'num_classes': dataset.get_num_classes(),
            'num_batches': len(dataloader),
            'batch_size': args.batch_size,
            'sample_shapes': [],
            'label_shapes': [],
            'loading_times': [],
            'memory_usage': [],
            'class_distribution': dataset.get_class_distribution()
        }

        # 遍历epoch
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

            batch_count = 0
            total_time = 0

            # 遍历批次
            for batch_idx, (data, labels) in enumerate(dataloader):
                start_time = time.time()

                # 验证数据
                assert isinstance(data, torch.Tensor), f"批次 {batch_idx}: data不是tensor"
                assert isinstance(labels, torch.Tensor), f"批次 {batch_idx}: labels不是tensor"

                # 记录形状
                data_shape = list(data.shape)
                label_shape = list(labels.shape)
                stats['sample_shapes'].append(data_shape)
                stats['label_shapes'].append(label_shape)

                # 检查数据范围
                if torch.isnan(data).any():
                    logger.warning(f"批次 {batch_idx}: 发现NaN值")
                if torch.isinf(data).any():
                    logger.warning(f"批次 {batch_idx}: 发现Inf值")

                # 数据类型检查
                if data.dtype != torch.float32:
                    logger.warning(f"批次 {batch_idx}: data类型为 {data.dtype}，应为float32")
                if labels.dtype != torch.int64:
                    logger.warning(f"批次 {batch_idx}: label类型为 {labels.dtype}，应为int64")

                # 加载时间
                load_time = time.time() - start_time
                stats['loading_times'].append(load_time)
                total_time += load_time

                batch_count += 1

                # 详细输出
                if args.verbose:
                    logger.info(
                        f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                        f"data={data_shape}, labels={label_shape}, "
                        f"time={load_time:.4f}s"
                    )

                # 快速测试模式
                if args.quick_test and batch_count >= 3:
                    break

            # epoch统计
            avg_load_time = total_time / batch_count if batch_count > 0 else 0
            logger.info(f"  完成 {batch_count} 批次")
            logger.info(f"  平均加载时间: {avg_load_time:.4f}s/批")

            # 内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                stats['memory_usage'].append(memory_used)
                logger.info(f"  GPU内存使用: {memory_used:.1f}MB / {memory_reserved:.1f}MB")

            # 清理GPU内存
            torch.cuda.empty_cache()

        # 计算总体统计
        stats['success'] = True
        stats['error'] = None
        stats['avg_loading_time'] = np.mean(stats['loading_times']) if stats['loading_times'] else 0
        stats['max_loading_time'] = np.max(stats['loading_times']) if stats['loading_times'] else 0
        stats['samples_per_second'] = stats['dataset_size'] / total_time if total_time > 0 else 0

        logger.info(f"✓ 数据集 {dataset_id} 测试完成")
        logger.info(f"  总样本数: {stats['dataset_size']}")
        logger.info(f"  样本/秒: {stats['samples_per_second']:.1f}")
        logger.info(f"  平均加载时间: {stats['avg_loading_time']:.4f}s")

        # 关闭数据集
        dataset.close()
        return stats

    except Exception as e:
        logger.error(f"✗ 数据集 {dataset_id} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset_id': dataset_id,
            'success': False,
            'error': str(e)
        }
    finally:
        # 确保内存清理
        if 'torch' in sys.modules:
            torch.cuda.empty_cache()
        gc.collect()


def get_all_dataset_ids(config):
    """获取所有可用的Dataset ID"""
    metadata_path = config['vbench_config']['data_dir'] + '/' + config['vbench_config']['metadata_file']

    try:
        metadata = pd.read_excel(metadata_path)
        dataset_ids = sorted(metadata['Dataset_id'].unique())

        # 过滤并转换为整数
        valid_ids = []
        for did in dataset_ids:
            try:
                if isinstance(did, (int, float)) or (isinstance(did, str) and did.isdigit()):
                    valid_ids.append(int(did))
            except:
                continue

        logger.info(f"找到 {len(valid_ids)} 个数据集: {valid_ids}")
        return valid_ids
    except Exception as e:
        logger.error(f"获取Dataset IDs失败: {e}")
        return []


def save_test_results(results, args):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON格式
    import json
    json_path = f"results/dataloader_test_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存到: {json_path}")

    # 保存CSV汇总
    csv_path = f"results/dataloader_test_summary_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("Dataset_ID,Success,Error,Size,Classes,Batches,Avg_Time(s),Samples/s\n")
        for result in results:
            if result['success']:
                f.write(f"{result['dataset_id']},True,,{result['dataset_size']},"
                          f"{result['num_classes']},{result['num_batches']},"
                          f"{result['avg_loading_time']:.3f},{result['samples_per_second']:.1f}\n")
            else:
                f.write(f"{result['dataset_id']},False,{result['error']},0,0,0,0,0\n")
    logger.info(f"汇总已保存到: {csv_path}")


def print_summary(results):
    """打印测试总结"""
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful

    print("\n" + "="*60)
    print("DataLoader测试总结")
    print("="*60)
    print(f"总数据集: {total}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    if total > 0:
        print(f"成功率: {100*successful/total:.1f}%")
    else:
        print("成功率: 无法计算（总数为0）")

    if successful > 0:
        sizes = [r['dataset_size'] for r in results if r['success']]
        times = [r['avg_loading_time'] for r in results if r['success']]

        print(f"\n成功的数据集统计:")
        print(f"  平均数据量: {np.mean(sizes):.0f}")
        print(f"  最大数据量: {max(sizes):.0f}")
        print(f"  平均加载时间: {np.mean(times):.4f}s")
        print(f"  最慢加载时间: {max(times):.4f}s")

    if failed > 0:
        failed_ids = [r['dataset_id'] for r in results if not r['success']]
        print(f"\n失败的数据集: {failed_ids}")

    print("="*60)


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    try:
        import yaml
        with open('configs/vbench/config_vbench_diagnosis.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("配置文件未找到")
        return

    # 获取数据集ID列表
    if args.all_datasets:
        dataset_ids = get_all_dataset_ids(config)
    elif args.id_range:
        start_id, end_id = map(int, args.id_range.split('-'))
        all_ids = get_all_dataset_ids(config)
        dataset_ids = [i for i in all_ids if start_id <= i <= end_id]
    elif args.single_dataset:
        dataset_ids = [args.single_dataset]
    else:
        logger.error("请指定数据集: --all-datasets, --id-range 或 --single-dataset")
        return

    logger.info(f"将测试 {len(dataset_ids)} 个数据集: {dataset_ids}")

    # 测试所有数据集
    results = []
    for dataset_id in dataset_ids:
        # 更新配置中的dataset_ids
        test_config = config.copy()
        test_config['vbench_config']['dataset_ids'] = [dataset_id]
        test_config['args']['dataset_task'] = f'VBENCH_{dataset_id}'

        # 测试数据集
        result = test_single_dataset(dataset_id, test_config, args)
        results.append(result)

        # 保存中间结果
        if args.save_stats:
            save_test_results([result], args)

    # 打印总结
    print_summary(results)

    # 保存最终结果
    if args.save_stats:
        save_test_results(results, args)


if __name__ == "__main__":
    main()