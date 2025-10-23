#!/usr/bin/env python3
"""
Vbench数据集遍历测试脚本
支持遍历所有20个数据集或指定ID范围
"""

import os
import sys
import argparse
import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vbench_test.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vbench数据集批量测试')

    # 数据集选择
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all-datasets',
        action='store_true',
        help='测试所有数据集（ID 1-20）'
    )
    group.add_argument(
        '--id-range',
        type=str,
        help='指定ID范围，格式：1-5, 10-20'
    )
    group.add_argument(
        '--single-dataset',
        type=int,
        help='测试单个数据集（指定Dataset ID）'
    )

    # 运行选项
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='每个数据集的训练轮数（默认：10）'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速测试模式（更小的数据集）'
    )
    parser.add_argument(
        '--config-template',
        type=str,
        default='configs/vbench/config_vbench_diagnosis.yaml',
        help='配置模板文件'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/vbench_test',
        help='结果输出目录'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从上次中断的地方继续'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        help='并行GPU数量'
    )

    return parser.parse_args()


def load_base_config(template_path):
    """加载基础配置模板"""
    try:
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"配置模板文件未找到: {template_path}")
        sys.exit(1)


def create_dataset_config(base_config, dataset_id, epochs=10, quick=False):
    """为单个数据集创建配置"""
    # 深拷贝配置
    config = {}
    for key, value in base_config.items():
        if key == 'vbench_config':
            config[key] = value.copy()
        else:
            config[key] = value

    # 更新数据集ID（直接使用数字）
    config['vbench_config']['dataset_ids'] = [dataset_id]
    config['vbench_config']['task_type'] = 'fault_diagnosis'
    config['vbench_config']['target_column'] = 'Label'

    # 快速模式配置
    if quick:
        config['vbench_config']['sampling_config']['target_per_class'] = 100
        config['vbench_config']['sampling_config']['ids_cap'] = 20

    # 更新训练参数
    config['args']['num_epochs'] = epochs
    config['args']['dataset_task'] = f'VBENCH_{dataset_id}'

    # 添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['args']['experiment_name'] = f'VBENCH_{dataset_id}_test_{timestamp}'

    return config


def save_config(config, dataset_id, output_dir):
    """保存配置文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 配置文件名
    config_name = f'config_VBENCH_{dataset_id}.yaml'
    config_path = os.path.join(output_dir, config_name)

    # 保存配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return config_path


def run_experiment(config_path, dataset_id, parallel=None, resume=False):
    """运行单个实验"""
    cmd = ['python', 'main.py', '--config_file', config_path]

    # 添加并行设置
    if parallel:
        cmd.extend(['--gpus', str(parallel)])

    # 添加恢复设置
    if resume:
        cmd.append('--resume')

    logger.info(f"运行数据集 {dataset_id}: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"数据集 {dataset_id} 完成，返回码: {result.returncode}")
        return True, result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"数据集 {dataset_id} 运行失败: {e}")
        return False, -1
    except Exception as e:
        logger.error(f"数据集 {dataset_id} 运行出错: {e}")
        return False, -1


def get_dataset_ids_from_metadata(metadata_path):
    """从metadata文件中获取所有Dataset ID"""
    try:
        metadata = pd.read_excel(metadata_path)
        # 提取所有Dataset_id并去重
        dataset_ids = metadata['Dataset_id'].dropna().unique()
        # 过滤出数字ID（去除可能的非数字值）
        numeric_ids = []
        for did in dataset_ids:
            # 尝试转换为数字
            try:
                if isinstance(did, (int, float)):
                    numeric_ids.append(int(did))
                elif isinstance(did, str) and did.isdigit():
                    numeric_ids.append(int(did))
            except:
                continue

        # 排序并去重
        unique_ids = sorted(list(set(numeric_ids)))
        return unique_ids
    except Exception as e:
        logger.error(f"读取metadata失败: {e}")
        return []


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载基础配置
    base_config = load_base_config(args.config_template)
    if not base_config:
        return

    # 获取metadata路径
    metadata_path = base_config['vbench_config']['data_dir'] + '/' + base_config['vbench_config']['metadata_file']

    # 获取所有可用的Dataset ID
    all_dataset_ids = get_dataset_ids_from_metadata(metadata_path)
    logger.info(f"发现 {len(all_dataset_ids)} 个数据集: {all_dataset_ids}")

    # 确定要测试的数据集
    if args.all_datasets:
        selected_ids = all_dataset_ids
        logger.info("模式：测试所有数据集")
    elif args.id_range:
        # 解析ID范围
        try:
            start_id, end_id = map(int, args.id_range.split('-'))
            selected_ids = [i for i in all_dataset_ids if start_id <= i <= end_id]
            logger.info(f"模式：测试ID范围 {start_id}-{end_id}")
        except:
            logger.error("ID范围格式错误，应为：1-5 或 10-20")
            return
    elif args.single_dataset:
        selected_ids = [args.single_dataset]
        if args.single_dataset not in all_dataset_ids:
            logger.error(f"Dataset ID {args.single_dataset} 不存在")
            logger.info(f"可用的ID: {all_dataset_ids}")
            return
        logger.info(f"模式：测试单个数据集 {args.single_dataset}")
    else:
        logger.error("请指定测试模式：--all-datasets, --id-range 或 --single-dataset")
        return

    logger.info(f"将测试 {len(selected_ids)} 个数据集: {selected_ids}")

    # 结果汇总
    results = {
        'start_time': datetime.now(),
        'total_datasets': len(selected_ids),
        'successful': [],
        'failed': [],
        'results': {}
    }

    # 遍历数据集
    for i, dataset_id in enumerate(selected_ids, 1):
        logger.info(f"\n[{i}/{len(selected_ids)}] 处理数据集 {dataset_id}")

        # 创建配置
        config = create_dataset_config(
            base_config,
            dataset_id,
            epochs=args.epochs,
            quick=args.quick
        )

        # 保存配置文件
        config_path = save_config(config, dataset_id, args.output_dir)
        logger.info(f"配置文件已保存: {config_path}")

        # 运行实验
        success, return_code = run_experiment(
            config_path,
            dataset_id,
            parallel=args.parallel,
            resume=args.resume
        )

        # 记录结果
        if success:
            results['successful'].append(dataset_id)
            results['results'][dataset_id] = {
                'status': 'success',
                'return_code': return_code
            }
            logger.info(f"✓ 数据集 {dataset_id} 完成")
        else:
            results['failed'].append(dataset_id)
            results['results'][dataset_id] = {
                'status': 'failed',
                'return_code': return_code
            }
            logger.error(f"✗ 数据集 {dataset_id} 失败")

    # 完成汇总
    results['end_time'] = datetime.now()
    results['duration'] = results['end_time'] - results['start_time']
    results['success_rate'] = len(results['successful']) / len(selected_ids) * 100

    # 保存结果汇总
    summary_path = os.path.join(args.output_dir, 'test_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    # 生成CSV报告
    csv_path = os.path.join(args.output_dir, 'test_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Dataset_ID,Status,Return_Code\n")
        for dataset_id in selected_ids:
            result = results['results'].get(dataset_id, {})
            status = result.get('status', 'unknown')
            return_code = result.get('return_code', -1)
            f.write(f"{dataset_id},{status},{return_code}\n")

    # 打印最终汇总
    logger.info("\n" + "="*50)
    logger.info("测试完成汇总")
    logger.info(f"开始时间: {results['start_time']}")
    logger.info(f"结束时间: {results['end_time']}")
    logger.info(f"总耗时: {results['duration']}")
    logger.info(f"总数据集: {results['total_datasets']}")
    logger.info(f"成功: {len(results['successful'])}")
    logger.info(f"失败: {len(results['failed'])}")
    logger.info(f"成功率: {results['success_rate']:.1f}%")

    if results['failed']:
        logger.info(f"失败的数据集: {results['failed']}")

    logger.info(f"详细结果保存在: {args.output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()