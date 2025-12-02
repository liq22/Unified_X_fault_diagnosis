#!/usr/bin/env python3
"""
简化的PHM-Vibench测试脚本
"""

import os
import sys
import torch
import yaml
import h5py
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from data.vbench_dataset import VbenchDataset
from model.TSPN import Transparent_Signal_Processing_Network as TSPN

def test_direct_data_access():
    """直接测试数据访问"""
    print("=" * 50)
    print("直接测试PHM-Vibench数据访问...")
    print("=" * 50)

    try:
        # 数据路径
        data_dir = Path("/home/user/data/PHMbenchdata/PHM-Vibench")
        metadata_file = data_dir / "metadata_6_11.xlsx"

        # 读取元数据
        print(f"读取元数据: {metadata_file}")
        metadata = pd.read_excel(metadata_file)

        # 筛选CWRU数据集
        cwru_data = metadata[metadata['Dataset_id'] == 1]
        cwru_data = cwru_data.dropna(subset=['Label']).reset_index(drop=True)

        print(f"CWRU数据集样本数: {len(cwru_data)}")
        print(f"标签分布: {cwru_data['Label'].value_counts().to_dict()}")

        # 读取H5文件
        h5_file = data_dir / "RM_001_CWRU.h5"
        print(f"打开H5文件: {h5_file}")

        with h5py.File(h5_file, 'r') as h5:
            # 获取前几个样本ID
            sample_ids = list(h5.keys())[:5]
            print(f"前5个样本ID: {sample_ids}")

            # 读取第一个样本
            sample_id = sample_ids[0]
            sample_data = h5[sample_id][...]
            print(f"样本 {sample_id} 形状: {sample_data.shape}")
            print(f"数据类型: {sample_data.dtype}")
            print(f"数据范围: [{sample_data.min():.3f}, {sample_data.max():.3f}]")

            # 提取窗口
            window_length = 2048
            if sample_data.shape[0] >= window_length:
                window = sample_data[:window_length]
                # 转换为 [C, L] 格式
                if window.ndim == 2:
                    window_tensor = torch.from_numpy(window.T).float()
                else:
                    window_tensor = torch.from_numpy(window).float().unsqueeze(0)

                print(f"窗口张量形状: {window_tensor.shape}")
                print("✅ 数据直接访问测试成功!")
                return True, window_tensor.shape
            else:
                print(f"样本长度不足: {sample_data.shape[0]} < {window_length}")
                return False, None

    except Exception as e:
        print(f"❌ 直接数据访问测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_vbench_dataset():
    """测试VbenchDataset类"""
    print("=" * 50)
    print("测试VbenchDataset类...")
    print("=" * 50)

    try:
        # 创建简单的配置对象
        class Args:
            def __init__(self):
                self.vbench_config = {
                    'data_dir': "/home/user/data/PHMbenchdata/PHM-Vibench",
                    'metadata_file': "metadata_6_11.xlsx",
                    'task_type': "fault_diagnosis",
                    'target_column': "Label",
                    'dataset_ids': [1],  # 仅CWRU
                    'sampling_config': {
                        'method': 'random'  # 使用随机采样
                    },
                    'window_config': {
                        'window_length': 2048,
                        'stride': 512,
                        'padding_mode': "reflect",
                        'min_window_ratio': 0.5
                    },
                    'domain_config': {
                        'leave_one_domain_out': False
                    },
                    'split_config': {
                        'train_ratio': 0.7,
                        'val_ratio': 0.15,
                        'test_ratio': 0.15,
                        'stratify_by': "Label"
                    },
                    'augmentation': {
                        'enable': False
                    }
                }
                self.seed = 17

        args = Args()

        # 创建数据集
        print("创建训练数据集...")
        train_dataset = VbenchDataset(args, flag='train')
        print(f"训练集大小: {len(train_dataset)}")

        # 获取一个样本
        print("获取第一个样本...")
        data, label = train_dataset[0]
        print(f"数据形状: {data.shape}")
        print(f"标签: {label}")
        print(f"数据类型: {data.dtype}")
        print(f"标签类型: {type(label)}")

        print("✅ VbenchDataset测试成功!")
        return True, data.shape, len(train_dataset.get_num_classes() if hasattr(train_dataset, 'get_num_classes') else [0])

    except Exception as e:
        print(f"❌ VbenchDataset测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0

def test_model_creation():
    """测试模型创建"""
    print("=" * 50)
    print("测试模型创建...")
    print("=" * 50)

    try:
        # 创建配置
        class Args:
            def __init__(self):
                self.model = 'TSPN'
                self.num_classes = 4
                self.in_dim = 2048
                self.out_dim = 2048
                self.in_channels = 2
                self.out_channels = 3
                self.scale = 4
                self.skip_connection = True

        args = Args()

        # 创建模型
        model = TSPN(args)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        dummy_input = torch.randn(1, 2, 2048)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
            print("✅ 模型创建和前向传播测试成功!")

        return True, output.shape[1]

    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """主测试函数"""
    print("PHM-Vibench简化集成测试")
    print("=" * 50)

    # 1. 直接数据访问测试
    data_success, data_shape = test_direct_data_access()

    # 2. VbenchDataset测试
    dataset_success, dataset_shape, num_classes = test_vbench_dataset()

    # 3. 模型创建测试
    model_success, output_classes = test_model_creation()

    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)

    print(f"直接数据访问: {'✅' if data_success else '❌'}")
    print(f"VbenchDataset: {'✅' if dataset_success else '❌'}")
    print(f"模型创建: {'✅' if model_success else '❌'}")

    if data_success and dataset_success and model_success:
        print("\n🎉 核心功能测试通过！PHM-Vibench数据集基础集成成功。")
        print("注意：需要修复采样器问题以支持完整的数据加载流程。")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()