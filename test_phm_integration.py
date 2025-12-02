#!/usr/bin/env python3
"""
PHM-Vibench数据集集成验证脚本
测试数据加载、模型兼容性和基本训练流程
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from data.data_provider import get_data
from model.TSPN import Transparent_Signal_Processing_Network as TSPN
from trainer.trainer_basic import Basic_plmodel
import pytorch_lightning as pl

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建args对象
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if key == 'args':
                    # 处理嵌套的args配置
                    for sub_key, sub_value in value.items():
                        setattr(self, sub_key, sub_value)
                else:
                    setattr(self, key, value)

    return Args(config)

def test_data_loading(config_path):
    """测试数据加载功能"""
    print("=" * 50)
    print("测试数据加载...")
    print("=" * 50)

    try:
        # 加载配置
        args = load_config(config_path)
        print(f"配置文件: {config_path}")
        print(f"数据集任务: {args.dataset_task}")
        print(f"数据目录: {args.vbench_config['data_dir']}")

        # 获取数据加载器
        train_loader, val_loader, test_loader = get_data(args)

        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")

        # 测试一个批次
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"批次 {batch_idx}:")
            print(f"  数据形状: {data.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  标签类型: {labels.dtype}")
            print(f"  数据范围: [{data.min():.3f}, {data.max():.3f}]")
            print(f"  唯一标签: {torch.unique(labels)}")
            break

        print("✅ 数据加载测试成功!")
        return True, (train_loader, val_loader, test_loader)

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_compatibility(config_path, data_loaders):
    """测试模型兼容性"""
    print("=" * 50)
    print("测试模型兼容性...")
    print("=" * 50)

    try:
        # 加载配置
        args = load_config(config_path)
        train_loader, _, _ = data_loaders

        # 获取一个批次的数据
        data, labels = next(iter(train_loader))

        # 创建模型
        print(f"创建模型: {args.model}")

        # 从数据中自动推断类别数
        num_classes = len(torch.unique(labels))
        args.num_classes = num_classes
        print(f"自动推断类别数: {num_classes}")

        if args.model == 'TSPN':
            model = TSPN(args)
        else:
            print(f"暂未支持模型: {args.model}")
            return False

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        print("测试前向传播...")
        with torch.no_grad():
            output = model(data)
            print(f"输出形状: {output.shape}")
            print(f"预期形状: [{data.shape[0]}, {num_classes}]")

            # 检查输出形状是否正确
            if output.shape[0] == data.shape[0] and output.shape[1] == num_classes:
                print("✅ 模型前向传播测试成功!")
            else:
                print("❌ 输出形状不匹配")
                return False

        # 测试完整训练步骤
        print("测试训练步骤...")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        print(f"损失值: {loss.item():.4f}")
        print("✅ 训练步骤测试成功!")

        return True

    except Exception as e:
        print(f"❌ 模型兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightning_integration(config_path):
    """测试PyTorch Lightning集成"""
    print("=" * 50)
    print("测试PyTorch Lightning集成...")
    print("=" * 50)

    try:
        # 加载配置
        args = load_config(config_path)

        # 获取数据加载器
        train_loader, val_loader, test_loader = get_data(args)

        # 获取一个批次推断类别数
        data, labels = next(iter(train_loader))
        args.num_classes = len(torch.unique(labels))

        # 创建Lightning模型
        print("创建Lightning模型...")
        pl_model = Basic_plmodel(args)

        # 测试一个训练步骤
        print("测试训练步骤...")
        loss = pl_model.training_step((data, labels), 0)
        print(f"训练损失: {loss.item():.4f}")

        # 测试验证步骤
        print("测试验证步骤...")
        val_loss = pl_model.validation_step((data, labels), 0)
        print(f"验证损失: {val_loss.item():.4f}")

        print("✅ PyTorch Lightning集成测试成功!")
        return True

    except Exception as e:
        print(f"❌ PyTorch Lightning集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("PHM-Vibench数据集集成验证")
    print("=" * 50)

    # 测试配置文件列表
    config_files = [
        "configs/PHM_Vibench/config_TSPN.yaml",
        "configs/PHM_Vibench/config_TKAN.yaml",
        "configs/PHM_Vibench/config_NNSPN.yaml",
    ]

    results = {}

    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"⚠️  配置文件不存在: {config_file}")
            continue

        print(f"\n📋 测试配置: {config_file}")

        # 1. 测试数据加载
        data_success, data_loaders = test_data_loading(config_file)

        if data_success and data_loaders:
            # 2. 测试模型兼容性
            model_success = test_model_compatibility(config_file, data_loaders)

            # 3. 测试Lightning集成
            lightning_success = test_lightning_integration(config_file)

            results[config_file] = {
                'data_loading': data_success,
                'model_compatibility': model_success,
                'lightning_integration': lightning_success
            }
        else:
            results[config_file] = {
                'data_loading': False,
                'model_compatibility': False,
                'lightning_integration': False
            }

    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)

    all_success = True
    for config, result in results.items():
        config_name = os.path.basename(config)
        status = "✅ 全部通过" if all(result.values()) else "❌ 部分失败"
        print(f"{config_name}: {status}")
        if not all(result.values()):
            for test, success in result.items():
                print(f"  {test}: {'✅' if success else '❌'}")
        all_success = all_success and all(result.values())

    if all_success:
        print("\n🎉 所有测试通过！PHM-Vibench数据集已成功集成到统一基线框架。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复。")

if __name__ == "__main__":
    main()