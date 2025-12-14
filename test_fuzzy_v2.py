#!/usr/bin/env python3
"""
快速测试FuzzyLogicV2模型
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.FuzzyLogic_v2 import create_model
from model.Signal_processing import SignalProcessingModuleDict
from configs.config import parse_arguments

def test_fuzzy_logic_v2():
    """测试FuzzyLogicV2模型的基本功能"""
    print("=" * 50)
    print("测试FuzzyLogicV2模型")
    print("=" * 50)

    # 创建模拟参数
    args = argparse.Namespace()
    args.in_dim = 4096
    args.out_dim = 4096
    args.in_channels = 2
    args.out_channels = 3
    args.scale = 4
    args.num_classes = 5
    args.skip_connection = True
    args.layer1 = ['I', 'WF', 'I']
    args.layer2 = ['I', 'WF', 'I']
    args.layer3 = ['I', 'WF', 'I']
    args.layer4 = ['I', 'WF', 'I']
    args.device = 'cpu'

    # 创建模型
    print("\n1. 创建FuzzyLogicV2模型...")
    try:
        signal_processing_modules = {}
        feature_extractor_modules = {}
        model = create_model(signal_processing_modules, feature_extractor_modules, args)
        print(f"   ✓ 模型创建成功")
        print(f"   ✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ 模型创建失败: {e}")
        return False

    # 创建测试数据
    print("\n2. 创建测试数据...")
    batch_size = 8
    seq_len = 4096
    channels = 2

    x = torch.randn(batch_size, seq_len, channels)
    print(f"   ✓ 输入数据形状: {x.shape}")

    # 前向传播测试
    print("\n3. 前向传播测试...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ 前向传播成功")
        print(f"   ✓ 输出形状: {output.shape}")
        print(f"   ✓ 输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试解释功能
    print("\n4. 测试可解释性功能...")
    try:
        explanations = model.get_rule_explanations(x)
        print(f"   ✓ 可解释性提取成功")
        for key, value in explanations.items():
            print(f"   - {key}: {value.shape}")
    except Exception as e:
        print(f"   ✗ 可解释性提取失败: {e}")
        return False

    # 测试训练模式
    print("\n5. 测试训练模式...")
    try:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 模拟训练步骤
        labels = torch.randint(0, args.num_classes, (batch_size,))
        output = model(x)
        loss = criterion(output, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        optimizer.step()

        print(f"   ✓ 训练步骤成功")
        print(f"   ✓ 损失值: {loss.item():.4f}")
        print(f"   ✓ 梯度计算: {'正常' if has_grad else '无梯度'}")
    except Exception as e:
        print(f"   ✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ 所有测试通过！FuzzyLogicV2模型工作正常")
    print("=" * 50)
    return True

def test_feature_fusion():
    """测试自适应特征融合模块"""
    print("\n" + "=" * 50)
    print("测试自适应特征融合模块")
    print("=" * 50)

    from model.FuzzyLogic_v2 import AdaptiveFeatureFusion

    batch_size = 8
    deep_dim = 256
    stat_dim = 13
    output_dim = 64

    fusion = AdaptiveFeatureFusion(deep_dim, stat_dim, output_dim)

    # 创建测试数据
    deep_features = torch.randn(batch_size, deep_dim)
    stat_features = torch.randn(batch_size, stat_dim)

    # 前向传播
    fused = fusion(deep_features, stat_features)

    print(f"深度特征形状: {deep_features.shape}")
    print(f"统计特征形状: {stat_features.shape}")
    print(f"融合特征形状: {fused.shape}")
    print(f"✓ 特征融合测试成功")

    return True

if __name__ == "__main__":
    # 测试主模型
    success = test_fuzzy_logic_v2()

    # 测试特征融合
    fusion_success = test_feature_fusion()

    if success and fusion_success:
        print("\n🎉 所有测试通过！模型已准备好进行训练。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查模型实现。")
        sys.exit(1)