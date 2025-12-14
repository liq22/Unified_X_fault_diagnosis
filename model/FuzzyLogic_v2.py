"""
Fuzzy Logic Network v2 for Unified Fault Diagnosis Framework

改进版本：
1. 修复硬编码维度问题
2. 增强特征融合机制
3. 扩展模糊规则库
4. 添加训练稳定性机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np

# 导入现有的基础类与特征提取器
from .Signal_processing import (
    SignalProcessingBase,
    SignalProcessingModuleDict,
    FFTSignalProcessing,
    HilbertTransform,
    WaveFilters,
    Identity,
)
from .FeatureExtractor import FeatureExtractor


class FuzzyMembershipFunction(nn.Module):
    """
    改进的模糊隶属函数实现
    """

    def __init__(self, num_features: int, num_membership_functions: int = 3):
        super(FuzzyMembershipFunction, self).__init__()
        self.num_features = num_features
        self.num_membership_functions = num_membership_functions

        # 改进的初始化策略
        self.centers = nn.Parameter(
            torch.randn(num_features, num_membership_functions) * 0.5
        )
        self.widths = nn.Parameter(
            torch.ones(num_features, num_membership_functions) * 0.3 + 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算模糊隶属度值
        """
        # x: (batch_size, num_features)
        x_expanded = x.unsqueeze(-1)  # (batch_size, num_features, 1)

        # 高斯隶属函数
        membership = torch.exp(
            -((x_expanded - self.centers) ** 2) / (2 * self.widths ** 2)
        )

        return membership  # (batch_size, num_features, num_membership_functions)


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    """

    def __init__(self, deep_feature_dim: int, stat_feature_dim: int, output_dim: int):
        super(AdaptiveFeatureFusion, self).__init__()
        self.deep_feature_dim = deep_feature_dim
        self.stat_feature_dim = stat_feature_dim
        self.output_dim = output_dim

        # 特征变换网络
        self.deep_transform = nn.Sequential(
            nn.Linear(deep_feature_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.stat_transform = nn.Sequential(
            nn.Linear(stat_feature_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, deep_features: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        """
        自适应融合深度特征和统计特征
        """
        # 特征变换
        deep_transformed = self.deep_transform(deep_features)  # (batch, output_dim//2)
        stat_transformed = self.stat_transform(stat_features)  # (batch, output_dim//2)

        # 拼接特征
        combined = torch.cat([deep_transformed, stat_transformed], dim=1)  # (batch, output_dim)

        # 计算注意力权重
        weights = self.attention(combined)  # (batch, 2)

        # 加权融合
        weighted_deep = deep_transformed * weights[:, 0:1]
        weighted_stat = stat_transformed * weights[:, 1:2]

        return weighted_deep + weighted_stat


class FuzzyRule(nn.Module):
    """
    扩展的模糊规则引擎
    """

    def __init__(self, num_features: int, num_rules: int = 50, num_classes: int = 5):
        super(FuzzyRule, self).__init__()
        self.num_features = num_features
        self.num_rules = num_rules
        self.num_classes = num_classes

        # 可学习的规则权重
        self.rule_weights = nn.Parameter(torch.ones(num_rules))

        # 规则前件（特征选择）
        self.rule_antecedents = nn.Parameter(
            torch.randn(num_rules, num_features) * 0.1
        )

        # 规则后件（类别权重）
        self.rule_consequents = nn.Parameter(
            torch.randn(num_rules, num_classes) * 0.1
        )

    def forward(self, membership_values: torch.Tensor) -> tuple:
        """
        模糊推理
        Args:
            membership_values: (batch_size, num_features, num_membership_functions)
        Returns:
            rule_outputs: (batch_size, num_rules, num_classes)
            rule_strengths: (batch_size, num_rules)
        """
        batch_size = membership_values.size(0)

        # 计算规则激活强度
        # 使用最大隶属度作为特征激活度
        feature_activation = torch.max(membership_values, dim=2)[0]  # (batch, num_features)

        # 规则前件匹配度
        rule_strengths = torch.zeros(batch_size, self.num_rules, device=membership_values.device)

        for i in range(self.num_rules):
            # 使用点积计算特征匹配度
            match = torch.sum(feature_activation * torch.abs(self.rule_antecedents[i]), dim=1)
            rule_strengths[:, i] = match

        # 归一化规则强度
        rule_strengths = torch.softmax(rule_strengths, dim=1)

        # 应用规则权重
        weighted_strengths = rule_strengths * torch.abs(self.rule_weights)

        # 规则输出
        rule_outputs = torch.zeros(batch_size, self.num_rules, self.num_classes, device=membership_values.device)

        for i in range(self.num_rules):
            rule_outputs[:, i] = weighted_strengths[:, i:i+1] * torch.abs(self.rule_consequents[i])

        return rule_outputs, rule_strengths


class FuzzyLogicNetworkV2(nn.Module):
    """
    改进的模糊逻辑网络
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        super(FuzzyLogicNetworkV2, self).__init__()

        # 基本参数
        self.input_dim = getattr(args, 'in_dim', 4096)
        self.in_channels = getattr(args, 'in_channels', 2)
        self.out_channels = getattr(args, 'out_channels', 3)
        self.scale = getattr(args, 'scale', 4)
        self.num_classes = getattr(args, 'num_classes', 5)
        self.skip_connection = getattr(args, 'skip_connection', True)

        # 信号处理层（与原版保持一致）
        from .TSPN import SignalProcessingLayer

        self.signal_processing_layers = nn.ModuleList()
        for i in range(4):
            layer_config = getattr(args, f'layer{i+1}', ['I', 'WF', 'I'])
            module_dict = SignalProcessingModuleDict({})

            for module_name in layer_config:
                if module_name == 'I':
                    module_dict[module_name] = Identity(args)
                elif module_name == 'WF':
                    module_dict[module_name] = WaveFilters(args)
                elif module_name == 'HT':
                    module_dict[module_name] = HilbertTransform(args)
                elif module_name == 'FFT':
                    module_dict[module_name] = FFTSignalProcessing(args)
                else:
                    module_dict[module_name] = Identity(args)

            in_ch = self.in_channels if i == 0 else self.out_channels
            out_ch = self.out_channels

            self.signal_processing_layers.append(
                SignalProcessingLayer(module_dict, in_ch, out_ch, self.skip_connection)
            )

        # 统计特征提取器
        self.feature_extractor = FeatureExtractor()

        # 深度特征提取器
        deep_feature_dim = self.out_channels * self.input_dim // 16
        self.feature_extractor_deep = nn.Sequential(
            nn.Linear(deep_feature_dim, deep_feature_dim // 2),
            nn.LayerNorm(deep_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 自适应特征融合
        stat_feature_dim = 13  # 统计特征维度
        fuzzy_feature_dim = 64  # 模糊特征维度

        self.feature_fusion = AdaptiveFeatureFusion(
            deep_feature_dim=deep_feature_dim // 2,
            stat_feature_dim=stat_feature_dim,
            output_dim=fuzzy_feature_dim
        )

        # 模糊隶属函数
        self.fuzzy_membership = FuzzyMembershipFunction(
            num_features=fuzzy_feature_dim,
            num_membership_functions=3  # Low, Medium, High
        )

        # 扩展的模糊规则库
        self.fuzzy_rules = FuzzyRule(
            num_features=fuzzy_feature_dim,
            num_rules=50,  # 从20增加到50
            num_classes=self.num_classes
        )

        # 输出层
        self.defuzzification = nn.Sequential(
            nn.Linear(self.num_classes, self.num_classes),
            nn.LayerNorm(self.num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 输入处理
        if x.dim() == 3:
            x = x.transpose(1, 2)

        # 信号处理
        for layer in self.signal_processing_layers:
            x = layer(x)

        # 池化和展平
        x_pooled = F.adaptive_avg_pool1d(x, self.input_dim // 16)
        x_flat = x_pooled.view(x_pooled.size(0), -1)

        # 深度特征提取
        deep_features = self.feature_extractor_deep(x_flat)

        # 统计特征提取
        stat_features = self.feature_extractor(x.transpose(1, 2))

        # 自适应特征融合
        fused_features = self.feature_fusion(deep_features, stat_features)

        # 模糊化
        membership_values = self.fuzzy_membership(fused_features)

        # 模糊推理
        rule_outputs, rule_strengths = self.fuzzy_rules(membership_values)

        # 规则聚合
        final_output = torch.sum(rule_outputs, dim=1)

        # 解模糊化
        logits = self.defuzzification(final_output)

        return logits

    def get_rule_explanations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取模糊规则解释
        """
        with torch.no_grad():
            # 前向传播获取中间结果
            if x.dim() == 3:
                x = x.transpose(1, 2)

            for layer in self.signal_processing_layers:
                x = layer(x)

            x_pooled = F.adaptive_avg_pool1d(x, self.input_dim // 16)
            x_flat = x_pooled.view(x_pooled.size(0), -1)
            deep_features = self.feature_extractor_deep(x_flat)
            stat_features = self.feature_extractor(x.transpose(1, 2))
            fused_features = self.feature_fusion(deep_features, stat_features)
            membership_values = self.fuzzy_membership(fused_features)
            rule_outputs, rule_strengths = self.fuzzy_rules(membership_values)

            return {
                'membership_values': membership_values,
                'rule_strengths': rule_strengths,
                'rule_outputs': rule_outputs,
                'fused_features': fused_features,
                'deep_features': deep_features,
                'stat_features': stat_features
            }


def create_model(signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any) -> FuzzyLogicNetworkV2:
    """
    创建改进的模糊逻辑网络模型
    """
    return FuzzyLogicNetworkV2(signal_processing_modules, feature_extractor_modules, args)