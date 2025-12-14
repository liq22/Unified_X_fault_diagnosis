"""
简单的特征提取器，用于FuzzyLogic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFeatureExtractor(nn.Module):
    """
    提取13种统计特征的简化版本
    """

    def __init__(self):
        super(SimpleFeatureExtractor, self).__init__()

    def forward(self, x):
        """
        提取统计特征

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, channels) 或 (batch_size, channels, seq_len)

        Returns:
            features: 统计特征，形状为 (batch_size, 13)
        """
        # 确保输入形状正确
        if x.dim() == 3 and x.size(1) == 2:  # (batch, channels, seq_len)
            x = x.transpose(1, 2)  # 转换为 (batch, seq_len, channels)

        # 合并通道维度
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # (batch, seq_len * channels)

        # 计算各种统计特征
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        max_val = torch.max(x, dim=1, keepdim=True)[0]
        min_val = torch.min(x, dim=1, keepdim=True)[0]
        abs_mean = torch.mean(torch.abs(x), dim=1, keepdim=True)

        # RMS (均方根)
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True))

        # 峰值因子
        peak_factor = max_val / (rms + 1e-8)

        # 偏度
        centered = x - mean
        skewness = torch.mean(centered ** 3, dim=1, keepdim=True) / (std ** 3 + 1e-8)

        # 峭度
        kurtosis = torch.mean(centered ** 4, dim=1, keepdim=True) / (std ** 4 + 1e-8)

        # 波形因子
        shape_factor = rms / (abs_mean + 1e-8)

        # 间隙因子
        clearance_factor = torch.sum(torch.square(torch.sqrt(torch.abs(centered))), dim=1, keepdim=True) / (rms ** 3 + 1e-8)

        # 拼接所有特征
        features = torch.cat([
            mean.squeeze(1),  # 1
            std.squeeze(1),   # 2
            var.squeeze(1),   # 3
            max_val.squeeze(1),  # 4
            min_val.squeeze(1),  # 5
            abs_mean.squeeze(1), # 6
            kurtosis.squeeze(1),  # 7
            rms.squeeze(1),       # 8
            peak_factor.squeeze(1), # 9
            skewness.squeeze(1),   # 10
            shape_factor.squeeze(1), # 11
            clearance_factor.squeeze(1), # 12
            # 额外添加一个特征，凑齐13个
            torch.mean(torch.square(x), dim=1)  # 13
        ], dim=1)

        return features  # (batch_size, 13)


# 创建别名以保持兼容性
FeatureExtractor = SimpleFeatureExtractor