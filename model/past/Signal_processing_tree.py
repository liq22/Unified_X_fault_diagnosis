import torch
import torch.nn as nn

class TreeNode(nn.Module):
    def __init__(self, in_channels, out_channels, signal_processing_module, skip_connection=True):
        super(TreeNode, self).__init__()
        self.norm = nn.InstanceNorm1d(in_channels)  # 根据实际情况选择合适的Norm层
        self.linear = nn.Linear(in_channels, out_channels)
        self.signal_processing_module = signal_processing_module
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_linear = nn.Linear(in_channels, out_channels)  # 调整维度以匹配主路径和skip路径

    
    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.linear(out)
        out = self.signal_processing_module(out)
        
        if self.skip_connection:
            identity = self.skip_linear(identity)
            out += identity  # 残差连接

        return out
