import math
import torch # .nn as nn
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


KERNEL_SIZE = 49 
FRE = 10 
DEVICE = 'cuda'
STRIDE = 1
T = torch.linspace(-KERNEL_SIZE/2,KERNEL_SIZE/2, KERNEL_SIZE).view(1,1,KERNEL_SIZE).to(DEVICE) # 暂定cuda 

def Morlet(t):
    C = pow(math.pi, 0.25)
    f = FRE
    w = 2 * math.pi * f    
    y = C * torch.exp(-torch.pow(t, 2) / 2) * torch.cos(w * t)
    return y

def Laplace(t):
    a = 0.08
    ep = 0.03
    tal = 0.1
    f = FRE
    w = 2 * math.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = a * torch.exp((-ep / (torch.sqrt(q))) * (w * (t - tal))) * (-torch.sin(w * (t - tal)))
    return y

class convlutional_operator(nn.Module):
    def __init__(self, kernel_op='conv_sin', dim=1, stride=STRIDE, kernel_size=KERNEL_SIZE, device='cuda', in_channels=1):
        super().__init__()
        self.affline = nn.InstanceNorm1d(num_features=dim, affine=True).to(device)
        op_dic = {'conv_sin': torch.sin,
                  'conv_sin2': lambda x: torch.sin(x ** 2),
                  'conv_exp': torch.exp,
                  'conv_exp2': lambda x: torch.exp(x ** 2),
                  'Morlet': Morlet,
                  'Laplace': Laplace}
        self.op = op_dic[kernel_op]
        self.stride = stride
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.t = torch.linspace(-math.pi / 2, math.pi / 2, kernel_size).view(1, 1, kernel_size).to(device)

    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        self.aff_t = self.affline(self.t)
        self.weight = self.op(self.aff_t).view(1, 1, -1).repeat(self.in_channels, 1, 1)
        conv = F.conv1d(x, self.weight, stride=self.stride, padding=(self.kernel_size - 1) // 2, dilation=1, groups=self.in_channels)
        conv = rearrange(conv, 'b c l -> b l c')
        return conv

class signal_filter_(nn.Module):
    def __init__(self, kernel_op='order1_MA', dim=1, stride=STRIDE, kernel_size=KERNEL_SIZE, device='cuda', in_channels=1):
        super().__init__()
        self.affline = nn.InstanceNorm1d(num_features=dim, affine=True).to(device)
        op_dic = {'order1_MA': torch.tensor([0.5, 0, 0.5]),
                  'order2_MA': torch.tensor([1 / 3, 1 / 3, 1 / 3]),
                  'order1_DF': torch.tensor([-1.0, 0, 1.0]),
                  'order2_DF': torch.tensor([-1.0, 2.0, -1.0])}
        self.weight = op_dic[kernel_op].view(1, 1, -1).to(device).repeat(in_channels, 1, 1)
        self.stride = stride
        self.kernel_size = 3
        self.in_channels = in_channels

    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        conv = F.conv1d(x, self.weight, stride=self.stride, padding=(self.kernel_size - 1) // 2, dilation=1, groups=self.in_channels)
        conv = rearrange(conv, 'b c l -> b l c')
        return conv