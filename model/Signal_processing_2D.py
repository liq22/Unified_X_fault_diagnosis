from numpy import identity
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import gc

from einops import rearrange
# sys.path.append('/home/user/LQ/B_Signal/Transparent_information_fusion/model')


class BaseTF(nn.Module):
    """基础时频变换类"""
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.fs = args.fs
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        
        self.Nw = args.window_length
        self.stride = args.stride
        
        
        self.CI_name = args.CI_name
        self.device = args.device
        self.to(self.device)
        
        # 注册高斯窗
        if args.window == 'hann':
            self.register_buffer('window', torch.hann_window(self.Nw).unsqueeze(0))
        elif args.window == 'hamming':
            self.register_buffer('window', torch.hamming_window(self.Nw).unsqueeze(0))
        elif args.window == 'rect':
            self.register_buffer('window', torch.ones(self.Nw).unsqueeze(0))
        elif args.window == 'gaussian':
            self.register_buffer('window', self._gauss_window(self.Nw).unsqueeze(0))
        
    def forward(self, x):
        # x should be B,L,C first
        raise NotImplementedError("This method should be implemented by subclass.")
        
    def _gauss_window(self, M, alpha=1.8):
        """高斯窗生成"""
        n = torch.arange(0, M) - (M - 1) / 2
        w = torch.exp(-0.5 * (alpha * n / ((M - 1)/2))**2)
        return w / w.sum()

    def prepare_signal(self, signal, stride):
        """信号预处理"""
        B, L, C = signal.shape
        signal = rearrange(signal, 'B L C -> B C L')
        
        pad_len = self.Nw // 2
        signal_padded = F.pad(signal, (pad_len, pad_len), "constant", 0)
        
        k = (L - stride) // stride
        Ta = torch.arange(k, device=signal.device).unsqueeze(1) * stride
        Tb = torch.arange(self.Nw, device=signal.device).unsqueeze(0)
        mother = Tb + Ta
        return signal_padded[:,:,mother], k
    
    def compute_CI(self, sub_TFRs):
        """计算调制指标"""
        if self.CI_name == 'Kurtosis':
            power2 = sub_TFRs ** 2
            power4 = sub_TFRs ** 4
            return power4.mean(dim=-1)/(power2.mean(dim=-1)**2 + 1e-12)
        elif self.CI_name == 'entropy':
            return (sub_TFRs**2 * torch.log(sub_TFRs**2 + 1e-12)).sum(dim=-1)

    def fuse_TFR(self, sub_TFRs, CI):
        """融合时频表示"""
        optimal_d = CI.argmax(dim=2)
        B, C, D, T, F = sub_TFRs.shape
        d_idx = rearrange(optimal_d, 'b c t -> b c () t ()').expand(B,C,1,T,F)
        res = torch.gather(sub_TFRs, dim=2, index=d_idx).squeeze(2)
        return rearrange(res, 'b c t f -> b t f c')
    
    def compute_TFRs(self, signal):
        raise NotImplementedError
    
    def forward(self, signal):
        TFRs = self.compute_TFRs(signal)
        CI = self.compute_CI(TFRs)
        return self.fuse_TFR(TFRs, CI)

class SignalProcessingBase(torch.nn.Module):
    def __init__(self, args):
        super(SignalProcessingBase, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.device = args.device
        self.to(self.device)


    def forward(self, x):
        # x should be B,L,C first
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device) 
        output = self.forward(test_input)
        assert output.shape == (2, self.out_dim, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_dim, self.out_channels)}"



class GlobalFilterOperator(nn.Module):
    def __init__(self, args):
        super(GlobalFilterOperator, self).__init__()
        # 定义可学习参数w，维度为(T, F)
        self.in_channels = args.scale 
        dim = (args.in_dim - args.stride) // args.stride
        self.w = nn.Parameter(torch.randn(dim, dim,1)) # B,T,F,C

    def forward(self, SBCT_output):
        """
        :input SBCT_output: (B, T, F, C)

        :return: (B, 1, T, F)
        """
        B, Time, Fre,C,  = SBCT_output.shape
        
        # 确保w的维度是(T, F)
        assert self.w.shape == (Time, Fre,C), "w的维度必须是(T, F)"
        
        # 计算TF_2 = σ(w * TF_1) + TF_1
        tf_1 = SBCT_output  # 提取出TF_1，维度为(B, T, F)
        tf_2 = F.silu(self.w * tf_1) + tf_1  # 使用silu作为阈值函数
        
        return tf_2

class SignalProcessingModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_dict):
        super(SignalProcessingModuleDict, self).__init__(module_dict)

    def forward(self, x, key):
        if key in self:
            return self[key](x)
        else:
            raise KeyError(f"No signal processing module found for key: {key}")
        
    def test_forward(self):
        for key in self.keys():
            self[key].test_forward()

class SBCT_NOP(BaseTF):

    """可伸缩基奇波变换"""
    def __init__(self, args):
        super(SBCT_NOP,self).__init__(args)
        # 初始化学习参数
        self.gamma = nn.Parameter((torch.rand(args.tf_in_channel,
                                              args.search_dim,
                                              args.order)-0.5)*torch.pi)
        self.order = args.order
        self.search_dim = args.search_dim
    def compute_TFRs(self, signal):
        """计算时频表示"""
        sig, k = self.prepare_signal(signal, self.stride)
        sig_windowed = (sig * self.window).to(torch.complex64)
        
        # 频率计算
        u = torch.arange(1, self.Nw+1, device=signal.device)/self.fs
        df = torch.arange(0, k, device=signal.device)/k * self.fs/2
        t_c = u[self.Nw//2]
        
        # 相位计算
        k_range = torch.arange(1, self.order+1, device=u.device).unsqueeze(0)
        u_t_c = (u - t_c).unsqueeze(1)
        power = u_t_c.pow(k_range + 1)
        
        slope = torch.tan(self.gamma)
        phase = u + torch.einsum('c d o,n o->c d n', slope, power)
        exponent = -1j * 2 * torch.pi * phase.unsqueeze(-1) * df.view(1,1,1,k)
        
        kernelb = torch.exp(exponent)
        
        # 内存清理
        del phase, exponent, power, u_t_c, k_range, slope
        gc.collect()
        
        sub_TFR_complex = torch.einsum('b c j n,c d n k->b c d j k', sig_windowed, kernelb)
        return sub_TFR_complex.abs()

# 4 ############################################# Identity module #######################################
class Identity(SignalProcessingBase):
    def __init__(self, args):
        super(Identity, self).__init__(args)
        self.name = "I"
    def forward(self, x):
        return x


if __name__ == "__main__":
    # 测试模块
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.in_dim = 1024
            self.out_dim = 1024
            self.in_channels = 2
            self.out_channels = 2
            self.fs = 2000
            self.window_length = 64
            self.stride = 4
            self.metric = 'kurtosis'
            self.CI_name = 'kurtosis'
            self.search_dim = 10
            self.order = 2
            self.window = 'gaussian'
            self.t_dim = 255
            self.f_dim = 255
            self.scale = 2            
            

    def test_sbct_nop():
        args = Args()
        torch.manual_seed(0)  # 设置随机种子
        model1 = SBCT_NOP(args).to(args.device)
        # torch.manual_seed(1)  # 设置不同的随机种子
        model2 = SBCT_NOP(args).to(args.device)
        input_data = torch.randn(2, args.in_dim, args.in_channels).to(args.device)
        output_data1 = model1(input_data)
        output_data2 = model2(input_data)
        print("Model 1 Gamma:", model1.gamma)
        print("Model 2 Gamma:", model2.gamma)
        print("Output 1 shape:", output_data1.shape)
        print("Output 2 shape:", output_data2.shape)
        
        filter = GlobalFilterOperator(args).to(args.device)
        output_data = filter(output_data1)
        print("Output shape:", output_data.shape)
        
    test_sbct_nop()
    args = Args()
    signal_module_1 = {
            "$SBCT$": SBCT_NOP(args),
             "$SBCT2$": SBCT_NOP(args),
        }

    signal_processing_modules = SignalProcessingModuleDict(signal_module_1)
    print(signal_processing_modules)
    
    x = torch.randn(2, 1024, 2)
    key = "$SBCT$"
    output = signal_processing_modules(x, key)
    print(output.shape)