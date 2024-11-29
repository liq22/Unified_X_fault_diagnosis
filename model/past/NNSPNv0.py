
from scipy import optimize
import torch 
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool1d(1) # B,C,L -> B,C,1
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.se=nn.Sequential(
            nn.Conv1d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv1d(channel//reduction,channel,1,bias=False)
        )
        # self.sigmoid=nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        # output=self.sigmoid(max_out+avg_out)
        output = self.softmax(max_out+avg_out)
        return output
                    
class TimeAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv1d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class SPAttention(nn.Module):

    def __init__(self, channel=512,reduction=8,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        # self.sa=TimeAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, l, c = x.size()
        x = rearrange(x, 'b l c -> b c l')
        residual=x
        self.gate = self.ca(x)
        out=x*self.gate
        # out=out*self.sa(out)
        res = out + residual

        return rearrange(res, 'b c l -> b l c')


# class SPAttention(nn.Module):

#     def __init__(self, channel=512,reduction=4):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, l, c = x.size()
#         x = rearrange(x, 'b l c -> b c l')
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c,1)
#         res = x * y
#         return rearrange(res, 'b c l -> b l c')

# class SPAttention(nn.Module):

#     def __init__(self, channel=512,reduction=8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)

#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#         self.k = lambda x: (((x - torch.mean(x, dim=-1, keepdim=True)) ** 4).mean(dim=-1, keepdim=True)) / ((torch.var(x, dim=-1, keepdim=True) ** 2) + 1e-6)
#         self.norm = CustomBatchNorm(channel)

#     def forward(self, x):
#         b, l, c = x.size()

#         x = rearrange(x, 'b l c -> b c l')
#         # y_m = self.avg_pool(x).view(b, c) # normlized feature attention
#         y_k = self.k(x).squeeze(-1) 

#         y_k = self.norm(y_k)

#         y_k = y_k

#         y = self.fc(y_k).unsqueeze(-1)
#         res = x * y
#         return rearrange(res, 'b c l -> b l c')

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=0.1):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(1,num_features))
        self.register_buffer('running_var', torch.ones(1,num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.eps) * self.running_mean + self.eps * mean
            self.running_var = (1 - self.eps) * self.running_var + self.eps * var
            out = (x - mean) / (var.sqrt() + self.eps)
        else:
            out = (x - self.running_mean) / (self.running_var.sqrt() + self.eps)
        return out

class SignalProcessingLayer(nn.Module):
    # TODO op first then weight connection -> attention
    def __init__(self, signal_processing_modules, input_channels, output_channels,num_heads = 4, skip_connection=True):
        super(SignalProcessingLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(input_channels)
        self.weight_connection = nn.Linear(input_channels, output_channels)
        self.channel_attention = SPAttention(channel = output_channels,reduction=4)
        # self.attention = nn.MultiheadAttention(embed_dim=output_channels, num_heads=num_heads, batch_first=True)
        self.signal_processing_modules = signal_processing_modules
        self.module_num = len(signal_processing_modules)
        self.temperature = 0.1
        
        if skip_connection:
            self.skip_connection = nn.Linear(input_channels, output_channels)
    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        # 通过线性层
        
        self.weight_connection.weight.data = F.softmax((1.0 / self.temperature) *
                                                       self.weight_connection.weight.data, dim=0)
        x = self.weight_connection(normed_x)
        x = self.channel_attention(x)

        # attn_output, _ = self.attention(x_projected, x_projected, x_projected)
        

        # 按模块数拆分
        splits = torch.split(x, x.size(2) // self.module_num, dim=2)

        # 通过模块计算
        outputs = []
        for module, split in zip(self.signal_processing_modules.values(), splits):
            outputs.append(module(split))
        x = torch.cat(outputs, dim=2)
        # 添加skip connection
        if hasattr(self, 'skip_connection'):
            # self.skip_connection.weight.data = F.softmax((1.0 / self.temperature) *
            #                                             self.skip_connection.weight.data, dim=0)
            x = x + self.skip_connection(normed_x)
        return x
    
class FeatureExtractorlayer(nn.Module):
    def __init__(self, feature_extractor_modules,in_channels=1, out_channels=1):
        super(FeatureExtractorlayer, self).__init__()
        self.weight_connection = nn.Linear(in_channels, out_channels)
        self.feature_extractor_modules = feature_extractor_modules
        # self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        out_channels = int(len(feature_extractor_modules) * out_channels)
        
        self.norm = CustomBatchNorm(out_channels)
        self.FEAttention = SPAttention(channel = out_channels,reduction=4)
        # self.temperature = 1
    # def norm(self,x): # feature normalization
    #     mean = x.mean(dim = 0,keepdim = True)
    #     std = x.std(dim = 0,keepdim = True)
    #     out = (x-mean)/(std + 1e-10)
    #     return out
           
    def forward(self, x):
        # TODO # self.weight_connection.weight.data = F.softmax((1.0 / self.temperature) *
        #                                                self.weight_connection.weight.data, dim=0)
        x = self.weight_connection(x)
        x = rearrange(x, 'b l c -> b c l')
        outputs = []
        for module in self.feature_extractor_modules.values():
            outputs.append(module(x))
        res = torch.cat(outputs, dim=1).squeeze() # B,C
        res = self.norm(res).unsqueeze(1)
        res = self.FEAttention(res)
        return res

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes): # TODO logic
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            
        )
        # self.clf = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.clf(x)

 

class NN_Signal_Processing_Network(nn.Module):
    def __init__(self, signal_processing_modules,feature_extractor, args):
        super(NN_Signal_Processing_Network, self).__init__()
        self.layer_num = len(signal_processing_modules)
        self.signal_processing_modules = signal_processing_modules
        self.feature_extractor_modules = feature_extractor
        self.args = args

        self.init_signal_processing_layers()
        self.init_feature_extractor_layers()
        self.init_classifier()

    def init_signal_processing_layers(self):
        print('# build signal processing layers')
        in_channels = self.args.in_channels
        out_channels = int(self.args.out_channels * self.args.scale)

        self.signal_processing_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.signal_processing_layers.append(SignalProcessingLayer(self.signal_processing_modules[i],
                                                                       in_channels,
                                                                         out_channels,
                                                                         self.args.skip_connection).to(self.args.device))
            in_channels = out_channels 
            assert out_channels % self.signal_processing_layers[i].module_num == 0 
            # out_channels = int(out_channels * self.args.scale)
        self.channel_for_feature = out_channels # // self.args.scale

    def init_feature_extractor_layers(self):
        print('# build feature extractor layers')
        self.feature_extractor_layers = FeatureExtractorlayer(self.feature_extractor_modules,
                                                              in_channels = self.channel_for_feature,
                                                              out_channels = self.channel_for_feature).to(self.args.device)
        len_feature = len(self.feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature


    def init_classifier(self):
        print('# build classifier')
        self.clf = Classifier(self.channel_for_classifier, self.args.num_classes).to(self.args.device)

    def forward(self, x):

        for layer in self.signal_processing_layers:
            x = layer(x)
        x = self.feature_extractor_layers(x)
        
        x = self.clf(x)
        return x
    


    
if __name__ == '__main__':
    from config import args # for debug model
    from config import signal_processing_modules,feature_extractor_modules
    import torchinfo
    net = NN_Signal_Processing_Network(signal_processing_modules,feature_extractor_modules, args)
    # net = Transparent_Signal_Processing_KAN(signal_processing_modules,feature_extractor_modules, args)
    x = torch.randn(64, 512, 2).cuda()
    y = net(x)
    print(y.shape)
    
    net_summaary= torchinfo.summary(net.cuda(),(3,512,2),device = "cuda")
    print(net_summaary)
    with open(f'save/NNSPN2.txt','w') as f:
        f.write(str(net_summaary))      
        
    # args = {
    #     'in_channels': 2,
    #     'out_channels': 6,  # 这应该与您网络设计中的输出通道数一致
    #     'scale': 1,  # 根据您的模型具体需要调整
    #     'num_classes': 5,  # 假设有5个类别
    #     'learning_rate': 0.001,
    #     'num_epochs': 100
    # }
    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    # optimizer = optim.Adam(net.parameters(), lr=args['learning_rate'])
    # loss_fn = nn.CrossEntropyLoss()

    # # 模拟训练数据和标签
    # def generate_random_data(batch_size, length, input_channels):
    #     data = torch.randn(batch_size, length, input_channels)
    #     labels = torch.randint(0, args['num_classes'], (batch_size,))
    #     return data, labels

    # # 训练循环
    # for epoch in range(args['num_epochs']):
    #     net.train()
    #     data, labels = generate_random_data(32, 4096, args['in_channels'])  # 假设序列长度为100
    #     data, labels = data.cuda(), labels.cuda()
    #     optimizer.zero_grad()
    #     outputs = net(data)
    #     loss = loss_fn(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
        
    #     if epoch % 10 == 0:  # 每10轮输出一次损失
    #         print(f"Epoch {epoch}, Loss: {loss.item()}")    
    
    
    ####################################################
# def test_backward():
#     from config import args
#     from config import signal_processing_modules,feature_extractor_modules

#     # 创建网络
#     net = Transparent_Signal_Processing_Network(signal_processing_modules,feature_extractor_modules, args).cuda()
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
#     # 创建随机输入和目标输出
#     x = torch.randn(2, 4096, 2).cuda()
#     target = torch.randint(0, args.num_classes, (2,)).cuda()

#     # 计算网络的输出
#     output = net(x)

#     # 计算损失
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # 检查梯度
#     for name, param in net.named_parameters():
#         assert param.grad is not None, f'No gradient for {name} in backward'

#     print('Backward pass test passed.')

# if __name__ == '__main__':
#     test_backward()