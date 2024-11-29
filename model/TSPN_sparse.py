
from scipy import optimize
import torch 
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math

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

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sparsity=0.9):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        # 初始化权重
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # 创建掩码
        self.register_buffer('mask', self.create_mask())

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def create_mask(self):
        # 根据稀疏度创建掩码
        mask = torch.rand_like(self.weight)
        threshold = torch.quantile(mask, self.sparsity)
        mask = (mask > threshold).float()
        return mask

    def forward(self, input):
        # 应用掩码以确保权重稀疏
        sparse_weight = self.weight * self.mask
        return F.linear(input, sparse_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparsity={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparsity
        )

class SignalProcessingLayer(nn.Module):
    def __init__(self, signal_processing_modules, input_channels, output_channels, skip_connection=True, temperature=0.1):
        super(SignalProcessingLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(input_channels)
        self.weight_connection = SparseLinear(input_channels, output_channels, bias=True, sparsity=0.9)
        self.signal_processing_modules = nn.ModuleList(signal_processing_modules)
        self.module_num = len(signal_processing_modules)
        self.temperature = temperature
        
        if skip_connection:
            self.skip_connection = SparseLinear(input_channels, output_channels, bias=True, sparsity=0.9)
        else:
            self.skip_connection = None

    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')

        # 通过线性层，并应用温度缩放后的 softmax
        weight = F.softmax(self.weight_connection.weight / self.temperature, dim=0)
        sparse_weight = weight * self.weight_connection.mask
        x = F.linear(normed_x, sparse_weight, self.weight_connection.bias)

        # 按模块数拆分
        splits = torch.split(x, x.size(2) // self.module_num, dim=2)

        # 通过模块计算
        outputs = [module(split) for module, split in zip(self.signal_processing_modules, splits)]
        x = torch.cat(outputs, dim=2)

        # 添加 skip connection
        if self.skip_connection is not None:
            skip = self.skip_connection(normed_x)
            x = x + skip

        return x

class FeatureExtractorLayer(nn.Module):
    def __init__(self, feature_extractor_modules, in_channels=1, out_channels=1, sparsity=0.9):
        super(FeatureExtractorLayer, self).__init__()
        self.weight_connection = SparseLinear(in_channels, out_channels, bias=True, sparsity=sparsity)
        self.feature_extractor_modules = nn.ModuleList(feature_extractor_modules)
        
        self.pre_norm = nn.InstanceNorm1d(in_channels)
        self.norm = nn.BatchNorm1d(len(feature_extractor_modules) * out_channels)

    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.pre_norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        
        # 通过线性层
        x = self.weight_connection(normed_x)
        x = rearrange(x, 'b l c -> b c l')
        
        # 通过特征提取模块
        outputs = [module(x) for module in self.feature_extractor_modules]
        res = torch.cat(outputs, dim=1).squeeze()  # B, C
        return self.norm(res)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.clf(x)

class TransparentSignalProcessingNetwork(nn.Module):
    def __init__(self, signal_processing_modules, feature_extractor_modules, args):
        super(TransparentSignalProcessingNetwork, self).__init__()
        self.layer_num = len(signal_processing_modules)
        self.args = args

        self.signal_processing_layers = nn.ModuleList([
            SignalProcessingLayer(
                signal_processing_modules[i],
                input_channels=args.in_channels if i == 0 else args.out_channels * args.scale,
                output_channels=int(args.out_channels * args.scale),
                skip_connection=args.skip_connection,
                temperature=args.temperature
            ) for i in range(self.layer_num)
        ])

        self.channel_for_feature = int(args.out_channels * args.scale)
        self.feature_extractor_layers = FeatureExtractorLayer(
            feature_extractor_modules,
            in_channels=self.channel_for_feature,
            out_channels=self.channel_for_feature,
            sparsity=0.9
        )

        len_feature = len(feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature

        self.clf = Classifier(self.channel_for_classifier, args.num_classes)

    def forward(self, x):
        for layer in self.signal_processing_layers:
            x = layer(x)
        x = self.feature_extractor_layers(x)
        x = self.clf(x)
        return x

    
# class Transparent_Signal_Processing_KAN(Transparent_Signal_Processing_Network):

#     def __init__(self, signal_processing_modules,feature_extractor, args):
#         super(Transparent_Signal_Processing_KAN, self).__init__(signal_processing_modules,
#                                                                     feature_extractor,
#                                                                       args)
#         self.init_classifier()
#     def init_classifier(self):
#         print('# build classifier')
#         self.clf = Kan_classifier(self.channel_for_classifier, self.args.num_classes).to(self.args.device)


    
if __name__ == '__main__':
    from config import args # for debug model
    from config import signal_processing_modules,feature_extractor_modules
    import torchinfo
    net = Transparent_Signal_Processing_Network(signal_processing_modules,feature_extractor_modules, args)
    # net = Transparent_Signal_Processing_KAN(signal_processing_modules,feature_extractor_modules, args)
    x = torch.randn(2, 4096, 2).cuda()
    y = net(x)
    print(y.shape)
    
    net_summaary= torchinfo.summary(net.cuda(),(2,4096,2),device = "cuda")
    print(net_summaary)
    with open(f'save/TSPN_WF.txt','w') as f:
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