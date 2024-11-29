from os import pread
from altair import AllSortString
import torch
from torch import nn

from Signal_processing import SignalProcessingBase,\
        SignalProcessingModuleDict,\
        FFTSignalProcessing,\
        HilbertTransform,\
        WaveFilters,\
        Identity

from Feature_extract import FeatureExtractionBase,\
        FeatureExtractionModuleDict,\
        MeanFeature,\
        StdFeature,\
        VarFeature,\
        EntropyFeature,\
        MaxFeature,\
        MinFeature,\
        AbsMeanFeature,\
        KurtosisFeature,\
        RMSFeature,\
        CrestFactorFeature ,\
        SkewnessFeature,\
        ClearanceFactorFeature,\
        ShapeFactorFeature
ALL_SP = {
    '$FFT$': FFTSignalProcessing,
    '$HT$': HilbertTransform,
    '$WF$': WaveFilters,
    '$I$':Identity,
}
ALL_FE = {
    '$Mean$': MeanFeature,
    '$Std$': StdFeature,
    '$Var$': VarFeature,
    '$Entropy$': EntropyFeature,
    '$Max$': MaxFeature,
    '$Min$': MinFeature,
    '$AbsMean$': AbsMeanFeature,
    '$Kurtosis$': KurtosisFeature,
    '$RMS$': RMSFeature,
    '$CrestFactor$': CrestFactorFeature,
    '$Skewness$': SkewnessFeature,
    '$ClearanceFactor$': ClearanceFactorFeature,
    '$ShapeFactor$': ShapeFactorFeature,
}

# logic

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
signal_processing_configs = {
'channel_1'：{
    'layer1': [FFTSignalProcessing(input_dim, in_channels), 
              HilbertTransform(input_dim, in_channels)],
    'layer2': [HilbertTransform(input_dim, in_channels), 
              WaveFilters(input_dim, in_channels, args)],
    ...
}
‘channel_2’：{
...
}
}
'''

'''
feature_extractor_configs = {
'channel_1'：{
    'features_1': [EntropyFeature, 
              EntropyFeature,
              ...
    'features_2': [RMSFeature, 
              ClearanceFactorFeature],
    ...
}
‘channel_2’：{
...
}
}
'''

class SignalProcessingNetwork(nn.Module):
    def __init__(self, signal_processing_configs,
                  feature_extractor_configs,
                  args):
        super(SignalProcessingNetwork, self).__init__()
        self.signal_processing_layers = nn.ModuleDict()
        self.feature_extractors = nn.ModuleDict()
        self.num_classes = args.num_classes
        self.args = args
        self.config_sp_layers(signal_processing_configs)
        self.config_fe_layers(feature_extractor_configs)
        self.config_classifier()
    

    def config_sp_layers(self, signal_processing_configs):
        # 构建信号处理层
        for channel, layers in signal_processing_configs.items(): # channels = ["channel_1", "channel_2"]
            for layer_name, modules in layers.items(): # layer_name = "layer_1", modules = [module_1, module_2]
                for i, module in enumerate(modules): # module_1, module_2
                    # 为每个模块添加一个InstanceNorm层
                    self.signal_processing_layers[f"{channel}_{layer_name}_instancenorm_{i}"] = nn.InstanceNorm1d(module.in_channels)
                    
                    # 确定线性层的输入输出维度
                    input_dim = module.input_dim # 实际上是信号输入的维度
                    output_dim = module.output_dim
                    # 为每个模块添加一个Linear层
                    self.signal_processing_layers[f"{channel}_{layer_name}_linear_{i}"] = nn.Linear(input_dim, output_dim)
                    
                    # 添加信号处理模块
                    self.signal_processing_layers[f"{channel}_{layer_name}_module_{i}"] = module

                    # 添加skip_connection
                    if self.args.skip_connection:
                        self.signal_processing_layers[f"{channel}_{layer_name}_skipconnection_{i}"] = nn.Linear(input_dim, output_dim)
                    
    def config_fe_layers(self, feature_extractor_configs):
        # 构建特征提取层
        self.final_dim = 0
        for channel, channel_features in feature_extractor_configs.items(): # channels = ["channel_1", "channel_2"]
            for layer_name, features in channel_features.items(): 
                for i, feature in enumerate(features):
                    self.feature_extractors[f"{channel}_{layer_name}_feature_{i}"] = feature
                    self.final_dim += 1
        
    def config_classifier(self):
        # 构建分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.final_dim, 100), # calculate the input dimension
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
        )
        

    def forward_sp_per_channel(self, x, channel):
        
        channel_x = x[:, :,channel]
        num_layer1 = len(self.signal_processing_configs[channel]['layer1'])
        prev_layer_outputs = [channel_x] * num_layer1 # list

        # 遍历所有信号处理层
        for layer_name, modules in self.signal_processing_configs[channel].items():
            current_layer_outputs = []
            for i, module in enumerate(modules):  # 保证层与层之间list一致 TODO 在 Module 设置 arguements
                
                # 应用InstanceNorm
                normed_signal = self.signal_processing_layers[f"{channel}_{layer_name}_instancenorm_{i}"](prev_layer_outputs[i])
                
                # 应用Linear变换
                linear_signal = self.signal_processing_layers[f"{channel}_{layer_name}_linear_{i}"](normed_signal)
                
                # 应用信号处理模块
                processed_signal = module(linear_signal)

                # skip connection
                if self.args.skip_connection:
                    processed_signal = processed_signal +\
                        self.signal_processing_layers[f"{channel}_{layer_name}_skipconnection_{i}"](prev_layer_outputs[i])
                
                # 保存当前模块的输出到当前层的输出字典中
                current_layer_outputs.append(processed_signal)
            
            # 将当前层的输出字典保存，以供下一层使用
            prev_layer_outputs = current_layer_outputs
        
        return current_layer_outputs

    def norm(self,x):
        mean = x.mean(dim = 0,keepdim = True)
        std = x.std(dim = 0,keepdim = True)
        out = (x-mean)/(std + 1e-10)
        return out
    
    def forward_fe(self, x):
        output_feature_list = []
        for channel, channel_features in self.feature_extractor_configs.items(): # channels = ["channel_1", "channel_2"]
            # 获取当前通道的信号
            channel_x = x[channel]
            # 遍历所有特征提取层
            for layer_name, features in channel_features.items():  # feature list
                for i, feature in enumerate(features): # feature
                    output_feature_list.append(self.feature_extractors[f"{channel}_{layer_name}_feature_{i}"](channel_x))
        output_features = torch.stack(output_feature_list, dim=1)
        feature = self.norm(output_features)
        return feature

    def forward(self, x):
        # 信号处理层
        channels = x.shape[2]
        x_signal_list = []
        for channel in range(channels):
            x_sp = self.forward_sp_per_channel(x, channel)
            x_signal_list.append(x_sp)
        x_fe = self.forward_fe(x_signal_list)
        x_logics = self.classifier(x_fe)
        return x_logics

    # def forward(self, x):
    #     # 信号处理层
    #     for name, module in self.signal_processing_modules.items():
    #         x = module(x)
    #         x = self.fc_layers[name](x)

    #     # 特征提取
    #     x = self.feature_extractors(x)

    #     # 分类
    #     x = self.classifier(x)
    #     return x
    def parse_network(self):
        pass

if __name__ == "__main__":
    import yaml

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    signal_processing_configs = config['signal_processing_configs']
    for channel, layers in signal_processing_configs.items():
        print(channel)
        for layer_name, modules in layers.items():
            print(layer_name)
            module_list = []
            for module in modules:
                print(module)
                module_list.append(module)
            signal_processing_configs[channel][layer_name] = module_list

    feature_extractor_configs = config['feature_extractor_configs']

    for channel, layers in feature_extractor_configs.items():
        print(channel)
        for layer_name, features in layers.items():
            print(layer_name)
            feature_list = []
            for feature in features:
                print(feature)
                feature_list.append(feature)
            feature_extractor_configs[channel][layer_name] = feature_list

    test_signal = torch.randn(2, 2048, 3)

# # 创建信号处理模块和特征提取器
# signal_processing_modules = SignalProcessingModuleDict(module_dict)
# feature_extractors = FeatureExtractionModuleDict(feature_extractors)

# # 创建神经网络
# num_classes = 10  # 假设有10个类别
# model = SignalProcessingNetwork(signal_processing_modules, feature_extractors, num_classes)