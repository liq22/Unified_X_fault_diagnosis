############# config##########
import argparse
from model.TSPN import Transparent_Signal_Processing_Network
from model.TSPN_KAN import Transparent_Signal_Processing_KAN
from model.NNSPN import NN_Signal_Processing_Network
from model.TFON import Time_Frequency_Operator_Network
from model.Fusion1D2D_simple import Fusion1D2D
from model.MoE_simple import MoEModel as MoE
from model.OperatorAttention_simple import OperatorAttentionModel
from model.FuzzyLogic_simple import FuzzyLogicNetwork
from model.FuzzyLogic_v2 import create_model as FuzzyLogicV2Model
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint

import torch
from pytorch_lightning import seed_everything
from configs.config import parse_arguments,config_network
import os
import pandas as pd
import multiprocessing
import wandb

# 为 Operator Attention 创建与统一框架兼容的包装器
try:
    from model.operator_attention import SimpleOperatorAttention, OperatorLibrary

    class OperatorAttentionNetwork(torch.nn.Module):
        """
        统一基线下使用的算子注意力网络封装

        - 使用 SimpleOperatorAttention 作为核心算子注意力模块
        - 使用 OperatorLibrary 构造 FFT/HT/WF/I 等算子
        - 输出通过一个简单的全连接分类器映射到故障类别
        """

        def __init__(self, signal_processing_modules, feature_extractor_modules, args):
            super().__init__()
            self.input_dim = getattr(args, "in_dim", 4096)
            self.in_channels = getattr(args, "in_channels", 2)
            self.out_channels = getattr(args, "out_channels", 3)
            self.num_classes = getattr(args, "num_classes", 5)
            self.device = getattr(args, "device", "cpu")

            # 核心算子注意力模块（只依赖通道数和设备）
            self.operator_attention = SimpleOperatorAttention(
                in_channels=self.in_channels,
                embed_dim=64,
                hidden_dim=128,
                device=self.device,
            )

            # 算子库：负责构造 FFT / HT / WF / I 等算子
            self.operator_library = OperatorLibrary(args)

            # 简单分类器：对算子注意力输出做全局池化后分类
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.in_channels, 64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(64, self.num_classes),
            )

        def forward(self, x):
            """
            Args:
                x: (batch_size, seq_len, channels) or (batch_size, channels, seq_len)
            """
            # 统一到 (B, L, C) 形式
            if x.dim() == 3 and x.shape[1] == self.input_dim:
                # (B, L, C) 假设 seq_len 在 dim=1
                pass
            elif x.dim() == 3:
                # 视作 (B, C, L)
                x = x.transpose(1, 2)

            # 应用算子注意力：需要 (B, L, C) 和算子字典
            fused, attention_weights, _ = self.operator_attention(
                x, self.operator_library.operators
            )

            # 全局平均池化 + 分类
            pooled = torch.mean(fused, dim=1)  # (B, C)
            logits = self.classifier(pooled)

            return logits

except ImportError:
    # 如果算子注意力模块不可用，则使用占位符，避免训练流程崩溃
    class OperatorAttentionNetwork(torch.nn.Module):
        def __init__(self, signal_processing_modules, feature_extractor_modules, args):
            super().__init__()
            self.num_classes = getattr(args, "num_classes", 5)
            self.classifier = torch.nn.Linear(10, self.num_classes)

        def forward(self, x):
            batch_size = x.size(0)
            # 返回随机输出，提示这是占位实现
            return self.classifier(torch.randn(batch_size, 10, device=x.device))

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    iteration = 5
    # 创建解析器
    parser = argparse.ArgumentParser(description='TSPN')

    # 添加参数
    parser.add_argument('--config_dir', type=str, default='configs/HUST_031/config_basic.yaml',
                        help='The directory of the configuration file')
    parser.add_argument('--notes', type=str, default='')

    meta_args = parser.parse_args()
    config_dir = meta_args.config_dir

    for it in range(iteration):
        configs,args,path,name = parse_arguments(config_dir,it)

        seed_everything(args.seed + it) # 17 args.seed
        wandb.init(project=args.dataset_task, name=name,notes=meta_args.notes)

        # 初始化模型
        signal_processing_modules, feature_extractor_modules = config_network(configs,args)

        MODEL_DICT = {
            'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
            'TKAN': lambda args: Transparent_Signal_Processing_KAN(signal_processing_modules, feature_extractor_modules,args),
            'NNSPN': lambda args: NN_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
            'TFON': lambda args: Time_Frequency_Operator_Network(signal_processing_modules, feature_extractor_modules,args),
            'Fusion1D2D': lambda args: Fusion1D2D(signal_processing_modules, feature_extractor_modules,args),
            'MoE': lambda args: MoE(signal_processing_modules, feature_extractor_modules,args),
            'OperatorAttention': lambda args: OperatorAttentionModel(signal_processing_modules, feature_extractor_modules,args),
            'FuzzyLogic': lambda args: FuzzyLogicNetwork(signal_processing_modules, feature_extractor_modules,args),
            'FuzzyLogicV2': lambda args: FuzzyLogicV2Model(signal_processing_modules, feature_extractor_modules,args),
        }

        model_plain = MODEL_DICT[args.model](args)

        # network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)
        #model trainer #
        model = Basic_plmodel(model_plain, args)
        model_structure = print(model.network)
        trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)

        # train
        trainer.fit(model,train_dataloader, val_dataloader) # TODO load best checkpoint

        model = load_best_model_checkpoint(model,trainer)

        result = trainer.test(model,test_dataloader)

        # 保存结果
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)
        wandb.finish()
