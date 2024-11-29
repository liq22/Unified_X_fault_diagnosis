############# learning ############
from cgi import test
from logging import config
from pytorch_lightning import seed_everything

from sklearn.calibration import log
import torch
############# config##########
import argparse
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint
# from configs.config import args
# from configs.config import signal_processing_modules,feature_extractor_modules
from configs.config import parse_arguments,config_network
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' for test ##########################

import numpy as np
from model_collection.Resnet import ResNet, BasicBlock
from model_collection.Sincnet import Sincnet,Sinc_net_m
from model_collection.WKN import WKN,WKN_m
from model_collection.EELM import Dong_ELM
from model_collection.MWA_CNN import A_cSE,Huan_net
from model_collection.TFN.Models.TFN import TFN_Morlet
from model_collection.MCN.models import MCN_GFK, MultiChannel_MCN_GFK
from model_collection.MCN.models import MCN_WFK,MultiChannel_MCN_WFK
import pandas as pd
# 创建解析器
parser = argparse.ArgumentParser(description='comparison model')
# 添加参数
parser.add_argument('--config_dir', type=str, default='configs/config_com.yaml',
                    help='The directory of the configuration file')
configs,args,path = parse_arguments(parser)
seed_everything(args.seed)    


ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1

MODEL_DICT = {
    'Resnet': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
    'WKN_m': lambda args: WKN_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
    'Sinc_net_m': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
    'Huan_net': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
    'TFN_Morlet': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes),
    'MCN_GFK': lambda args: MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes),
}

# 初始化模型
model_plain = MODEL_DICT[args.model](args)
model_structure = print(model_plain)
############## model train ########## 

model = Basic_plmodel(model_plain, args)
trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)
# train
trainer.fit(model,train_dataloader, val_dataloader)
model = load_best_model_checkpoint(model,trainer)
result = trainer.test(model,test_dataloader)

# 保存结果
result_df = pd.DataFrame(result)
result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)