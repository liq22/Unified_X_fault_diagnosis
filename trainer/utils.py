from pytorch_lightning.callbacks import Callback
import torch
import copy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import numpy as np

def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=1)

def random_shuffle_channels(tensor):
    # 获取C通道的数量
    C = tensor.size(1)
    
    # 生成随机的C通道索引
    perm = torch.randperm(C)
    
    # 打乱C通道
    shuffled_tensor = tensor[:, perm]

    return shuffled_tensor

def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower) + x 

def l1_reg(param):
    return torch.sum(torch.abs(param))

def sim_reg(tensor):
    shuffle_tensor = random_shuffle_channels(tensor)
    sim = cosine_similarity(tensor, shuffle_tensor)
    sim_sum = l1_reg(sim)
    return sim_sum
    
def mixup(batch,alpha = 0.8):
    # mix_ratio = np.random.dirichlet(np.ones(3) * 0.9,size=1) # 设置为0.9
    lamda = np.random.beta(alpha,alpha)
    x,y = batch
    index = torch.randperm(x.size(0)).cuda()
    
    x = lamda * x + (1-lamda) * x[index]
    y = lamda * y + (1-lamda) * y[index]
        
    return x,y

import pandas as pd
import os

class ModelParametersLoggingCallback(Callback):
    def __init__(self, path="./", module_type=None,name = "/params_history.csv"):
        super().__init__()
        # 初始化一个空的DataFrame来保存参数历史
        self.params_history = pd.DataFrame()
        self.module_type = module_type
        self.path = path
        self.name = name

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_params = {}
        layers = get_all_layers(pl_module.network,[])
        for num, layer in enumerate(layers):
            if isinstance(layer, self.module_type):
                for name, params in layer.named_parameters():
                    param_values = params.clone().detach().cpu().numpy()
                    if param_values.ndim > 1:  # 假设一个多维数组需要展开
                        # 展开参数数组到不同的key下，这里假设param_values是二维的，适用于大多数情况
                        for i, value in enumerate(param_values.flatten()):
                            # 为每个子参数创建独特的key
                            key_name = f'{num}_{layer}_{name}_{i}'
                            epoch_params[key_name] = value
                    else:
                        key_name = f'{num}_{layer}_{name}'
                        epoch_params[key_name] = param_values.item()  # 使用.item()提取标量值，并将其包装在列表中
                    
        # 如果params_history是空的，直接从epoch_params创建一个新的DataFrame
        if self.params_history.empty:
            # self.params_history = pd.DataFrame.from_dict(epoch_params)  # ValueError: If using all scalar values, you must pass an index
            self.params_history = pd.DataFrame([epoch_params], index=[0])
        else:
            # 否则，将新的一行添加到params_history中
            new_row = pd.DataFrame([epoch_params], index=[0])
            self.params_history = pd.concat([self.params_history, new_row], ignore_index=True)

    def on_train_end(self, trainer, pl_module):
        # 检查params_history是否为空
        if self.params_history.empty:
            print("No parameters to save.")
            return
        
        # 确保保存路径存在
        directory = os.path.dirname(self.path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        
        # 保存params_history到CSV文件
        self.params_history.to_csv(self.path + self.name , index=False)
        print(f"Saved all parameters history to {self.path}")

    
def get_all_layers(module, layers=[]):
    for child in module.children():
        # 如果子模块没有更深层的子模块，则直接添加
        if not list(child.children()):
            layers.append(child)
        else:
            # 否则，递归调用自身
            get_all_layers(child, layers)
    return layers

def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """
    加载训练过程中保存的最佳模型检查点。

    参数:
    - model: 要加载检查点权重的模型实例。
    - trainer: 用于训练模型的训练器实例。

    返回:
    - 加载了最佳检查点权重的模型实例。
    """
    # 从trainer的callbacks中找到ModelCheckpoint实例，并获取best_model_path
    model_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            model_checkpoint = callback
            break

    if model_checkpoint is None:
        raise ValueError("ModelCheckpoint callback not found in trainer's callbacks.")

    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # 确保最佳模型路径不是空的
    if not best_model_path:
        raise ValueError("No best model path found. Please check if the training process saved checkpoints.")

    # 加载最佳检查点
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict['state_dict'])
    return model