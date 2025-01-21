import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.TSPN import Transparent_Signal_Processing_Network
# from config import args
# from config import signal_processing_modules,feature_extractor_modules
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from .utils import l1_reg,get_all_layers,wgn2,sim_reg,mixup
import numpy as np

def check_attr(args,attr = 'attention_norm'):
    if not hasattr(args, attr):
        setattr(args, attr, False)

class Basic_plmodel(pl.LightningModule):
    def __init__(self, network, args):
        super().__init__()
        self.network = network # placeholder
        self.args = args
        self.loss = nn.CrossEntropyLoss()
        self.acc_val = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        
        args_dict = vars(args)
        self.save_hyperparameters(args_dict,
                                  ignore=['network'])
    
        # print('### network:\n',self.network)
        
    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.args.snr:
            # print('add noise')
            snr = np.random.randint(self.args.snr,0) if self.args.snr < 0 else np.random.randint(0,self.args.snr)
            x = wgn2(x, snr)
            
        y_hat = self(x)
        loss = self.loss(y_hat, y.long())
        
        check_attr(self.args,'mixup')
        if self.args.mixup:
            x_mix,y_mix = mixup(batch,alpha = self.args.mixup)
            y_hat_mix = self(x_mix)
            loss += self.loss(y_hat_mix, y_mix.long())

        acc = self.acc_train(y_hat, y.long())  # torch.argmax(y_hat,dim =1)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)  # sync_dist = False lead to BUG
        self.log('train_acc', acc,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        
        if self.args.l1_norm > 0: # l1 regularization
            # for param in self.network.parameters():  # TODO 只norm linear的权重
            #     regularization_loss += l1_reg(param = param) 
                
            regularization_loss = self.update_regularization_loss()      
                          
            loss += self.args.l1_norm * regularization_loss          
            self.log('l1_loss_', regularization_loss,prog_bar =True,sync_dist=True)   
        
        check_attr(self.args,'attention_norm') 
        if self.args.attention_norm:
            attention_loss = self.update_attention_loss()
            loss += self.args.attention_norm * attention_loss
            self.log('attention_loss', attention_loss,prog_bar =True,sync_dist=True)
        
        
        return loss

    def update_regularization_loss(self):
        regularization_loss = 0
        for i, (name,param) in enumerate(self.network.named_parameters()):
            if 'WF' not in name:
                regularization_loss += l1_reg(param = param)
        return regularization_loss
    
    def update_attention_loss(self):
        regularization_loss = 0

        for layer in self.network.signal_processing_layers:
            gate_value = layer.channel_attention.gate
            regularization_loss += sim_reg(tensor = gate_value)
            
        gate_value = self.network.feature_extractor_layers.FEAttention.gate
        regularization_loss += sim_reg(tensor = gate_value)
        
        return regularization_loss

    
    
    def validation_step(self, batch, batch_idx):
        # self.eval()
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y.long())
        acc = self.acc_val(y_hat, y.long())
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        # return val_loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss(y_hat, y.long())
        self.log('test_loss', test_loss,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        acc = self.acc_test(y_hat, y.long())
        self.log('test_acc', acc,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return {'test_acc': acc, 'test_loss': test_loss}
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = self.config_different_lr_optimizer()
        # optimizer = Adam(self.parameters(), lr=self.args.learning_rate, weight_decay = self.args.weight_decay)
        out = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer),
            "monitor": self.args.monitor ,
            "frequency": self.args.patience//2

            },
        }
        return out
    def config_different_lr_optimizer(self):
        '''config different learning rate for different layers'''
        
        if not hasattr(self.args, 'learnable_parameter_learning_rate'):
            setattr(self.args, 'learnable_parameter_learning_rate', self.args.learning_rate)  # fix bug
            
        layers = get_all_layers(self.network, layers = [])
        # for i, module in enumerate(self.network.children()):
        #     # if not isinstance(module, nn.Sequential):
        #         layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]
        parameters_conv = []        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                # 假设我们只想为线性层的权重参数设置不同的学习率
                parameters_conv.append({'params': layer.weight, 'lr': self.args.learning_rate, 'weight_decay': self.args.weight_decay})
                # 如果你也想为偏置项设置学习率，可以像这样添加:
                # parameters_conv.append({'params': layer.bias, 'lr': self.args.learning_rate})
        
        # 确保网络中未被上述步骤指定的其它所有参数都有一个默认的学习率
        base_params = filter(lambda p: id(p) not in [id(param['params']) for param in parameters_conv], self.network.parameters())

        optimizer = Adam([
            {'params': base_params},
            *parameters_conv
        ], lr=self.args.learnable_parameter_learning_rate, weight_decay=self.args.weight_decay)
        
        return optimizer



