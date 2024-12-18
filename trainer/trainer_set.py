import accelerate
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks import EarlyStopping

from .utils import ModelParametersLoggingCallback


import pytorch_lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger


from torch.utils.tensorboard.writer import SummaryWriter   
############### data ###############
from data.data_provider import get_data

###### model ###
from model.Signal_processing import WaveFilters

def trainer_set(args,path):
    # 设置检查点回调以保存模型

    wandb_logger = WandbLogger(project=args.dataset_task)

    callback_list = call_backs(args,path)

    if not hasattr(args, 'wandb_flag'):
        setattr(args, 'wandb_flag', True)  # TODO add this arg to fix bug
    log_list = [CSVLogger(path, name="logs"),wandb_logger] if args.wandb_flag else [CSVLogger(path, name="logs")]

    accelerate_type = 'cpu' if args.device == 'cpu' else 'auto'
    
    trainer = pl.Trainer(callbacks=callback_list,
                        #  profiler='pytorch', # TODO args
                         accelerator = accelerate_type,
                        max_epochs=args.num_epochs,
                        devices= args.gpus,
                        logger = log_list,# ,TensorBoardLogger(path, name="logs")],
                        log_every_n_steps=1,)
    
    train_dataloader, val_dataloader, test_dataloader = get_data(args)

    return trainer,train_dataloader, val_dataloader, test_dataloader

def call_backs(args,path):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}-{l1_loss:.4f}',
        save_top_k=8,
        mode='min',
        dirpath = path
    )
    # 初始化训练器
    callback_list = [checkpoint_callback]
    prune_callback = Prune_callback(args)
    if prune_callback is not None:
        callback_list.append(prune_callback)
        
    if not hasattr(args, 'log_parameters'):
        setattr(args, 'log_parameters', None)  # TODO add this arg to fix bug
        
    if args.log_parameters:
        callback_list.append(ModelParametersLoggingCallback(path = path, module_type = WaveFilters))
        
    early_stopping = create_early_stopping_callback(args)
    callback_list.append(early_stopping)
    
    return callback_list


def Prune_callback(args):
    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        if epoch == args.num_epochs//4:
            return args.pruning[0]
        elif epoch == args.num_epochs//2:
            return args.pruning[1]
        elif 3 * args.num_epochs//4 < epoch:
            return args.pruning[2]
        
    if isinstance(args.pruning, (int, float)):
        prune_callback = ModelPruning("l1_unstructured",
                                      parameter_names = ['weight'],
                                      amount=args.pruning)
    elif isinstance(args.pruning, list):
        prune_callback = ModelPruning("l1_unstructured",
                                      parameter_names = ['weight'],
                                      amount = compute_amount)
    else:
        prune_callback = None
    return prune_callback

def create_early_stopping_callback(args):
    """
    根据args参数创建EarlyStopping回调实例。
    
    参数:
    - args: 包含配置的对象，比如Namespace对象。
    
    返回:
    - 一个配置好的EarlyStopping实例。
    
    """
        # 使用args中指定的patience值
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.patience,  # 从args中读取patience值
        verbose=True,
        mode='min',
        check_finite=True,  # 当监控指标为无穷大或NaN时停止
        check_on_train_epoch_end=False  # 仅在验证阶段检查
    )
    
    return early_stopping
