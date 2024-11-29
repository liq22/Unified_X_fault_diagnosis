import wandb
import yaml
import copy
import os

def train():
    with wandb.init() as run:
        config = run.config

        # 读取默认配置文件
        with open('configs/DIRG_020/config_NNSPN_gen.yaml', 'r') as file:  # python main.py --config_dir configs/DIRG_020/config_NNSPN_gen.yaml
            base_config = yaml.safe_load(file)

        # 更新配置文件
        new_config = copy.deepcopy(base_config)
        
        new_config['args']['learning_rate'] = config.learning_rate
        new_config['args']['l1_norm'] = config.l1_norm
        new_config['args']['scale'] = config.scale

        # 保存新配置文件
        new_config_path = f'script/NNSPN_paper/sweep_config/config_{run.id}.yaml'
        with open(new_config_path, 'w') as file:
            yaml.safe_dump(new_config, file)

        # 这里调用你的训练函数
        os.system(f'python main.py --config_dir {new_config_path}')

        # # 可选：运行结束后删除配置文件
        # os.remove(new_config_path)

sweep_config = {
    'method': 'bayes',  # 使用贝叶斯优化
    'metric': {
        'name': 'val_loss',  # 目标是最小化验证损失
        'goal': 'minimize'   
    },
    'parameters': {
        'l1_norm': {
            'distribution': 'uniform',  # 均匀分布
            'min': 5e-05,
            'max': 0.01
        },
        'learning_rate': {
            'distribution': 'uniform',  # 均匀分布
            'min': 5e-05,
            'max': 0.01
        },
        'scale': {
            'distribution': 'int_uniform',  # 整数均匀分布
            'min': 2,
            'max': 8
        },
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        'max_iter': 30
    }
}

# sweep_id = wandb.sweep(sweep_config, project="DIRG_020_geberalization")
# wandb.agent(sweep_id, train)

# sweep_id = wandb.sweep(sweep_config, project="DIRG_020_geberalization")
wandb.agent('liki/DIRG_020_geberalization/by9x2g8d', train)
