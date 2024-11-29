import wandb
import yaml
import copy
import os

# yaml_file = 'configs/DIRG_020/config_NNSPN.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN_ablation_onlyHT.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN_ablation_onlyI.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN_ablation_onlyKurtosis.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN_ablation_onlyMean.yaml'
# yaml_file = 'configs/DIRG_020/config_NNSPN_ablation_onlyWF.yaml'


def train():
    with wandb.init() as run:
        config = run.config
        
        with open(config.script, 'r') as file:  # python main.py --config_dir configs/DIRG_020/config_NNSPN_gen.yaml
            base_config = yaml.safe_load(file)

        # 更新配置文件
        new_config = copy.deepcopy(base_config)
        
        new_config['args']['learning_rate'] = config.learning_rate
        new_config['args']['l1_norm'] = config.l1_norm
        new_config['args']['aba'] = config.script


        # 保存新配置文件
        new_config_path = f'script/NNSPN_paper/sweep_config/config_{run.id}.yaml'
        with open(new_config_path, 'w') as file:
            yaml.safe_dump(new_config, file)

        # 这里调用你的训练函数
        os.system(f'python main.py --config_dir {new_config_path}')

        # # 可选：运行结束后删除配置文件
        # os.remove(new_config_path)

sweep_config = {
    'method': 'grid',  # 使用贝叶斯优化
    'metric': {
        'name': 'val_loss',  # 目标是最小化验证损失
        'goal': 'minimize'   
    },
    'parameters': {
        'l1_norm': {
            'values': [0.1, 0.01, 0.001,0.0001,0.00001]
        },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001,0.0001,0.00001]
        },
        'script': {
            'values': ['configs/DIRG_020/config_NNSPN.yaml',
                      'configs/DIRG_020/config_NNSPN_ablation_onlyHT.yaml',
                      'configs/DIRG_020/config_NNSPN_ablation_onlyI.yaml',
                      'configs/DIRG_020/config_NNSPN_ablation_onlyKurtosis.yaml',
                      'configs/DIRG_020/config_NNSPN_ablation_onlyMean.yaml',
                      'configs/DIRG_020/config_NNSPN_ablation_onlyWF.yaml']
    },
    }
    # 'early_terminate': {
    #     'type': 'hyperband',
    #     'min_iter': 5,s
    #     'max_iter': 30
    # }
}
# sweep_id = wandb.sweep(sweep_config, project="DIRG_020_abalation4")
# wandb.agent(sweep_id, train)

# sweep_id = wandb.sweep(sweep_config, project="DIRG_020_geberalization")
wandb.agent('liki/DIRG_020_abalation4/8j2rx6eh', train)

