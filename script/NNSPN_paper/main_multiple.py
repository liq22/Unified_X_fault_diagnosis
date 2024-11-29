import wandb
import yaml
import copy
import os

sweep_id = wandb.sweep(sweep_config, project="DIRG_020_geberalization")
wandb.agent(sweep_id, train)