#!/bin/bash

# 定义YAML文件所在的目录
# CONFIG_DIR="configs/THU_018ablation"

# # 遍历目录下的所有.yaml文件
# for config_file in "$CONFIG_DIR"/*.yaml
# do
#     echo "Running experiment with config: $config_file"
#     # 执行Python命令，并将YAML文件作为参数传递
#     python main_ablation_exp.py --config_dir "$config_file"
# done

# echo "All experiments completed."

# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN.yaml
# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_noskip.yaml
python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_woHT.yaml
python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_onlyI.yaml
python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_onlyMean.yaml
python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_woWF.yaml

CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/THU_006ablation/config_TSPN_RWF.yaml --results_file RWF.csv
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/THU_006ablation/config_TSPN_CWF.yaml --results_file CWF.csv
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/THU_006ablation/config_TSPN_LWF.yaml --results_file LWF.csv

sensors | grep "Core"
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
