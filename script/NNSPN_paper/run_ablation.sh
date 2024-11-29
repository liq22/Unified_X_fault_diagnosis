# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_woHT.yaml
# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_onlyI.yaml
# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_onlyMean.yaml
# python main_ablation_exp.py --config_dir configs/THU_018ablation/config_TSPN_woWF.yaml

CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_no_skip.yaml 
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_onlyHT.yaml 
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_onlyI.yaml 
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_onlyKurtosis.yaml 
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_onlyMean.yaml 
CUDA_VISIBLE_DEVICES=7 python main_ablation_exp.py --config_dir configs/DIRG_020/config_NNSPN_ablation_onlyWF.yaml

CUDA_VISIBLE_DEVICES=0 python /home/user/LQ/B_Signal/Transparent_information_fusion/script/NNSPN_paper/main_sweep.py
CUDA_VISIBLE_DEVICES=1 python /home/user/LQ/B_Signal/Transparent_information_fusion/script/NNSPN_paper/main_sweep.py
CUDA_VISIBLE_DEVICES=7 python /home/user/LQ/B_Signal/Transparent_information_fusion/script/NNSPN_paper/main_sweep.py
CUDA_VISIBLE_DEVICES=3 python /home/user/LQ/B_Signal/Transparent_information_fusion/script/NNSPN_paper/main_sweep.py

CUDA_VISIBLE_DEVICES=7 python /home/user/LQ/B_Signal/Transparent_information_fusion/script/NNSPN_paper/main_sweep_basic.py