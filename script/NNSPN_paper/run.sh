# 006
CUDA_VISIBLE_DEVICES=0

# CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/THU_006/config_NNSPN_gen.yaml #
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/THU_006/config_NNSPN_gen.yaml #
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/THU_006/config_NNSPN.yaml
# DIRG


# DIRG_20
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/DIRG_020/config_NNSPN.yaml


CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_Resnet.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_Sincnet.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_WKN.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_MWA_CNN.yaml 

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/DIRG_020/config_NNSPN_gen.yaml


CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_Resnet_gen.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_Sincnet_gen.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_WKN_gen.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/DIRG_020/config_MWA_CNN_gen.yaml 



