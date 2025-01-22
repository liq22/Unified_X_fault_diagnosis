## SEU_20Hz
# LNO and ablation
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEU/config_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEUablation/config_TSPN_noHT.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEUablation/config_TSPN_noLNO.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEUablation/config_TSPN_noskip.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEUablation/config_TSPN_onlyI.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_010_SEUablation/config_TSPN_onlymean_std.yaml

# other models
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_MCN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_MWA_CNN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_TFN_basic.yaml
# CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_Resnet_basic.yaml
# CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_Sincnet.yaml
# CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_010_SEU/config_WKN.yaml