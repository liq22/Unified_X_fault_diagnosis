## HUST_031_20Hz
# LNO and ablation
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUST/config_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noHT.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noLNO.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noskip.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_onlyI.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_onlymean_std.yaml

# other models
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_MCN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_MWA_CNN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_TFN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_Resnet_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_Sincnet.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_WKN.yaml

## HUST_031_0_40_0Hz
# LNO and ablation
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUST/config_basic_2.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noHT_2.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noLNO_2.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_noskip_2.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_onlyI_2.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/config_TSPN_onlymean_std_2.yaml

# other models
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_MCN_basic_2.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_MWA_CNN_basic_2.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_TFN_basic_2.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_Resnet_basic_2.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_Sincnet_2.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/config_WKN_2.yaml