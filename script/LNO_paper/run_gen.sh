
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUST/gen/config_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUST/gen/config_TFON.yaml
# other models
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_MCN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_MWA_CNN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_TFN_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_Resnet_basic.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_Sincnet.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_031_HUST/gen/config_WKN.yaml

