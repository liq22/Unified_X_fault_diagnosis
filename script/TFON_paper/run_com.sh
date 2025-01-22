# chmod +x

CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_Resnet.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_WKN.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_Sincnet.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_MWA_CNN.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_MCN.yaml
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/a_017_Ottawa/config_TFN.yaml
