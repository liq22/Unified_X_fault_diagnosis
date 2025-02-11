
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_LNO.yaml
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_TFON.yaml
# other models
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_MCN_basic.yaml
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_MWA_CNN_basic.yaml
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_TFN_basic.yaml
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_Resnet_basic.yaml
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_Sincnet.yaml
CUDA_VISIBLE_DEVICES=6 python main_com.py --config_dir configs/a_017_Ottawa/DG_ABC_D/config_WKN.yaml

