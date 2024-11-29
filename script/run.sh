# 006

CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/THU_006/config_TSPN_CPU.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/THU_006/config_TSPN.yaml
python main.py --config_dir configs/THU_006/config_TKAN.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/THU_006/config_NNSPN.yaml

python main_com.py --config_dir configs/THU_006/config_Resnet.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/THU_006/config_Sincnet.yaml 
CUDA_VISIBLE_DEVICES=7 python main_com.py --config_dir configs/THU_006/config_WKN.yaml
python main_com.py --config_dir configs/THU_006/config_MWA_CNN.yaml 

# 006 gen
python main.py --config_dir configs/THU_006/config_NNSPN_gen.yaml #
python main.py --config_dir configs/THU_006/config_TSPN_gen.yaml
python main.py --config_dir configs/THU_006/config_DEN_gen.yaml
python main.py --config_dir configs/THU_006/config_TKAN_gen.yaml
python main_com.py --config_dir configs/THU_006/config_WKN_gen.yaml
python main_com.py --config_dir configs/THU_006/config_Sincnet_gen.yaml
python main_com.py --config_dir configs/THU_006/config_Resnet_gen.yaml
python main_com.py --config_dir configs/THU_006/config_MWA_CNN_gen.yaml

# 018
python main.py --config_dir configs/THU_018/config_TSPN.yaml
python main_com.py --config_dir configs/THU_018/config_WKN.yaml
python main_com.py --config_dir configs/THU_018/config_Sincnet.yaml
python main_com.py --config_dir configs/THU_018/config_Resnet.yaml
python main_com.py --config_dir configs/THU_018/config_MWA_CNN.yaml

# 018 k-shot
python main_kshotexp.py --config_dir configs/THU_018/config_TSPN_shot.yaml
python main_com_kshotexp.py --config_dir configs/THU_018/config_WKN_shot.yaml
python main_com_kshotexp.py --config_dir configs/THU_018/config_Sincnet_shot.yaml
python main_com_kshotexp.py --config_dir configs/THU_018/config_Resnet_shot.yaml
python main_com_kshotexp.py --config_dir configs/THU_018/config_MWA_CNN_shot.yaml
# debug

# python main.py --config_dir configs/config_basic.yaml --debug

# prune
CUDA_VISIBLE_DEVICES=0  python main.py --config_dir configs/THU_006/config_TSPN_prune.yaml
python main.py --config_dir configs/THU_018/config_TSPN_prune.yaml

# ablation study
