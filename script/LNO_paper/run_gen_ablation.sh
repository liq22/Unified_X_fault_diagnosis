
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/gen/config_TSPN_noHT.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/gen/config_TSPN_noLNO.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/gen/config_TSPN_noskip.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/gen/config_TSPN_onlyI.yaml
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/a_031_HUSTablation/gen/config_TSPN_onlymean_std.yaml


