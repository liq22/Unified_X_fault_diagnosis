############# learning ############
from cgi import test
from logging import config
from pytorch_lightning import seed_everything

from sklearn.calibration import log
import torch
############# config##########
import argparse
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint
# from configs.config import args
# from configs.config import signal_processing_modules,feature_extractor_modules
from configs.config import parse_arguments,config_network
import os
import wandb
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' for test ##########################

import numpy as np
from model_collection.Resnet import ResNet, BasicBlock
from model_collection.Sincnet import Sincnet,Sinc_net_m
from model_collection.WKN import WKN,WKN_m
from model_collection.EELM import Dong_ELM
from model_collection.MWA_CNN import A_cSE,Huan_net
from model_collection.TFN.Models.TFN import TFN_Morlet
from model_collection.MCN.models import MCN_GFK, MultiChannel_MCN_GFK
from model_collection.MCN.models import MCN_WFK,MultiChannel_MCN_WFK
import pandas as pd
import multiprocessing
# Unified schema writer (best-effort)
from pathlib import Path
import time
from datetime import datetime, timezone

try:
    from uxfd.data import get_default_catalog
    from uxfd.io.schema_v1 import RunContext, infer_paper_id, write_run_schema
    from uxfd.explain import ExplainEvalConfig, eval_explainability_on_batch
    from uxfd.registry import PAPER_REGISTRY
except Exception:  # pragma: no cover
    get_default_catalog = None
    RunContext = None
    infer_paper_id = None
    write_run_schema = None
    ExplainEvalConfig = None
    eval_explainability_on_batch = None
    PAPER_REGISTRY = {}
# 导入新的统一基线模型
from model.OperatorAttention_simple import OperatorAttentionModel
from model.OperatorAttention_enhanced import EnhancedOperatorAttentionModel
from model.FuzzyLogic_simple import FuzzyLogicNetwork
from model.MoE_simple import MoEModel
from model.MoE import MoEModel as MoEAdvancedModel
from model.Fusion1D2D_simple import Fusion1D2D
if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # 创建解析器
    iteration = 5
    parser = argparse.ArgumentParser(description='comparison model')
    # 添加参数
    parser.add_argument('--config_dir', type=str, default='configs/SEU_010/config_MCN_basic.yaml',
                        help='Path to configuration file (legacy name)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Alias of --config_dir (for backward compatibility)')
    parser.add_argument('--iteration', type=int, default=5,
                        help='Number of repeated runs (default: 5, same as legacy)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override args.seed (base seed) in config')
    meta_args = parser.parse_args()
    config_dir = meta_args.config_file or meta_args.config_dir
    iteration = int(meta_args.iteration)
    for it in range(iteration):
        configs,args,path,name = parse_arguments(config_dir,it)
        run_start_wall = time.time()
        run_start_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        # Optional seed override
        if meta_args.seed is not None:
            args.seed = int(meta_args.seed)
            try:
                import yaml

                snapshot_path = getattr(args, "config_snapshot_path", None)
                if snapshot_path:
                    snap = yaml.safe_load(Path(snapshot_path).read_text(encoding="utf-8"))
                    if isinstance(snap, dict) and "args" in snap and isinstance(snap["args"], dict):
                        snap["args"]["seed"] = args.seed
                        Path(snapshot_path).write_text(
                            yaml.safe_dump(snap, sort_keys=False, allow_unicode=True),
                            encoding="utf-8",
                        )
            except Exception:
                pass

        actual_seed = int(args.seed) + int(it)
        seed_everything(actual_seed) # 17 args.seed
        wandb.init(project=args.dataset_task, name=name)

        # 构建信号处理和特征提取模块
        signal_processing_modules, feature_extractor_modules = config_network(configs, args)

        ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1

        MODEL_DICT = {
            'Resnet': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'WKN_m': lambda args: WKN_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'Sinc_net_m': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'Huan_net': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
            'TFN_Morlet': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes),
            'MCN_GFK': lambda args: MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes),
            # 统一基线模型
            'OperatorAttention': lambda args: OperatorAttentionModel(signal_processing_modules, feature_extractor_modules, args),
            'OperatorAttention_enhanced': lambda args: EnhancedOperatorAttentionModel(signal_processing_modules, feature_extractor_modules, args),
            'FuzzyLogic': lambda args: FuzzyLogicNetwork(signal_processing_modules, feature_extractor_modules, args),
            'MoE_simple': lambda args: MoEModel(signal_processing_modules, feature_extractor_modules, args),
            'MoE': lambda args: MoEAdvancedModel(args),
            'Fusion1D2D': lambda args: Fusion1D2D(args),
        }

        # 初始化模型
        model_plain = MODEL_DICT[args.model](args)
        model_structure = print(model_plain)
        ############## model train ########## 

        model = Basic_plmodel(model_plain, args)
        trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)
        # train
        trainer.fit(model,train_dataloader, val_dataloader)
        model = load_best_model_checkpoint(model,trainer)
        result = trainer.test(model,test_dataloader)

        # 保存结果
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)

        # 写入统一 schema（best-effort）
        if write_run_schema is not None and RunContext is not None:
            try:
                run_end_wall = time.time()
                run_end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

                run_dir = Path(path)
                test_csv = run_dir / "test_result.csv"

                dataset_id = getattr(args, "dataset_task", "UNKNOWN")
                dataset_numeric_id = None
                vb = getattr(args, "vbench_config", None)
                if isinstance(vb, dict):
                    ids = vb.get("dataset_ids")
                    if isinstance(ids, list) and len(ids) == 1:
                        dataset_numeric_id = int(ids[0])
                        if get_default_catalog is not None:
                            name_map = get_default_catalog().name_of(dataset_numeric_id)
                            if name_map:
                                dataset_id = name_map

                paper_id = infer_paper_id(getattr(args, "model", "UNKNOWN"), os.getenv("UXFD_PAPER_ID"))
                paper_dir = PAPER_REGISTRY.get(paper_id).paper_dir if paper_id in PAPER_REGISTRY else Path(".")

                cfg_path = getattr(args, "config_snapshot_path", None) or str(config_dir)
                cmd = " ".join([f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}" if os.getenv("CUDA_VISIBLE_DEVICES") else "", "python", "main_com.py", "--config_dir", str(config_dir)]).strip()

                ctx = RunContext(
                    run_dir=run_dir,
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    model_id=str(getattr(args, "model", "UNKNOWN")),
                    seed=actual_seed,
                    dataset_id=str(dataset_id),
                    dataset_numeric_id=dataset_numeric_id,
                    command=cmd,
                    config_path=str(cfg_path),
                    device=str(getattr(args, "device", "UNKNOWN")),
                )
                metrics_patch = {}
                if eval_explainability_on_batch is not None and ExplainEvalConfig is not None:
                    try:
                        batch = next(iter(test_dataloader))
                        x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                        device = next(model.network.parameters()).device
                        x_batch = x_batch.to(device)
                        explainability = eval_explainability_on_batch(model.network, x_batch, cfg=ExplainEvalConfig())
                        metrics_patch = {"explainability": explainability}
                    except Exception as exc:
                        print(f"[WARN] failed to compute explainability metrics for {path}: {exc}")

                run_meta_patch = {
                    "timestamps": {
                        "start_utc": run_start_utc,
                        "end_utc": run_end_utc,
                        "duration_sec": float(run_end_wall - run_start_wall),
                    }
                }
                write_run_schema(
                    ctx,
                    test_result_csv=test_csv,
                    run_meta_patch=run_meta_patch,
                    metrics_patch=metrics_patch,
                )
            except Exception as exc:
                print(f"[WARN] failed to write schema for {path}: {exc}")
        wandb.finish()
