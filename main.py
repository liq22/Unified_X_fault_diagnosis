############# config##########
import argparse
from model.TSPN import Transparent_Signal_Processing_Network
from model.TSPN_KAN import Transparent_Signal_Processing_KAN
from model.NNSPN import NN_Signal_Processing_Network
from model.TFON import Time_Frequency_Operator_Network
from model.Fusion1D2D_simple import Fusion1D2D
from model.MoE_simple import MoEModel as MoE
from model.OperatorAttention_simple import OperatorAttentionModel
from model.FuzzyLogic_simple import FuzzyLogicNetwork
from model.FuzzyLogic_v2 import create_model as FuzzyLogicV2Model
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint

import torch
from pytorch_lightning import seed_everything
from configs.config import parse_arguments,config_network
import os
import pandas as pd
import multiprocessing
import wandb
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

# 为 Operator Attention 创建与统一框架兼容的包装器
try:
    from model.operator_attention import SimpleOperatorAttention, OperatorLibrary

    class OperatorAttentionNetwork(torch.nn.Module):
        """
        统一基线下使用的算子注意力网络封装

        - 使用 SimpleOperatorAttention 作为核心算子注意力模块
        - 使用 OperatorLibrary 构造 FFT/HT/WF/I 等算子
        - 输出通过一个简单的全连接分类器映射到故障类别
        """

        def __init__(self, signal_processing_modules, feature_extractor_modules, args):
            super().__init__()
            self.input_dim = getattr(args, "in_dim", 4096)
            self.in_channels = getattr(args, "in_channels", 2)
            self.out_channels = getattr(args, "out_channels", 3)
            self.num_classes = getattr(args, "num_classes", 5)
            self.device = getattr(args, "device", "cpu")

            # 核心算子注意力模块（只依赖通道数和设备）
            self.operator_attention = SimpleOperatorAttention(
                in_channels=self.in_channels,
                embed_dim=64,
                hidden_dim=128,
                device=self.device,
            )

            # 算子库：负责构造 FFT / HT / WF / I 等算子
            self.operator_library = OperatorLibrary(args)

            # 简单分类器：对算子注意力输出做全局池化后分类
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.in_channels, 64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(64, self.num_classes),
            )

        def forward(self, x):
            """
            Args:
                x: (batch_size, seq_len, channels) or (batch_size, channels, seq_len)
            """
            # 统一到 (B, L, C) 形式
            if x.dim() == 3 and x.shape[1] == self.input_dim:
                # (B, L, C) 假设 seq_len 在 dim=1
                pass
            elif x.dim() == 3:
                # 视作 (B, C, L)
                x = x.transpose(1, 2)

            # 应用算子注意力：需要 (B, L, C) 和算子字典
            fused, attention_weights, _ = self.operator_attention(
                x, self.operator_library.operators
            )

            # 全局平均池化 + 分类
            pooled = torch.mean(fused, dim=1)  # (B, C)
            logits = self.classifier(pooled)

            return logits

except ImportError:
    # 如果算子注意力模块不可用，则使用占位符，避免训练流程崩溃
    class OperatorAttentionNetwork(torch.nn.Module):
        def __init__(self, signal_processing_modules, feature_extractor_modules, args):
            super().__init__()
            self.num_classes = getattr(args, "num_classes", 5)
            self.classifier = torch.nn.Linear(10, self.num_classes)

        def forward(self, x):
            batch_size = x.size(0)
            # 返回随机输出，提示这是占位实现
            return self.classifier(torch.randn(batch_size, 10, device=x.device))

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # 创建解析器
    parser = argparse.ArgumentParser(description='TSPN')

    # 添加参数
    parser.add_argument('--config_dir', type=str, default='configs/HUST_031/config_basic.yaml',
                        help='Path to configuration file (legacy name)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Alias of --config_dir (for backward compatibility)')
    parser.add_argument('--notes', type=str, default='')
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

        # Optional seed override (keeps legacy behavior when not provided)
        if meta_args.seed is not None:
            args.seed = int(meta_args.seed)
            # Update config snapshot for reproducibility (best-effort)
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
        wandb.init(project=args.dataset_task, name=name,notes=meta_args.notes)

        # 初始化模型
        signal_processing_modules, feature_extractor_modules = config_network(configs,args)

        MODEL_DICT = {
            'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
            'TKAN': lambda args: Transparent_Signal_Processing_KAN(signal_processing_modules, feature_extractor_modules,args),
            'NNSPN': lambda args: NN_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
            'TFON': lambda args: Time_Frequency_Operator_Network(signal_processing_modules, feature_extractor_modules,args),
            'Fusion1D2D': lambda args: Fusion1D2D(signal_processing_modules, feature_extractor_modules,args),
            'MoE': lambda args: MoE(signal_processing_modules, feature_extractor_modules,args),
            'OperatorAttention': lambda args: OperatorAttentionModel(signal_processing_modules, feature_extractor_modules,args),
            'FuzzyLogic': lambda args: FuzzyLogicNetwork(signal_processing_modules, feature_extractor_modules,args),
            'FuzzyLogicV2': lambda args: FuzzyLogicV2Model(signal_processing_modules, feature_extractor_modules,args),
        }

        model_plain = MODEL_DICT[args.model](args)

        # network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)
        #model trainer #
        model = Basic_plmodel(model_plain, args)
        model_structure = print(model.network)
        trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)

        # train
        trainer.fit(model,train_dataloader, val_dataloader) # TODO load best checkpoint

        model = load_best_model_checkpoint(model,trainer)

        result = trainer.test(model,test_dataloader)

        # 保存结果
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)

        # 写入统一 schema（run_meta.yaml + metrics.json + artifacts/）
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
                cmd = " ".join([f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}" if os.getenv("CUDA_VISIBLE_DEVICES") else "", "python", "main.py", "--config_dir", str(config_dir)]).strip()

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
                    notes=str(meta_args.notes or ""),
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
