#!/usr/bin/env python3
"""Generate radar visualizations for the TSPN signal pipeline."""

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from configs.config import config_network
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel


def load_config(config_path: Path) -> tuple[dict, SimpleNamespace]:
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    args = SimpleNamespace(**config['args'])
    return config, args


def build_network(config_path: Path, checkpoint_path: Path, device: torch.device) -> Transparent_Signal_Processing_Network:
    config, args = load_config(config_path)
    args.device = device.type
    signal_processing_modules, feature_extractor_modules = config_network(config, args)
    network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules, args)
    lightning_wrapper = Basic_plmodel(network, args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    lightning_wrapper.load_state_dict(checkpoint['state_dict'])
    network = lightning_wrapper.network.to(device)
    network.eval()
    return network


def reduce_batch(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'mean':
        return values.mean(dim=0)
    if reduction == 'median':
        return values.median(dim=0).values
    return values


def stage_input_metrics(stage_tensor: torch.Tensor, reduction: str) -> OrderedDict:
    magnitudes = torch.sqrt((stage_tensor.abs() ** 2).mean(dim=1))
    reduced = reduce_batch(magnitudes, reduction)
    result = OrderedDict()
    for idx in range(reduced.shape[0]):
        result[f'Ch{idx}'] = float(reduced[idx].item())
    return result


def stage_module_metrics(stage_tensor: torch.Tensor, layer, reduction: str) -> OrderedDict:
    module_names = list(layer.signal_processing_modules.keys())
    channels_per_module = stage_tensor.shape[-1] // max(len(module_names), 1)
    result = OrderedDict()
    for module_idx, module_name in enumerate(module_names):
        start = module_idx * channels_per_module
        end = start + channels_per_module
        module_tensor = stage_tensor[:, :, start:end]
        magnitudes = torch.sqrt((module_tensor.abs() ** 2).mean(dim=(1, 2)))
        reduced = reduce_batch(magnitudes, reduction)
        result[module_name] = float(reduced.item())
    return result


def stage_feature_metrics(feature_tensor: torch.Tensor, network, reduction: str) -> OrderedDict:
    feature_names = list(network.feature_extractor_layers.feature_extractor_modules.keys())
    feature_tensor = feature_tensor.clone().detach()
    if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    if feature_tensor.ndim == 2:
        feature_tensor = feature_tensor.unsqueeze(-1)
    channel_count = network.channel_for_feature
    feature_tensor = feature_tensor.view(feature_tensor.shape[0], len(feature_names), channel_count)
    magnitudes = torch.sqrt((feature_tensor.abs() ** 2).mean(dim=2))
    reduced = reduce_batch(magnitudes, reduction)
    result = OrderedDict()
    for idx, name in enumerate(feature_names):
        result[name] = float(reduced[idx].item())
    return result


def stage_classifier_metrics(logit_tensor: torch.Tensor, reduction: str, class_names: list[str]) -> OrderedDict:
    probabilities = torch.softmax(logit_tensor, dim=-1)
    reduced = reduce_batch(probabilities, reduction)
    result = OrderedDict()
    for idx, name in enumerate(class_names):
        result[name] = float(reduced[idx].item())
    return result


def collect_pipeline_outputs(network: Transparent_Signal_Processing_Network, signal_batch: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = OrderedDict()
    x = signal_batch.to(device)
    outputs['Input'] = x.detach().cpu()
    for idx, layer in enumerate(network.signal_processing_layers):
        x = layer(x)
        outputs[f'Layer{idx + 1}'] = x.detach().cpu()
    features = network.feature_extractor_layers(x)
    outputs['Features'] = features.detach().cpu()
    logits = network.clf(features)
    outputs['Logits'] = logits.detach().cpu()
    return outputs


def summarize_pipeline(outputs: dict[str, torch.Tensor], network: Transparent_Signal_Processing_Network, class_names: list[str], reduction: str) -> OrderedDict:
    summary = OrderedDict()
    summary['Input'] = stage_input_metrics(outputs['Input'], reduction)
    for idx, layer in enumerate(network.signal_processing_layers):
        key = f'Layer{idx + 1}'
        summary[key] = stage_module_metrics(outputs[key], layer, reduction)
    summary['Features'] = stage_feature_metrics(outputs['Features'], network, reduction)
    summary['Classifier'] = stage_classifier_metrics(outputs['Logits'], reduction, class_names)
    return summary


def normalize_summary(summary: OrderedDict) -> OrderedDict:
    normalized = OrderedDict()
    for stage, metrics in summary.items():
        values = np.array(list(metrics.values()), dtype=np.float64)
        scale = np.max(np.abs(values)) if values.size else 0.0
        stage_norm = OrderedDict()
        for key, value in metrics.items():
            if scale > 0.0:
                stage_norm[key] = float(value / scale)
            else:
                stage_norm[key] = float(value)
        normalized[stage] = stage_norm
    return normalized


def plot_stage_radars(summary: OrderedDict, title: str, output_path: Path, figure_format: str) -> None:
    plt.rcParams['font.family'] = 'Times New Roman'
    stages = list(summary.items())
    num_stages = len(stages)
    ncols = min(3, num_stages)
    nrows = math.ceil(num_stages / ncols)
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for index, (stage_name, metrics) in enumerate(stages, start=1):
        labels = list(metrics.keys())
        values = list(metrics.values())
        if not labels:
            continue
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        values += values[:1]
        ax = fig.add_subplot(nrows, ncols, index, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, values, color='#283D7E', linewidth=2)
        ax.fill(angles, values, color='#5291B2', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        radial_max = max(max(values), 1e-6)
        ax.set_ylim(0, radial_max)
        ax.set_title(stage_name, y=1.10)
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(f'.{figure_format}'), dpi=300)
    plt.close(fig)


def write_summary_csv(summary: OrderedDict, output_path: Path) -> None:
    lines = ['Stage,Metric,Value']
    for stage, metrics in summary.items():
        for name, value in metrics.items():
            lines.append(f'{stage},{name},{value}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Radar visualization pipeline for TSPN.')
    parser.add_argument('--config', type=Path, default=Path('configs/a_010_SEU/config_basic.yaml'), help='Path to the YAML config file.')
    parser.add_argument('--checkpoint', type=Path, default=Path('save/test/model_seu/tspn.ckpt'), help='Path to the trained checkpoint.')
    parser.add_argument('--data', type=Path, default=Path('E:/dataset/generate/SEU_bearing/SEU_bearing_20Hz_2_data.npy'), help='Path to the .npy signal array.')
    parser.add_argument('--labels', type=Path, default=Path('E:/dataset/generate/SEU_bearing/SEU_bearing_20Hz_2_label.npy'), help='Path to the .npy label array.')
    parser.add_argument('--samples', type=int, nargs='+', default=[25, 260, 510, 770, 1070], help='Sample indices to visualize.')
    parser.add_argument('--class-names', type=str, nargs='*', default=None, help='Optional class names aligned with label indices.')
    parser.add_argument('--output-dir', type=Path, default=Path('save/figure/seu/radar'), help='Directory to store generated figures.')
    parser.add_argument('--figure-format', type=str, default='png', choices=['png', 'svg', 'pdf'], help='Output format for figures.')
    parser.add_argument('--csv', action='store_true', help='Export stage statistics as CSV alongside figures.')
    parser.add_argument('--normalize', action='store_true', help='Normalize each radar stage by its maximum absolute value.')
    parser.add_argument('--batch-reduction', type=str, default='mean', choices=['mean', 'median', 'none'], help='Reduction applied across batch dimension.')
    parser.add_argument('--device', type=str, default=None, help='Force computation device, e.g. "cpu" or "cuda".')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    network = build_network(args.config, args.checkpoint, device)

    signals = np.load(args.data)
    labels = np.load(args.labels) if args.labels.exists() else None
    if hasattr(network.clf, 'clf'):
        class_count = network.clf.clf[-1].out_features
    else:
        class_count = network.clf.out_features
    if args.class_names and len(args.class_names) == class_count:
        class_names = args.class_names
    else:
        class_names = [f'Class{idx}' for idx in range(class_count)]

    torch.set_grad_enabled(False)

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= signals.shape[0]:
            print(f'Skip sample {sample_idx}: index out of range.')
            continue
        sample_batch = torch.from_numpy(signals[sample_idx:sample_idx + 1]).float()
        outputs = collect_pipeline_outputs(network, sample_batch, device)
        summary = summarize_pipeline(outputs, network, class_names, args.batch_reduction)
        plot_summary = normalize_summary(summary) if args.normalize else summary

        label_name = None
        if labels is not None and sample_idx < labels.shape[0]:
            label_idx = int(labels[sample_idx])
            if args.class_names and label_idx < len(args.class_names):
                label_name = args.class_names[label_idx]
            elif label_idx < len(class_names):
                label_name = class_names[label_idx]
        title = f'Sample {sample_idx}' + (f' | Label: {label_name}' if label_name else '')

        figure_path = args.output_dir / f'sample_{sample_idx}_radar'
        plot_stage_radars(plot_summary, title, figure_path, args.figure_format)

        if args.csv:
            csv_path = figure_path.with_suffix('.csv')
            write_summary_csv(summary, csv_path)


if __name__ == '__main__':
    main()
