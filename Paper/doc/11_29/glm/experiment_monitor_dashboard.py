#!/usr/bin/env python3
"""
统一故障诊断实验监控仪表板
实时监控三个统一基线实验的训练进度和性能表现
"""

import os
import time
import re
import json
from datetime import datetime
from pathlib import Path
import subprocess
import threading

class ExperimentMonitor:
    def __init__(self):
        self.experiments = {
            'Fusion1D2D': {
                'gpu_id': 3,
                'config': 'configs/unified_baseline/config_Fusion1D2D.yaml',
                'shell_id': None,
                'status': 'initializing',
                'current_epoch': 0,
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'val_acc': 0.0,
                'train_acc': 0.0,
                'l1_loss': 0.0,
                'best_val_loss': float('inf'),
                'progress': 0.0,
                'eta': 'N/A',
                'errors': []
            },
            'MoE': {
                'gpu_id': 4,
                'config': 'configs/unified_baseline/config_MoE.yaml',
                'shell_id': None,
                'status': 'initializing',
                'current_epoch': 0,
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'val_acc': 0.0,
                'train_acc': 0.0,
                'l1_loss': 0.0,
                'best_val_loss': float('inf'),
                'progress': 0.0,
                'eta': 'N/A',
                'errors': []
            },
            'OperatorAttention': {
                'gpu_id': 7,
                'config': 'configs/unified_baseline/config_OperatorAttention.yaml',
                'shell_id': None,
                'status': 'initializing',
                'current_epoch': 0,
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'val_acc': 0.0,
                'train_acc': 0.0,
                'l1_loss': 0.0,
                'best_val_loss': float('inf'),
                'progress': 0.0,
                'eta': 'N/A',
                'errors': []
            }
        }
        self.max_epochs = 100
        self.start_time = time.time()

    def parse_log_line(self, line):
        """解析日志行提取训练指标"""
        # 匹配epoch信息
        epoch_match = re.search(r'Epoch (\d+):', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))

        # 匹配损失信息
        train_loss_match = re.search(r'train_loss_step=([\d.]+e[-+]?\d+|[\d.]+)', line)
        val_loss_match = re.search(r'val_loss=([\d.]+)', line)
        train_acc_match = re.search(r'train_acc_step=([\d.]+)', line)
        val_acc_match = re.search(r'val_acc=([\d.]+)', line)
        l1_loss_match = re.search(r'l1_loss__step=([\d.]+e[-+]?\d+)', line)

        # 匹配进度信息
        progress_match = re.search(r'(\d+)%\|', line)

        result = {
            'epoch': epoch_match.group(1) if epoch_match else None,
            'train_loss': float(train_loss_match.group(1)) if train_loss_match else None,
            'val_loss': float(val_loss_match.group(1)) if val_loss_match else None,
            'train_acc': float(train_acc_match.group(1)) if train_acc_match else None,
            'val_acc': float(val_acc_match.group(1)) if val_acc_match else None,
            'l1_loss': float(l1_loss_match.group(1)) if l1_loss_match else None,
            'progress': int(progress_match.group(1)) / 100 if progress_match else None
        }

        return result

    def update_experiment_status(self, exp_name, log_output):
        """更新实验状态"""
        exp = self.experiments[exp_name]

        try:
            lines = log_output.split('\n')
            for line in reversed(lines[-50:]):  # 只检查最近50行
                if 'Epoch' in line and ('train_loss' in line or 'val_loss' in line):
                    metrics = self.parse_log_line(line)

                    if metrics['epoch']:
                        exp['current_epoch'] = int(metrics['epoch'])
                    if metrics['train_loss']:
                        exp['train_loss'] = metrics['train_loss']
                    if metrics['val_loss']:
                        exp['val_loss'] = metrics['val_loss']
                        if metrics['val_loss'] < exp['best_val_loss']:
                            exp['best_val_loss'] = metrics['val_loss']
                    if metrics['val_acc']:
                        exp['val_acc'] = metrics['val_acc']
                    if metrics['train_acc']:
                        exp['train_acc'] = metrics['train_acc']
                    if metrics['l1_loss']:
                        exp['l1_loss'] = metrics['l1_loss']
                    if metrics['progress']:
                        exp['progress'] = metrics['progress']

                    break

            # 检查错误
            if 'Error' in log_output or 'Traceback' in log_output or 'FAILED' in log_output:
                exp['status'] = 'error'
                exp['errors'].append(log_output.split('Error')[-1].strip()[:200])
            elif exp['current_epoch'] > 0:
                exp['status'] = 'training'

            # 计算ETA
            if exp['current_epoch'] > 0:
                elapsed = time.time() - self.start_time
                time_per_epoch = elapsed / max(exp['current_epoch'], 1)
                remaining_epochs = self.max_epochs - exp['current_epoch']
                eta_seconds = remaining_epochs * time_per_epoch
                exp['eta'] = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

        except Exception as e:
            exp['status'] = 'error'
            exp['errors'].append(f"解析错误: {str(e)}")

    def get_wandb_links(self):
        """获取WandB链接"""
        links = {}
        try:
            # 列出wandb目录
            wandb_dir = Path("wandb")
            if wandb_dir.exists():
                for run_dir in wandb_dir.glob("run-20251128_*"):
                    # 从运行目录中提取模型信息
                    try:
                        with open(run_dir / "wandb-metadata.json", 'r') as f:
                            metadata = json.load(f)
                        run_name = metadata.get('name', '')
                        run_id = metadata.get('id', '')

                        # 确定模型类型
                        for exp_name in self.experiments.keys():
                            if exp_name.lower() in run_name.lower():
                                links[exp_name] = f"https://wandb.ai/PHM_bench/THU_018_basic/runs/{run_id}"
                                break
                    except:
                        continue
        except Exception as e:
            print(f"获取WandB链接时出错: {e}")

        return links

    def display_dashboard(self):
        """显示监控仪表板"""
        os.system('clear')

        print("=" * 120)
        print("🚀 统一故障诊断实验监控仪表板")
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  运行时间: {int((time.time() - self.start_time) // 60)}m {int((time.time() - self.start_time) % 60)}s")
        print("=" * 120)

        # 获取WandB链接
        wandb_links = self.get_wandb_links()

        # 表头
        print(f"{'模型名称':<20} {'GPU':<5} {'状态':<12} {'Epoch':<8} {'训练损失':<12} {'验证损失':<12} {'验证精度':<10} {'训练精度':<10} {'进度':<8} {'ETA':<10}")
        print("-" * 120)

        # 显示每个实验的状态
        for exp_name, exp in self.experiments.items():
            # 状态图标
            status_icon = "🔄" if exp['status'] == 'training' else "⚠️" if exp['status'] == 'error' else "⏳"
            status_text = f"{status_icon} {exp['status']}"

            # 格式化损失值
            train_loss = f"{exp['train_loss']:.4f}" if exp['train_loss'] != float('inf') else "N/A"
            val_loss = f"{exp['val_loss']:.4f}" if exp['val_loss'] != float('inf') else "N/A"

            # 格式化精度
            val_acc = f"{exp['val_acc']:.3f}" if exp['val_acc'] > 0 else "N/A"
            train_acc = f"{exp['train_acc']:.3f}" if exp['train_acc'] > 0 else "N/A"

            # 格式化进度
            progress = f"{exp['progress']*100:.1f}%" if exp['progress'] > 0 else "N/A"

            print(f"{exp_name:<20} {exp['gpu_id']:<5} {status_text:<12} {exp['current_epoch']:<8} "
                  f"{train_loss:<12} {val_loss:<12} {val_acc:<10} {train_acc:<10} {progress:<8} {exp['eta']:<10}")

        print("\n" + "=" * 120)

        # 显示最佳结果
        print("📊 当前最佳结果:")
        for exp_name, exp in self.experiments.items():
            if exp['best_val_loss'] != float('inf'):
                print(f"  {exp_name}: 验证损失 = {exp['best_val_loss']:.4f}, 验证精度 = {exp['val_acc']:.3f}")

        # 显示WandB链接
        if wandb_links:
            print("\n🔗 WandB监控链接:")
            for exp_name, link in wandb_links.items():
                print(f"  {exp_name}: {link}")

        # 显示错误信息
        has_errors = any(exp['errors'] for exp in self.experiments.values())
        if has_errors:
            print("\n⚠️ 错误信息:")
            for exp_name, exp in self.experiments.items():
                if exp['errors']:
                    print(f"  {exp_name}: {exp['errors'][-1]}")

        print("=" * 120)

    def monitor_continuously(self):
        """连续监控"""
        try:
            while True:
                # 这里应该从实际的bash输出中获取日志
                # 由于我们无法直接访问之前的bash输出，这里使用模拟数据
                # 在实际使用中，需要集成bash输出获取机制

                self.display_dashboard()
                time.sleep(10)  # 每10秒更新一次

        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    """主函数"""
    monitor = ExperimentMonitor()

    print("启动实验监控仪表板...")
    print("按 Ctrl+C 停止监控")

    try:
        monitor.monitor_continuously()
    except KeyboardInterrupt:
        print("\n\n实验监控结束。")
        print("感谢使用统一故障诊断实验监控仪表板！")

if __name__ == "__main__":
    main()