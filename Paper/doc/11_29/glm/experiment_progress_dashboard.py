#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验进度仪表板
实时跟踪统一基线实验进度

创建时间: 2025-11-29
"""

import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class ExperimentTracker:
    def __init__(self):
        self.experiments = {
            "FuzzyLogic": {
                "gpu": 1,
                "config": "configs/unified_baseline/config_FuzzyLogic.yaml",
                "status": "running",
                "start_time": None,
                "current_epoch": 0,
                "best_val_acc": 0.0,
                "l1_loss": "N/A",
                "notes": "L1损失过大问题已缓解，完成30轮训练"
            },
            "OperatorAttention": {
                "gpu": 2,
                "config": "configs/unified_baseline/config_OperatorAttention.yaml",
                "status": "running",
                "start_time": None,
                "current_epoch": 0,
                "best_val_acc": 0.0,
                "l1_loss": "N/A",
                "notes": "L1从0.0001降至0.00001进行调优实验"
            },
            "TSPN": {
                "gpu": 0,
                "config": "configs/unified_baseline/config_TSPN.yaml",
                "status": "completed",
                "start_time": None,
                "current_epoch": 30,
                "best_val_acc": 0.9524,
                "l1_loss": "N/A",
                "notes": "基线实验已完成，性能优秀"
            },
            "Fusion1D2D": {
                "gpu": 0,
                "config": "configs/unified_baseline/config_Fusion1D2D.yaml",
                "status": "completed",
                "start_time": None,
                "current_epoch": 30,
                "best_val_acc": 0.9468,
                "l1_loss": "N/A",
                "notes": "可视化已完成，性能略低于TSPN"
            },
            "MoE": {
                "gpu": 0,
                "config": "configs/unified_baseline/config_MoE.yaml",
                "status": "visualization_completed",
                "start_time": None,
                "current_epoch": 30,
                "best_val_acc": 0.9385,
                "l1_loss": "N/A",
                "notes": "专家可解释性分析已完成，生成5张图表"
            }
        }

        self.visualizations = {
            "1D-2D Fusion": {
                "status": "completed",
                "files": [
                    "Paper/1D-2D_fusion_explainable/results/performance_comparison.png",
                    "Paper/1D-2D_fusion_explainable/results/contribution_heatmap.png",
                    "Paper/1D-2D_fusion_explainable/results/attention_weights.png"
                ],
                "description": "性能对比、模态贡献热力图、注意力权重"
            },
            "MoE": {
                "status": "completed",
                "files": [
                    "Paper/MOE_explainable/results/expert_activation_heatmap.png",
                    "Paper/MOE_explainable/results/expert_utilization_analysis.png",
                    "Paper/MOE_explainable/results/gating_weights_distribution.png",
                    "Paper/MOE_explainable/results/load_balancing_analysis.png",
                    "Paper/MOE_explainable/results/path_signature_visualization.png"
                ],
                "description": "专家激活、利用率分析、门控权重、负载均衡、路径签名"
            },
            "OperatorAttention": {
                "status": "pending",
                "files": [],
                "description": "算子权重可视化、性能对比分析"
            },
            "FuzzyLogic": {
                "status": "pending",
                "files": [],
                "description": "模糊规则可视化、隶属度函数分析"
            }
        }

    def print_dashboard(self):
        """打印实验进度仪表板"""
        print("\n" + "="*80)
        print(f"🔬 统一基线实验进度仪表板 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # 实验状态
        print("\n📊 **实验执行状态**")
        print("-"*60)
        print(f"{'模型名称':<20} {'GPU':<5} {'状态':<15} {'当前轮次':<10} {'最佳验证准确率':<15}")
        print("-"*60)

        for name, exp in self.experiments.items():
            status_emoji = {"running": "🟢", "completed": "✅", "pending": "⏳", "visualization_completed": "📊"}.get(exp["status"], "❓")
            print(f"{status_emoji} {name:<18} {exp['gpu']:<5} {exp['status']:<15} {exp['current_epoch']:<10} {exp['best_val_acc']:<15.4f}")

        # 可视化状态
        print(f"\n🎨 **可视化完成状态**")
        print("-"*60)
        for name, viz in self.visualizations.items():
            status_emoji = {"completed": "✅", "pending": "⏳", "in_progress": "🔄"}.get(viz["status"], "❓")
            file_count = len(viz["files"])
            print(f"{status_emoji} {name}: {file_count}个文件 - {viz['description']}")

        # 下一步任务
        print(f"\n📋 **待完成任务**")
        print("-"*60)
        pending_tasks = [
            "1. 监控FuzzyLogic和OperatorAttention实验完成",
            "2. 创建OperatorAttention算子权重可视化",
            "3. 创建FuzzyLogic模糊规则可视化",
            "4. 对所有5个模型进行3次seed稳定性测试",
            "5. 汇总统一基线v1实验结果表",
            "6. 准备实验型论文的最小可发表实验集"
        ]
        for task in pending_tasks:
            print(f"⏳ {task}")

        # 数据文件汇总
        print(f"\n📁 **生成的关键文件**")
        print("-"*60)
        result_dirs = [
            "Paper/1D-2D_fusion_explainable/results/",
            "Paper/MOE_explainable/results/"
        ]

        for dir_path in result_dirs:
            if Path(dir_path).exists():
                files = list(Path(dir_path).glob("*.png"))
                print(f"📊 {dir_path}: {len(files)}个PNG文件")

        print("\n" + "="*80)

    def get_current_best_results(self):
        """获取当前最佳结果汇总"""
        results = {}
        for name, exp in self.experiments.items():
            if exp["best_val_acc"] > 0:
                results[name] = exp["best_val_acc"]

        # 按性能排序
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print(f"\n🏆 **当前模型性能排名** (验证准确率)")
        print("-"*50)
        for i, (name, acc) in enumerate(sorted_results, 1):
            medal = ["🥇", "🥈", "🥉"][min(i-1, 2)] if i <= 3 else f"{i}."
            print(f"{medal} {name:<20}: {acc:.4f}")

        return sorted_results

    def check_gpu_status(self):
        """检查GPU使用状态"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            print(f"\n💻 **GPU状态**")
            print("-"*50)
            print("nvidia-smi输出:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'python' in line or any(exp['gpu'] == 0 and 'CUDA_VISIBLE_DEVICES=0' in line
                                        for exp in self.experiments.values()):
                    print(f"  {line.strip()}")
        except:
            print("无法获取GPU状态")

    def save_progress_report(self):
        """保存进度报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"experiment_progress_report_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "experiments": self.experiments,
            "visualizations": self.visualizations
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 进度报告已保存到: {report_file}")

def main():
    """主函数"""
    tracker = ExperimentTracker()

    # 显示仪表板
    tracker.print_dashboard()

    # 显示性能排名
    tracker.get_current_best_results()

    # 检查GPU状态（可选）
    # tracker.check_gpu_status()

    # 保存进度报告
    tracker.save_progress_report()

if __name__ == "__main__":
    main()