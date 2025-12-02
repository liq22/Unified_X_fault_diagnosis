#!/usr/bin/env python3
"""
PHM-Vibench实验监控脚本
实时监控实验进度和资源使用情况
"""

import os
import sys
import time
import psutil
import GPUtil
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb未安装，将使用本地监控")

class PHMExperimentMonitor:
    def __init__(self):
        self.log_dir = Path("logs/PHM_monitor")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()

    def log_system_status(self):
        """记录系统状态"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU信息
        gpu_info = "N/A"
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                gpu_info = f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}/{gpu.memoryTotal}MB"
        except:
            pass

        # 创建日志条目
        log_entry = f"""
{timestamp}
CPU使用率: {cpu_percent}%
内存使用: {memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB ({memory.percent}%)
磁盘使用: {disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB ({disk.percent}%)
{gpu_info}
----------------------------------------
"""

        # 写入日志文件
        with open(self.log_dir / "system_status.log", "a") as f:
            f.write(log_entry)

        return log_entry

    def monitor_wandb_experiments(self):
        """监控W&B实验"""
        if not WANDB_AVAILABLE:
            print("W&B不可用，跳过实验监控")
            return

        try:
            api = wandb.Api()

            # 监控的项目列表
            projects = [
                "PHM-Vibench-Unified-Baseline",
                "PHM-Vibench-Test",
                "PHM-Domain-Adaptation",
                "PHM-Few-Shot"
            ]

            for project_name in projects:
                try:
                    project = api.project(project_name)
                    runs = list(project.runs())

                    # 只显示最近24小时的运行
                    recent_runs = [r for r in runs if
                                 (datetime.now() - r.created_at).total_seconds() < 24*3600]

                    if recent_runs:
                        print(f"\n项目: {project_name}")
                        print("-" * 50)

                        for run in recent_runs:
                            status = "🟢 运行中" if run.state == "running" else \
                                    "🟡 完成" if run.state == "finished" else \
                                    "🔴 失败"

                            print(f"{run.name}: {status}")
                            if run.state == "running":
                                print(f"  运行时间: {datetime.now() - run.created_at}")
                                if run.summary.get("val_loss"):
                                    print(f"  当前验证损失: {run.summary['val_loss']:.4f}")

                except Exception as e:
                    print(f"无法访问项目 {project_name}: {e}")

        except Exception as e:
            print(f"W&B监控失败: {e}")

    def check_log_files(self):
        """检查实验日志文件"""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return

        # 查找最近的实验日志
        recent_logs = []
        for log_dir in logs_dir.glob("PHM_*"):
            if log_dir.is_dir():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=os.path.getctime)
                    recent_logs.append((log_dir.name, latest_log))

        if recent_logs:
            print("\n最近实验日志:")
            print("-" * 50)
            for log_name, log_file in recent_logs[-5:]:  # 显示最近5个
                mod_time = datetime.fromtimestamp(os.path.getctime(log_file))
                size = log_file.stat().st_size / 1024 / 1024  # MB
                print(f"{log_name}: {mod_time.strftime('%H:%M:%S')} ({size:.1f}MB)")

    def run_continuous_monitor(self, interval=60):
        """连续监控模式"""
        print("开始PHM-Vibench实验连续监控")
        print(f"监控间隔: {interval}秒")
        print(f"日志目录: {self.log_dir}")
        print("按 Ctrl+C 停止监控")
        print("=" * 60)

        try:
            while True:
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 40)

                # 系统状态
                self.log_system_status()

                # W&B实验监控
                if WANDB_AVAILABLE:
                    self.monitor_wandb_experiments()

                # 检查本地日志
                self.check_log_files()

                # 等待下一个监控周期
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n监控已停止")

    def run_once(self):
        """运行一次监控"""
        print("PHM-Vibench实验状态报告")
        print("=" * 60)
        print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

        # 系统状态
        print("系统状态:")
        self.log_system_status()

        # W&B实验监控
        if WANDB_AVAILABLE:
            print("\nW&B实验状态:")
            self.monitor_wandb_experiments()

        # 本地日志检查
        self.check_log_files()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PHM-Vibench实验监控")
    parser.add_argument("--interval", type=int, default=60,
                       help="连续监控间隔(秒)，默认60")
    parser.add_argument("--once", action="store_true",
                       help="只运行一次监控")

    args = parser.parse_args()

    monitor = PHMExperimentMonitor()

    if args.once:
        monitor.run_once()
    else:
        monitor.run_continuous_monitor(args.interval)

if __name__ == "__main__":
    main()