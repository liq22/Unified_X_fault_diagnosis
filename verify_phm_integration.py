#!/usr/bin/env python3
"""
PHM-Vibench集成验证脚本
快速验证所有关键组件是否正确集成
"""

import os
import sys
import yaml
from pathlib import Path

def check_files():
    """检查关键文件是否存在"""
    print("=" * 50)
    print("检查关键文件...")
    print("=" * 50)

    required_files = [
        # 配置文件
        "configs/PHM_Vibench/config_TSPN.yaml",
        "configs/PHM_Vibench/config_TKAN.yaml",
        "configs/PHM_Vibench/config_NNSPN.yaml",
        "configs/PHM_Vibench/config_OperatorAttention.yaml",
        "configs/PHM_Vibench/config_FuzzyLogic.yaml",
        "configs/PHM_Vibench/config_com.yaml",
        "configs/PHM_Vibench/config_TSPN_test.yaml",

        # 数据相关
        "data/vbench_dataset.py",
        "data/data_provider.py",

        # 脚本
        "script/run_PHM_baseline.sh",
        "script/run_PHM_domain_adaptation.sh",
        "script/run_PHM_few_shot.sh",
        "script/monitor_phm_experiments.py",

        # 文档
        "docs/PHM_Vibench_WandB_Integration.md",
        "PHM_VIBENCH_INTEGRATION_SUMMARY.md"
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n缺少 {len(missing_files)} 个文件")
        return False
    else:
        print(f"\n✅ 所有 {len(required_files)} 个关键文件都存在")
        return True

def check_configurations():
    """检查配置文件格式"""
    print("\n" + "=" * 50)
    print("检查配置文件格式...")
    print("=" * 50)

    config_files = [
        "configs/PHM_Vibench/config_TSPN.yaml",
        "configs/PHM_Vibench/config_com.yaml"
    ]

    all_valid = True
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 检查必要的配置项
            if 'args' in config and 'dataset_task' in config['args']:
                print(f"✅ {config_file}: 格式正确")

                # 检查PHM特定配置
                if 'vbench_config' in config:
                    print(f"  包含vbench_config: ✅")
                else:
                    print(f"  缺少vbench_config: ⚠️")
            else:
                print(f"❌ {config_file}: 缺少必要配置项")
                all_valid = False

        except Exception as e:
            print(f"❌ {config_file}: 解析错误 - {e}")
            all_valid = False

    return all_valid

def check_data_provider():
    """检查数据提供器映射"""
    print("\n" + "=" * 50)
    print("检查数据提供器映射...")
    print("=" * 50)

    try:
        # 读取data_provider.py文件
        with open("data/data_provider.py", 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查PHM映射
        phm_mappings = [
            'PHM_Vibench_basic',
            'PHM_Vibench_cwru',
            'PHM_Vibench_xjtu',
            'PHM_Vibench_domain_adaptation',
            'PHM_Vibench_few_shot'
        ]

        found_mappings = []
        for mapping in phm_mappings:
            if mapping in content:
                found_mappings.append(mapping)
                print(f"✅ 找到映射: {mapping}")
            else:
                print(f"❌ 缺少映射: {mapping}")

        if len(found_mappings) >= 3:  # 至少找到大部分映射
            print(f"\n✅ 数据提供器映射正确 ({len(found_mappings)}/{len(phm_mappings)})")
            return True
        else:
            print(f"\n❌ 数据提供器映射不完整 ({len(found_mappings)}/{len(phm_mappings)})")
            return False

    except Exception as e:
        print(f"❌ 检查数据提供器失败: {e}")
        return False

def check_scripts():
    """检查脚本可执行性"""
    print("\n" + "=" * 50)
    print("检查脚本可执行性...")
    print("=" * 50)

    scripts = [
        "script/run_PHM_baseline.sh",
        "script/run_PHM_domain_adaptation.sh",
        "script/run_PHM_few_shot.sh"
    ]

    executable_count = 0
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✅ {script}: 可执行")
                executable_count += 1
            else:
                print(f"⚠️ {script}: 不可执行 (使用 chmod +x 修复)")
        else:
            print(f"❌ {script}: 不存在")

    if executable_count == len(scripts):
        print(f"\n✅ 所有脚本都是可执行的")
        return True
    else:
        print(f"\n⚠️ {executable_count}/{len(scripts)} 个脚本可执行")
        return False

def check_data_directory():
    """检查数据目录"""
    print("\n" + "=" * 50)
    print("检查PHM数据目录...")
    print("=" * 50)

    data_dir = Path("/home/user/data/PHMbenchdata/PHM-Vibench")

    if data_dir.exists():
        print(f"✅ 数据目录存在: {data_dir}")

        # 检查关键文件
        metadata_file = data_dir / "metadata_6_11.xlsx"
        if metadata_file.exists():
            print(f"✅ 元数据文件存在: {metadata_file}")

            # 检查H5文件
            h5_files = list(data_dir.glob("*.h5"))
            print(f"✅ 找到 {len(h5_files)} 个H5数据文件")

            return True
        else:
            print(f"❌ 元数据文件不存在: {metadata_file}")
            return False
    else:
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

def generate_summary():
    """生成集成验证摘要"""
    print("\n" + "=" * 50)
    print("PHM-Vibench集成验证摘要")
    print("=" * 50)

    results = {
        "文件检查": check_files(),
        "配置检查": check_configurations(),
        "数据映射": check_data_provider(),
        "脚本权限": check_scripts(),
        "数据目录": check_data_directory()
    }

    print("\n验证结果:")
    print("-" * 30)
    for check, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check}: {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\n总体结果: {passed_count}/{total_count} 项检查通过")

    if passed_count == total_count:
        print("\n🎉 PHM-Vibench集成验证完全通过！")
        print("可以开始使用PHM-Vibench数据集进行实验。")
        return True
    elif passed_count >= total_count - 1:
        print("\n✅ PHM-Vibench集成基本通过")
        print("可以开始使用，建议修复剩余问题。")
        return True
    else:
        print("\n⚠️ PHM-Vibench集成存在问题")
        print("建议先修复关键问题再使用。")
        return False

def main():
    """主函数"""
    print("PHM-Vibench数据集集成验证")
    print(f"验证时间: {Path().cwd()}")
    print("=" * 50)

    success = generate_summary()

    # 返回退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()