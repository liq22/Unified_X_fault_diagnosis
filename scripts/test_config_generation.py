#!/usr/bin/env python3
"""
测试配置文件生成是否正确
"""

import yaml
import pandas as pd

# 测试配置生成
def test_config_generation():
    # 读取基础配置
    with open('configs/vbench/config_vbench_diagnosis.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    print("=== 基础配置读取成功 ===")

    # 测试为dataset_id=1生成配置
    config = base_config.copy()
    config['vbench_config']['dataset_ids'] = [1]

    # 检查生成的配置
    print("\n=== 生成的配置 ===")
    print(f"Dataset IDs: {config['vbench_config']['dataset_ids']}")
    print(f"Data dir: {config['vbench_config']['data_dir']}")
    print(f"Metadata file: {config['vbench_config']['metadata_file']}")

    # 验证metadata路径
    import os
    metadata_path = config['vbench_config']['data_dir'] + '/' + config['vbench_config']['metadata_file']
    print(f"\nMetadata路径: {metadata_path}")
    print(f"文件存在: {os.path.exists(metadata_path)}")

    # 测试读取metadata
    if os.path.exists(metadata_path):
        metadata = pd.read_excel(metadata_path)
        dataset_1_data = metadata[metadata['Dataset_id'] == 1]
        print(f"\nDataset_id=1 的数据量: {len(dataset_1_data)}")
        print(f"Label 列值: {dataset_1_data['Label'].unique()}")
        print(f"类别数: {len(dataset_1_data['Label'].unique())}")
    else:
        print("\n错误: metadata文件不存在")

    # 保存测试配置
    with open('test_config_1.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("\n测试配置已保存到: test_config_1.yaml")

if __name__ == "__main__":
    test_config_generation()