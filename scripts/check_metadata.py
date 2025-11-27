#!/usr/bin/env python3
"""
验证metadata中Dataset_id到Name的映射关系
"""
import pandas as pd
import os

# 读取metadata
metadata_path = "/home/user/data/PHMbenchdata/PHM-Vibench/metadata_6_11.xlsx"
df = pd.read_excel(metadata_path)

# 获取唯一的Dataset_id到Name的映射
dataset_mapping = df[['Dataset_id', 'Name']].drop_duplicates().sort_values('Dataset_id')

print("Dataset_id 到 Name 的映射关系:")
print("=" * 50)
for _, row in dataset_mapping.iterrows():
    dataset_id = int(row['Dataset_id'])
    name = row['Name']
    h5_file = f"/home/user/data/PHMbenchdata/PHM-Vibench/{name}.h5"
    exists = os.path.exists(h5_file)

    print(f"Dataset_id: {dataset_id:2d} → Name: {name:<20} → H5文件: {'✅' if exists else '❌'}")

print(f"\n总共有 {len(dataset_mapping)} 个数据集")
existing_files = sum(1 for _, row in dataset_mapping.iterrows()
                    if os.path.exists(f"/home/user/data/PHMbenchdata/PHM-Vibench/{row['Name']}.h5"))
print(f"对应的H5文件存在: {existing_files}/{len(dataset_mapping)}")
