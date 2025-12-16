# PHM‑Vibench 数据集目录（来自 `metadata_6_11.xlsx`，供本仓库统一引用）

> 本文件用于把 Vibench 的 `Dataset_id ↔ Name` 映射“落盘到本仓库”，便于 6 篇 Paper 在 `run_meta.yaml / metrics.json` 与论文正文中引用一致的数据集标识。  
> 数据真源：`/home/user/data/PHMbenchdata/PHM-Vibench/metadata_6_11.xlsx`（字段：`Dataset_id`, `Name`）。  
> 数据集细节（采样率/负载/故障类型等）请参考上游：`/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/README.md`。

## Dataset_id → Name

| Dataset_id | Name |
|---:|---|
| 1 | RM_001_CWRU |
| 2 | RM_002_XJTU |
| 3 | RM_003_FEMTO |
| 4 | RM_004_IMS |
| 5 | RM_005_Ottawa23 |
| 6 | RM_006_THU |
| 7 | RM_007_MFPT |
| 8 | RM_008_UNSW |
| 9 | RM_010_SEU |
| 11 | RM_015_susu |
| 12 | RM_016_JNU |
| 13 | RM_017_Ottawa19 |
| 14 | RM_018_THU24 |
| 15 | RM_010_SEU |
| 16 | RM_020_DIRG |
| 17 | RM_023_HIT23 |
| 18 | RM_024_JUST |
| 19 | RM_031_HUST24 |
| 20 | RM_027_PU |

## 统一引用建议（强烈推荐）

- 在 `run_meta.yaml` 中：
  - `run.dataset_id` 使用 **Name**（例如 `RM_001_CWRU`），便于跨论文阅读；
  - 额外增加 `run.dataset_numeric_id`（例如 `1`），便于与 metadata 直接对齐（推荐字段）。
- 在 config 中（`vbench_config.dataset_ids`）继续使用 numeric id（VbenchDataset 的真实筛选条件）。

