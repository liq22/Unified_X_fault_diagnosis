# data 目录说明（中文）

本目录包含 **数据加载与预处理** 相关代码，是所有实验共享的数据层基础设施。

- `vbench_dataset.py`：推荐使用的数据集封装（VBench 风格），统一不同数据集的访问接口。  
- `vbench_utils.py`：与 VBench 数据/评估相关的辅助函数（元数据检查、切分策略等）。  
- `data_provider.py` / `datasets.py`：旧版数据接口与数据集定义，后续逐步迁移到 VBench 体系。  
- `utils.py`：通用数据工具函数（路径处理、缓存等）。  

新方法/新 Paper 建议优先复用 `vbench_dataset.py` + `vbench_utils.py`，不要在子目录重复实现 Dataset。  

