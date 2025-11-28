# model_collection 目录说明（中文）

本目录存放 **对比模型（baseline models）**，用于与主方法（TSPN、Fusion1D2D、MoE 等）做统一性能对比。

- 典型模型：  
  - `Resnet.py`：ResNet 深度卷积网络。  
  - `Sincnet.py`：基于 Sinc 滤波器的 CNN。  
  - `WKN.py`、`MCN.py`、`TFN.py` 等信号处理型模型。  
  - `EELM.py`、`F_EQL.py` 等符号/极限学习机方法。  
- 某些模型有对应子目录（如 `MCN/`、`TFN/`）存放变体与附加代码。  

统一基线实验、论文对比表应优先从这里选择基线模型。  

