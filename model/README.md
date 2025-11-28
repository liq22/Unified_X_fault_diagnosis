# model 目录说明（中文）

本目录包含本项目自建的 **核心模型与算子实现**，属于方法层的基础部分。

- 核心文件：  
  - `Signal_processing.py`：透明信号处理算子库（FFT、HT、WF、LNO 等）。  
  - `Feature_extract.py`：统计特征提取模块。  
  - `DEN.py`、`Fusion1D2D.py`、`kan.py` 等：不同结构的专用模型。  
  - `explainable_base.py` / `llm_explainable_base.py`：可解释模型与 LLM 增强模型的基类。  
- `doc/`：与模型相关的补充说明与设计文档。  

推荐：  
- 通用算子/模块优先放在此目录，Paper 目录只写“方法逻辑和实验配置”。  

