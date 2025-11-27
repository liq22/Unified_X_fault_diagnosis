# configs 目录说明（中文）

本目录存放本仓库所有实验的 **配置文件**，用于控制数据集、模型结构、训练超参数等。

- 子目录按数据集/任务划分，例如：`a_018_THU/`、`a_006_THU/` 等。  
- 每个子目录下通常包含：  
  - 主模型配置（如 TSPN / NNSPN / Fusion1D2D / MoE 等）；  
  - 对比模型配置（model_collection 中的 ResNet、WKN、MCN、TFN 等）；  
  - 消融实验与特殊实验（k-shot、generalization）的配置。  
- 推荐新实验统一通过这里的 YAML/JSON 配置文件驱动，避免在代码中写死参数。  

