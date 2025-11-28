# trainer 目录说明（中文）

本目录提供本仓库统一的 **训练/验证/测试框架**。

- `trainer_basic.py`：核心训练循环（epoch/batch 级控制）。  
- `trainer_set.py`：训练配置与 Lightning/日志等集成部分。  
- `fusion_trainer.py`：针对 1D-2D 融合等特殊任务的训练逻辑。  
- `utils.py`：训练相关工具函数（损失封装、回调等）。  

建议：  
- 新模型优先复用这里的训练框架，通过配置文件切换模型与数据；  
- Paper 子项目避免在内部重复实现完整训练器。  

