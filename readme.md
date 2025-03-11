# 仓库介绍

本仓库提供了一种通用的、可解释的故障诊断方法框架，基于该框架可以实现以下模型：

- TON
- TIFN
- DEN
- EOAN
- EQL
- MCN
- WKN
- EELM
- F_EQL

# 快速开始

## 环境配置

请按照以下命令创建并激活 Conda 环境：

```shell
conda env create -f environment.yml
conda activate your_environment_name  # 将 'your_environment_name' 替换为实际的环境名称
```

## 配置文件

在运行前，需要根据需求配置模型参数。配置文件位于：

```
Unified_X_fault_diagnosis/configs/THU_018/config_TSPN.yaml
```

以下是配置文件中主要参数的说明：

| 参数                          | 描述                                     | 示例值                                                           |
|-------------------------------|------------------------------------------|------------------------------------------------------------------|
| **signal_processing_configs** | 信号处理层的配置                         |                                                                  |
| layer1                        | 第一层信号处理模块                       | ['HT', 'WF', 'I']                                                |
| layer2                        | 第二层信号处理模块                       | ['HT', 'WF', 'I']                                                |
| layer3                        | 第三层信号处理模块                       | ['HT', 'WF', 'I']                                                |
| layer4                        | 第四层信号处理模块                       | ['HT', 'WF', 'I']                                                |
| **feature_extractor_configs** | 特征提取模块的配置                       | ['Mean', 'Std', 'Var', 'Entropy', 'Max', 'Min', 'AbsMean', ... ] |
| **args**                      | 其他参数配置                             |                                                                  |
| device                        | 运行设备                                 | 'cuda'                                                           |
| data_dir                      | 数据集路径                               | '/home/user/data/a_bearing/a_018_THU24_pro/'                     |
| dataset_task                  | 数据集任务名称                           | 'THU_018_basic'                                                  |
| target                        | 目标类型，数据集中的参数，设置目标域       | 'IF'                                                             |
| k_shot                        | 少样本学习中的样本数量                   | 64                                                               |
| **model**                     | 模型参数配置                             |                                                                  |
| model                         | 模型名称                                 | 'TSPN'                                                           |
| skip_connection               | 是否使用残差连接                         | true                                                             |
| num_classes                   | 分类数                                   | 5                                                                |
| in_dim                        | 输入维度                                 | 4096                                                             |
| out_dim                       | 输出维度                                 | 4096                                                             |
| in_channels                   | 输入通道数                               | 2                                                                |
| out_channels                  | 输出通道数                               | 3                                                                |
| scale                         | 缩放比例                                 | 4                                                                |
| f_c_mu                        | 滤波器中心频率初始化均值                         | 0                                                                |
| f_c_sigma                     | 滤波器中心频率初始化标准差                       | 0.1                                                              |
| f_b_mu                        | 偏置初始化均值                           | 0                                                                |
| f_b_sigma                     | 偏置初始化标准差                         | 0.1                                                              |
| **hyperparameter**            | 超参数设置                               |                                                                  |
| learning_rate                 | 学习率                                   | 0.001                                                            |
| batch_size                    | 批量大小                                 | 64                                                               |
| num_epochs                    | 训练轮数                                 | 300                                                              |
| weight_decay                  | 权重衰减系数                             | 0.0001                                                           |
| num_workers                   | 数据加载器的工作线程数                   | 32                                                               |
| seed                          | 随机种子                                 | 17                                                               |
| **train**                     | 训练参数配置                             |                                                                  |
| monitor                       | 监控指标                                 | 'val_loss'                                                       |
| patience                      | 提前停止的等待次数                       | 200                                                              |
| gpus                          | 使用的 GPU 数目                          | 1                                                                |
| l1_norm                       | L1 正则化系数                            | 0.01                                                             |
| pruning                       | 剪枝比例（可为 None）                    | None                                                             |
| snr                           | 信噪比                                   | 1                                                                |

## 运行示例

请使用以下命令运行模型：

```shell
python main.py --config_file configs/THU_018/config_TSPN.yaml
```

或者执行脚本：

```shell
./script/demo.sh
```

# 数据集任务映射

`DATASET_TASK_CLASS` 字典定义了数据集任务名称与对应的数据集类之间的映射关系：

```python
DATASET_TASK_CLASS = {
    'THU_006_basic': THU_006or018_basic,
    'THU_018_basic': THU_006or018_basic,
    'THU_018_few_shot': THU_006or018_few_shot,
    'THU_006_few_shot': THU_006or018_few_shot,
    'THU_006_generalization': THU_006_generalization
}
```

# 项目目录结构

```
├── .gitattributes
├── .gitignore
├── .vscode
│   ├── launch.json
│   └── settings.json
├── 1_paperfig
│   ├── figs/
│   └── TIIfigure/
├── configs
│   ├── config.py                # 项目的配置文件解析
│   ├── config_basic.yaml
│   ├── config_com.yaml          # 对比方法的配置
│   ├── THU_006/                 # 根据不同数据集的配置文件
│   ├── THU_018/
│   └── ...                      # 更多配置文件
├── data
│   ├── data_provider.py
│   ├── datasets.py
│   └── utils.py
├── main.py                      # 主程序入口
├── main_ablation_exp.py         # 消融实验，网格学习率
├── main_com.py                  # 对比方法
├── main_com_kshotexp.py         # K-shot 对比方法
├── main_kshotexp.py             # K-shot 实验
├── model                        # 自建模型目录
│   ├── Feature_extract.py
│   ├── Logic_inference.py
│   ├── parse_network.py         # 网络解析和可视化工具
│   ├── Signal_processing.py
│   ├── DEN.py                   # 改进的算子模型
│   ├── TSPN.py                  # TIFN 工作，多源信息融合
│   ├── NNSPN.py                 # EOAN 工作，加入 Attention
│   └── utils.py
├── model_collection             # 模型集合
│   ├── EELM.py                  # 工作
│   ├── F_EQL.py                 # EQL 工作
│   ├── MCN.py                   # MCN 工作 
│   ├── MWA_CNN.py
│   ├── Resnet.py
│   ├── Sincnet.py
│   ├── TFN.py
│   └── WKN.py
├── plot                         # 绘图文件夹
├── post                         # 后处理、结果分析
├── post_analysis.ipynb
├── readme.md
├── save                         # 保存的模型和结果
├── script                       # 运行脚本
├── test
├── trainer
│   ├── trainer_basic.py         # 训练、验证、测试的基本循环
│   ├── trainer_set.py           # 训练配置，包括日志、检查点、剪枝等
│   └── utils.py                 # 辅助工具，如损失函数和回调类
├── utils
└── wandb
```

# 信号处理模块

1. **FFT**：快速傅里叶变换。

   $$X(k) = \sum_{n=0}^{N-1} x(n)e^{-j2\pi kn/N}$$

2. **小波变换**：小波变换。

   $$W(a,b) = \frac{1}{\sqrt{|a|}}\int_{-\infty}^{+\infty} x(t)\psi\left(\frac{t-b}{a}\right)dt$$

3. **希尔伯特变换**：希尔伯特变换。

   $$H(x(t)) = \frac{1}{\pi} \mathrm{P} \int_{-\infty}^{+\infty} \frac{x(\tau)}{t-\tau} d\tau$$

4. **小波滤波**：小波滤波。

   $$y(t) = \sum_{n=0}^{N-1} h(n)x(t-n)$$

# 特征提取模块

1. **MeanFeature**：均值计算。

   $$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$

2. **StdFeature**：标准差计算。

   $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}$$

3. **VarFeature**：方差计算。

   $$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2$$

4. **EntropyFeature**：熵计算。

   $$H(x) = -\sum_{i=1}^{N} p(x_i) \log p(x_i)$$

5. **MaxFeature**：最大值计算。

   $$\max(x) = \max_{i} x_i$$

6. **MinFeature**：最小值计算。

   $$\min(x) = \min_{i} x_i$$

7. **AbsMeanFeature**：绝对值均值计算。

   $$\text{abs\_mean}(x) = \frac{1}{N}\sum_{i=1}^{N} |x_i|$$

8. **KurtosisFeature**：峰度计算。

   $$\text{kurtosis}(x) = \frac{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4}$$

9. **RMSFeature**：均方根值计算。

   $$\text{rms}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$

10. **CrestFactorFeature**：峰值因子计算。

    $$\text{crest\_factor}(x) = \frac{\max_{i} x_i}{\text{rms}(x)}$$

11. **ClearanceFactorFeature**：间隙因子计算。

    $$\text{clearance\_factor}(x) = \frac{\max_{i} x_i}{\text{abs\_mean}(x)}$$

12. **SkewnessFeature**：偏度计算。

    $$\text{skewness}(x) = \frac{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^3}{\sigma^3}$$

13. **ShapeFactorFeature**：形状因子计算。

    $$\text{shape\_factor}(x) = \frac{\text{rms}(x)}{\text{abs\_mean}(x)}$$

14. **CrestFactorDeltaFeature**：峰值因子差分值计算。

    $$\text{crest\_factor\_delta}(x) = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_{i+1} - x_i)^2}}{\text{abs\_mean}(x)}$$

# 逻辑推理模块

TODO


# 模型规约

1. 1D 模型
2. 2D 时频模型
