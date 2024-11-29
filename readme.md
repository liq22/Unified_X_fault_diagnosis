# task 


DATASET_TASK_CLASS = {
    'THU_006_basic': THU_006or018_basic,
    'THU_018_basic': THU_006or018_basic,
    'THU_018_few_shot': THU_006or018_few_shot,
    'THU_006_few_shot': THU_006or018_few_shot,
    'THU_006_generalization': THU_006_generalization
    
}

# file organization

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
│   ├── __pycache__/
│   ├── config.py  # 项目中的config 其中有 parse network
│   ├── config_3_11_01hz.yaml
│   ├── config_3_11_15hz.yaml
│   ├── config_basic.yaml
│   ├── config_com.yaml # 对比方法
│   ├── config_test.ipynb
│   ├── DIRG_020/
│   ├── THU_006/ 
│   ├── ...根据数据集的不同，有不同的配置文件
│   └── THU_006ablation/ 
├── data
│   ├── __pycache__/
│   ├── data_provider.py
│   ├── datasets.py
│   └── utils.py
├── main_ablation_exp.py # 消融实验，网格学习率
├── main_com.py # 对比方法
├── main_com_kshotexp.py # kshot对比方法
├── main_kshotexp.py # kshot
├── main.py # 主程序
├── model # 保存着自己的模型们
│   ├── __pycache__/
│   ├── dict.md
│   ├── Feature_extract.py
│   ├── Logic_inference.py
│   ├── parse_network.py  # 网络解析和可视化，可以用来查看网络结构，目前不太好用
│   ├── past/
│   ├── Signal_processing.py
│   ├── DEN.py  # 更换了一些算子
│   ├── TSPN.py  # TIFN 工作 multi-source
│   ├── NNSPN.py  # EOAN 工作 加了attention
│   └── utils.py
├── model_collection
│   ├── __pycache__/
│   ├── EELM.py # 王冬工作
│   ├── F_EQL.py # EQL 工作
│   ├── MCN/
│   ├── MCN.py # MCN 工作 
│   ├── MWA_CNN.py
│   ├── Resnet.py
│   ├── Sincnet.py
│   ├── test_models.py
│   ├── TFN/
│   ├── TFN.py
│   └── WKN.py
├── plot # 根据不同工作划分文件夹
├── post # 后处理后分析
├── post_analysis.ipynb
├── readme.md
├── save # 保存的程序，原先是以模型命名，现在以任务开头命名，最后根据需要的工作整理到对应文件夹中
├── script # 程序运行脚本
├── test
├── trainer
│   ├── __pycache__/
│   ├── trainer_basic.py # 训练验证测试的基本loop 方法
│   ├── trainer_set.py # 训练配置，包括 log，checkpoint，prune，earlystop，scheduler
│   └── utils.py # 辅助工具，一些loss，callback类等
├── Transparent_information_fusion.code-workspace
├── utils
└── wandb
```

# Signal_processing
1. FFT：快速傅里叶变换。公式为：$X(k) = \sum_{n=0}^{N-1} x(n)e^{-j2\pi kn/N}$。
2. wavelet_transform：小波变换。公式为：$W(a,b) = \frac{1}{\sqrt{|a|}}\int_{-\infty}^{+\infty} x(t)\psi(\frac{t-b}{a})dt$。
3. Hilbert_transform：希尔伯特变换。公式为：$H(x(t)) = \frac{1}{\pi}P\int_{-\infty}^{+\infty} \frac{x(\tau)}{t-\tau}d\tau$。
4. wavefilter：小波滤波。公式为：$y(t) = \sum_{n=0}^{N-1} h(n)x(t-n)$。

# Feature_extractor
1. MeanFeature：计算输入x的均值。公式为：$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$。

2. StdFeature：计算输入x的标准差。公式为：$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}$。

3. VarFeature：计算输入x的方差。公式为：$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2$。

4. EntropyFeature：计算输入x的熵。公式为：$H(x) = -\sum_{i=1}^{N} p(x_i) \log p(x_i)$。

5. MaxFeature：计算输入x的最大值。公式为：$max(x) = \max_{i} x_i$。

6. MinFeature：计算输入x的最小值。公式为：$min(x) = \min_{i} x_i$。

7. AbsMeanFeature：计算输入x的绝对值的均值。公式为：$abs_mean(x) = \frac{1}{N}\sum_{i=1}^{N} |x_i|$。

8. KurtosisFeature：计算输入x的峰度。公式为：$kurtosis(x) = \frac{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4}$。

9. RMSFeature：计算输入x的均方根值。公式为：$rms(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$。

10. CrestFactorFeature：计算输入x的峰值因子。公式为：$crest_factor(x) = \frac{\max_{i} x_i}{rms(x)}$。

11. ClearanceFactorFeature：计算输入x的间隙因子。公式为：$clearance_factor(x) = \frac{\max_{i} x_i}{abs_mean(x)}$。

12. SkewnessFeature：计算输入x的偏度。公式为：$skewness(x) = \frac{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^3}{\sigma^3}$。

13. ShapeFactorFeature：计算输入x的形状因子。公式为：$shape_factor(x) = \frac{rms(x)}{abs_mean(x)}$。

14. CrestFactorDeltaFeature：计算输入x的峰值因子的差分值。公式为：$crest_factor_delta(x) = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_{i+1} - x_i)^2}}{abs_mean(x)}$。

# Logic_inference

# note

没有用wandb记录 以及sweep 以及自动调参


git filter-branch --force --index-filter \
  "git rm --cached --post_analysis.ipynb" \
  --prune-empty --tag-name-filter cat -- --all
  
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'post_analysis.ipynb'" \
  --prune-empty --tag-name-filter cat -- --all
