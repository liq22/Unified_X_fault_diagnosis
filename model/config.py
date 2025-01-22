# from ast import arg
# from os import pread
# from altair import AllSortString
import torch
from torch import nn
from Signal_processing import SignalProcessingBase,\
        SignalProcessingModuleDict,\
        FFTSignalProcessing,\
        HilbertTransform,\
        WaveFilters,\
        Identity
from Signal_processing import *
from Signal_processing_2D import *

from Feature_extract import FeatureExtractionBase,\
        FeatureExtractionModuleDict,\
        MeanFeature,\
        StdFeature,\
        VarFeature,\
        EntropyFeature,\
        MaxFeature,\
        MinFeature,\
        AbsMeanFeature,\
        KurtosisFeature,\
        RMSFeature,\
        CrestFactorFeature ,\
        SkewnessFeature,\
        ClearanceFactorFeature,\
        ShapeFactorFeature
from Logic_inference import LogicInferenceBase,\
        ImplicationOperation,\
        EquivalenceOperation,\
        NegationOperation,\
        WeakConjunctionOperation,\
        WeakDisjunctionOperation,\
        StrongConjunctionOperation,\
        StrongDisjunctionOperation
ALL_SP = {
    'FFT': FFTSignalProcessing,
    'HT': HilbertTransform,
    'WF': WaveFilters,
    'I': Identity,
    'RWF':RickerWaveletFilter,
    'LWF':LaplaceWaveletFilter,
    'CWF':ChirpletWaveletFilter,
    'MWF':MorletWaveletFilter,
    
    'Morlet':Morlet, # 'Morlet':Morlet,
    'Laplace':Laplace,
    'Order1MAFilter':Order1MAFilter,
    'Order2MAFilter':Order2MAFilter,
    'Order1DFFilter':Order1DFFilter,
    'Order2DFFilter':Order2DFFilter,
    'Log':LogOperation,
    'Squ':SquOperation,
    'Sin':SinOperation,
    # 2arity
    'Add':AddOperation,
    'Mul':MulOperation,
    'Div':DivOperation,
    
    'SBCT':SBCT_NOP,
    'GF':GlobalFilterOperator,
    
}
ALL_FE = {
    'Mean': MeanFeature,
    'Std': StdFeature,
    'Var': VarFeature,
    'Entropy': EntropyFeature,
    'Max': MaxFeature,
    'Min': MinFeature,
    'AbsMean': AbsMeanFeature,
    'Kurtosis': KurtosisFeature,
    'RMS': RMSFeature,
    'CrestFactor': CrestFactorFeature,
    'Skewness': SkewnessFeature,
    'ClearanceFactor': ClearanceFactorFeature,
    'ShapeFactor': ShapeFactorFeature,
}

ALL_LI = {
    'imp': ImplicationOperation,
    'equ': EquivalenceOperation,
    'neg': NegationOperation,
    'conj': WeakConjunctionOperation,
    'disj': WeakDisjunctionOperation,
    'sconj': StrongConjunctionOperation,
    'sdisj': StrongDisjunctionOperation,
}
# logic

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


import yaml
from types import SimpleNamespace

config_dir = 'configs/config_basic_2D.yaml'# configs/config_basic.yaml configs/THU_006/config_DEN_gen.yaml
# 读取YAML文件
with open(config_dir, 'r') as f:
    config = yaml.safe_load(f)
args = SimpleNamespace(**config['args'])



signal_processing_modules = []
for layer in config['signal_processing_configs'].values():
    signal_module = OrderedDict()
    for module_name in layer:
        module_class = ALL_SP[module_name]

    module_count = {}  # 用于跟踪每个模块名出现的次数
    
    for module_name in layer:
        module_class = ALL_SP[module_name]
        
        # 检查模块名是否已存在
        if module_name in module_count:
            module_count[module_name] += 1
            new_module_name = f"{module_name}_{module_count[module_name]}"
        else:
            module_count[module_name] = 0
            new_module_name = module_name

        signal_module[new_module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数 ,但是有些模块需要参数
    signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

feature_extractor_modules = OrderedDict()
for feature_name in config['feature_extractor_configs']:
    module_class = ALL_FE[feature_name]
    feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
