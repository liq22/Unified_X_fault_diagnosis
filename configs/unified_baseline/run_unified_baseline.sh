#!/bin/bash

# 统一基线实验运行脚本
# 对比所有方法与基线模型性能

echo "开始统一基线实验..."

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 创建结果目录
RESULT_DIR="/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/results/unified_baseline"
mkdir -p $RESULT_DIR

# 记录开始时间
START_TIME=$(date)
echo "实验开始时间: $START_TIME"

# 定义模型列表
NEW_MODELS=(
    "TSPN"
    "TFON"
    "NNSPN"
    "TKAN"
    "OperatorAttention"
    "MoE"
    "Fusion1D2D"
)

BASELINE_MODELS=(
    "Resnet"
    "SincNet"
    "WKN"
    "MCN"
    "TFN"
)

# 运行新方法实验
echo "开始运行7个新方法实验..."
for model in "${NEW_MODELS[@]}"; do
    echo "正在运行: $model"

    CONFIG_FILE="/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/configs/unified_baseline/config_${model}.yaml"

    if [ -f "$CONFIG_FILE" ]; then
        echo "配置文件存在: $CONFIG_FILE"

        # 运行实验
        if [ "$model" == "Fusion1D2D" ]; then
            python main_fusion.py --config_file $CONFIG_FILE > $RESULT_DIR/${model}_log.txt 2>&1
        else
            python main.py --config_file $CONFIG_FILE > $RESULT_DIR/${model}_log.txt 2>&1
        fi

        # 检查是否运行成功
        if [ $? -eq 0 ]; then
            echo "$model 实验完成"
        else
            echo "$model 实验失败，请检查日志: $RESULT_DIR/${model}_log.txt"
        fi
    else
        echo "配置文件不存在: $CONFIG_FILE"
    fi

    # 短暂休息，避免GPU过热
    sleep 5
done

# 运行基线模型实验
echo "开始运行5个基线模型实验..."
for model in "${BASELINE_MODELS[@]}"; do
    echo "正在运行: $model"

    CONFIG_FILE="/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/configs/unified_baseline/config_${model}.yaml"

    if [ -f "$CONFIG_FILE" ]; then
        echo "配置文件存在: $CONFIG_FILE"

        # 运行实验
        python main_com.py --config_dir $CONFIG_FILE > $RESULT_DIR/${model}_log.txt 2>&1

        # 检查是否运行成功
        if [ $? -eq 0 ]; then
            echo "$model 实验完成"
        else
            echo "$model 实验失败，请检查日志: $RESULT_DIR/${model}_log.txt"
        fi
    else
        echo "配置文件不存在: $CONFIG_FILE"
    fi

    # 短暂休息
    sleep 5
done

# 记录结束时间
END_TIME=$(date)
echo "实验结束时间: $END_TIME"

# 生成实验总结
echo "统一基线实验完成！"
echo "结果保存在: $RESULT_DIR"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"

# 统计结果文件
echo "生成的实验文件："
ls -la $RESULT_DIR/