#!/bin/bash
# PHM-Vibench统一基线实验脚本
# 运行所有模型在PHM数据集上的对比实验

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建日志目录
LOG_DIR="logs/PHM_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# PHM配置文件目录
PHM_CONFIG_DIR="configs/PHM_Vibench"

# 模型列表
MODELS=(
    "TSPN"
    "TKAN"
    "NNSPN"
    "OperatorAttention"
    "FuzzyLogic"
)

# 数据集配置列表
DATASET_CONFIGS=(
    "config_TSPN_test.yaml"
    "config_TKAN.yaml"
    "config_NNSPN.yaml"
    "config_OperatorAttention.yaml"
    "config_FuzzyLogic.yaml"
)

# 打印实验信息
echo "=========================================="
echo "PHM-Vibench统一基线实验"
echo "时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 记录实验开始
echo "PHM-Vibench统一基线实验开始" > $LOG_DIR/experiment.log
echo "开始时间: $(date)" >> $LOG_DIR/experiment.log
echo "GPU配置: $CUDA_VISIBLE_DEVICES" >> $LOG_DIR/experiment.log

# 运行每个模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CONFIG="${DATASET_CONFIGS[$i]}"
    CONFIG_PATH="$PHM_CONFIG_DIR/$CONFIG"

    echo "----------------------------------------"
    echo "运行模型: $MODEL"
    echo "配置文件: $CONFIG"
    echo "开始时间: $(date)"
    echo "----------------------------------------"

    # 检查配置文件是否存在
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "错误: 配置文件 $CONFIG_PATH 不存在"
        continue
    fi

    # 创建模型专用日志文件
    MODEL_LOG="$LOG_DIR/${MODEL}_$(date +%H%M%S).log"

    # 运行训练
    echo "开始训练 $MODEL..." | tee -a $MODEL_LOG

    # 使用source激活环境并运行训练
    source activate LQ_signal && {
        # 运行主训练脚本
        python main.py \
            --config_file "$CONFIG_PATH" \
            2>&1 | tee -a $MODEL_LOG

        # 检查训练结果
        if [ $? -eq 0 ]; then
            echo "✅ $MODEL 训练完成" | tee -a $MODEL_LOG
            echo "$MODEL: SUCCESS" >> $LOG_DIR/results.txt
        else
            echo "❌ $MODEL 训练失败" | tee -a $MODEL_LOG
            echo "$MODEL: FAILED" >> $LOG_DIR/results.txt
        fi
    }

    echo "完成时间: $(date)" >> $MODEL_LOG
    echo ""

    # 短暂休息，避免GPU过热
    sleep 5
done

# 运行对比模型实验
echo "----------------------------------------"
echo "运行对比模型实验"
echo "开始时间: $(date)"
echo "----------------------------------------"

COM_CONFIG_PATH="$PHM_CONFIG_DIR/config_com.yaml"
COM_LOG="$LOG_DIR/baseline_models_$(date +%H%M%S).log"

if [ -f "$COM_CONFIG_PATH" ]; then
    # 对比模型列表
    BASELINE_MODELS=(
        "Resnet"
        "SincNet"
        "WKN"
        "MCN"
        "TFN"
    )

    for MODEL in "${BASELINE_MODELS[@]}"; do
        echo "训练对比模型: $MODEL" | tee -a $COM_LOG

        source activate LQ_signal && {
            python main_com.py \
                --config_file "$COM_CONFIG_PATH" \
                --model "$MODEL" \
                2>&1 | tee -a $COM_LOG

            if [ $? -eq 0 ]; then
                echo "✅ 对比模型 $MODEL 训练完成" | tee -a $COM_LOG
                echo "Baseline_$MODEL: SUCCESS" >> $LOG_DIR/results.txt
            else
                echo "❌ 对比模型 $MODEL 训练失败" | tee -a $COM_LOG
                echo "Baseline_$MODEL: FAILED" >> $LOG_DIR/results.txt
            fi
        }

        sleep 3
    done
else
    echo "警告: 对比模型配置文件 $COM_CONFIG_PATH 不存在"
fi

# 记录实验结束
echo "----------------------------------------"
echo "实验完成"
echo "结束时间: $(date)"
echo "----------------------------------------"

echo "实验结束时间: $(date)" >> $LOG_DIR/experiment.log

# 生成结果摘要
echo "========================================" >> $LOG_DIR/summary.txt
echo "PHM-Vibench统一基线实验结果摘要" >> $LOG_DIR/summary.txt
echo "完成时间: $(date)" >> $LOG_DIR/summary.txt
echo "========================================" >> $LOG_DIR/summary.txt

if [ -f "$LOG_DIR/results.txt" ]; then
    echo "详细结果:" >> $LOG_DIR/summary.txt
    cat "$LOG_DIR/results.txt" >> $LOG_DIR/summary.txt

    # 统计成功和失败的数量
    SUCCESS_COUNT=$(grep -c "SUCCESS" $LOG_DIR/results.txt)
    FAILED_COUNT=$(grep -c "FAILED" $LOG_DIR/results.txt)

    echo "" >> $LOG_DIR/summary.txt
    echo "统计:" >> $LOG_DIR/summary.txt
    echo "成功: $SUCCESS_COUNT" >> $LOG_DIR/summary.txt
    echo "失败: $FAILED_COUNT" >> $LOG_DIR/summary.txt
fi

echo "实验日志保存在: $LOG_DIR"
echo "结果摘要: $LOG_DIR/summary.txt"

# 显示结果摘要
if [ -f "$LOG_DIR/summary.txt" ]; then
    cat $LOG_DIR/summary.txt
fi

echo "🎉 PHM-Vibench统一基线实验脚本执行完成！"