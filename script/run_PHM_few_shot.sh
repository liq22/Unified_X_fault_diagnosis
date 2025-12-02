#!/bin/bash
# PHM-Vibench少样本学习实验脚本
# 测试模型在少样本条件下的性能

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建日志目录
LOG_DIR="logs/PHM_FewShot_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 基础配置
BASE_CONFIG="configs/PHM_Vibench/config_TSPN_test.yaml"

# 少样本设置
K_SHOTS=(1 5 10 20 50)
MODEL="TSPN"

echo "=========================================="
echo "PHM-Vibench少样本学习实验"
echo "时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 记录实验开始
echo "PHM-Vibench少样本学习实验开始" > $LOG_DIR/experiment.log
echo "开始时间: $(date)" >> $LOG_DIR/experiment.log

# 对每个k-shot设置进行实验
for K_SHOT in "${K_SHOTS[@]}"; do
    echo "----------------------------------------"
    echo "K-shot 设置: $K_SHOT"
    echo "开始时间: $(date)"
    echo "----------------------------------------"

    # 创建临时配置文件
    TEMP_CONFIG="$LOG_DIR/few_shot_${K_SHOT}shot.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"

    # 修改配置文件以设置少样本学习
    python << EOF
import yaml

# 读取配置
with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# 修改为少样本学习设置
config['args']['k_shot'] = $K_SHOT
config['vbench_config']['sampling_config']['target_per_class'] = $K_SHOT
config['vbench_config']['sampling_config']['method'] = 'balanced'  # 确保平衡采样
config['vbench_config']['sampling_config']['min_samples_per_id'] = 1
config['vbench_config']['sampling_config']['ensure_unique_ids'] = True

# 减少训练轮数（少样本学习通常更快收敛）
config['args']['num_epochs'] = 50
config['args']['patience'] = 10

# 降低学习率（少样本学习需要更保守的学习策略）
config['args']['learning_rate'] = 0.0005

# 增加正则化以防止过拟合
config['args']['weight_decay'] = 0.001
config['args']['l1_norm'] = 0.001

# 修改实验标识
config['args']['experiment_type'] = 'PHM_few_shot'
config['args']['description'] = f"少样本学习实验: {K_SHOT}-shot"
config['args']['wandb_group'] = f'FewShot_{K_SHOT}shot'

# 保存修改后的配置
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"少样本学习配置已保存: $TEMP_CONFIG")
EOF

    # 创建k-shot专用日志文件
    KSHOT_LOG="$LOG_DIR/few_shot_${K_SHOT}shot_$(date +%H%M%S).log"

    # 运行少样本学习实验
    echo "开始 ${K_SHOT}-shot 学习实验" | tee -a $KSHOT_LOG

    source activate LQ_signal && {
        python main.py \
            --config_file "$TEMP_CONFIG" \
            2>&1 | tee -a $KSHOT_LOG

        # 检查实验结果
        if [ $? -eq 0 ]; then
            echo "✅ ${K_SHOT}-shot 学习实验完成" | tee -a $KSHOT_LOG
            echo "FewShot_${K_SHOT}shot: SUCCESS" >> $LOG_DIR/results.txt
        else
            echo "❌ ${K_SHOT}-shot 学习实验失败" | tee -a $KSHOT_LOG
            echo "FewShot_${K_SHOT}shot: FAILED" >> $LOG_DIR/results.txt
        fi
    }

    echo "${K_SHOT}-shot 实验完成时间: $(date)" >> $KSHOT_LOG
    echo ""

    # 清理临时配置文件
    rm -f "$TEMP_CONFIG"

    # 短暂休息
    sleep 5
done

# 运行多模型的少样本对比实验（使用k=10）
echo "----------------------------------------"
echo "多模型少样本对比实验 (K=10)"
echo "开始时间: $(date)"
echo "----------------------------------------"

MULTI_MODEL_LOG="$LOG_DIR/multi_model_few_shot_$(date +%H%M%S).log"
MODELS=("TSPN" "TKAN" "NNSPN" "OperatorAttention")

for MODEL in "${MODELS[@]}"; do
    echo "运行 $MODEL 的10-shot实验" | tee -a $MULTI_MODEL_LOG

    # 找到对应的配置文件
    if [ "$MODEL" = "TSPN" ]; then
        CONFIG_FILE="configs/PHM_Vibench/config_TSPN_test.yaml"
    else
        CONFIG_FILE="configs/PHM_Vibench/config_${MODEL}.yaml"
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "警告: 配置文件 $CONFIG_FILE 不存在，跳过 $MODEL"
        continue
    fi

    # 创建临时配置文件
    TEMP_CONFIG="$LOG_DIR/${MODEL}_few_shot_10shot.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"

    # 修改配置
    python << EOF
import yaml

with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

config['args']['k_shot'] = 10
config['vbench_config']['sampling_config']['target_per_class'] = 10
config['vbench_config']['sampling_config']['method'] = 'balanced'
config['args']['num_epochs'] = 50
config['args']['patience'] = 10
config['args']['learning_rate'] = 0.0005
config['args']['experiment_type'] = 'PHM_few_shot'
config['args']['description'] = f"$MODEL 少样本学习: 10-shot"
config['args']['wandb_group'] = 'FewShot_Comparison'

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF

    source activate LQ_signal && {
        python main.py \
            --config_file "$TEMP_CONFIG" \
            2>&1 | tee -a $MULTI_MODEL_LOG

        if [ $? -eq 0 ]; then
            echo "✅ $MODEL 10-shot实验完成" | tee -a $MULTI_MODEL_LOG
            echo "${MODEL}_10shot: SUCCESS" >> $LOG_DIR/multi_model_results.txt
        else
            echo "❌ $MODEL 10-shot实验失败" | tee -a $MULTI_MODEL_LOG
            echo "${MODEL}_10shot: FAILED" >> $LOG_DIR/multi_model_results.txt
        fi
    }

    rm -f "$TEMP_CONFIG"
    sleep 3
done

# 记录实验结束
echo "----------------------------------------"
echo "少样本学习实验完成"
echo "结束时间: $(date)"
echo "----------------------------------------"

echo "少样本学习实验结束时间: $(date)" >> $LOG_DIR/experiment.log

# 生成结果摘要
echo "========================================" >> $LOG_DIR/summary.txt
echo "PHM-Vibench少样本学习实验结果摘要" >> $LOG_DIR/summary.txt
echo "完成时间: $(date)" >> $LOG_DIR/summary.txt
echo "========================================" >> $LOG_DIR/summary.txt

echo "实验设置:" >> $LOG_DIR/summary.txt
echo "- K-shot设置: ${K_SHOTS[*]}" >> $LOG_DIR/summary.txt
echo "- 主模型: $MODEL" >> $LOG_DIR/summary.txt
echo "- 对比模型: ${MODELS[*]}" >> $LOG_DIR/summary.txt
echo "" >> $LOG_DIR/summary.txt

if [ -f "$LOG_DIR/results.txt" ]; then
    echo "TSPN K-shot实验结果:" >> $LOG_DIR/summary.txt
    cat "$LOG_DIR/results.txt" >> $LOG_DIR/summary.txt
    echo "" >> $LOG_DIR/summary.txt
fi

if [ -f "$LOG_DIR/multi_model_results.txt" ]; then
    echo "多模型10-shot对比结果:" >> $LOG_DIR/summary.txt
    cat "$LOG_DIR/multi_model_results.txt" >> $LOG_DIR/summary.txt
    echo "" >> $LOG_DIR/summary.txt
fi

# 统计结果
if [ -f "$LOG_DIR/results.txt" ]; then
    SUCCESS_COUNT=$(grep -c "SUCCESS" $LOG_DIR/results.txt)
    FAILED_COUNT=$(grep -c "FAILED" $LOG_DIR/results.txt)
    TOTAL_COUNT=$((SUCCESS_COUNT + FAILED_COUNT))

    echo "TSPN K-shot统计:" >> $LOG_DIR/summary.txt
    echo "总计: $TOTAL_COUNT, 成功: $SUCCESS_COUNT, 失败: $FAILED_COUNT" >> $LOG_DIR/summary.txt
    if [ $TOTAL_COUNT -gt 0 ]; then
        echo "成功率: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%" >> $LOG_DIR/summary.txt
    fi
    echo "" >> $LOG_DIR/summary.txt
fi

if [ -f "$LOG_DIR/multi_model_results.txt" ]; then
    SUCCESS_COUNT=$(grep -c "SUCCESS" $LOG_DIR/multi_model_results.txt)
    FAILED_COUNT=$(grep -c "FAILED" $LOG_DIR/multi_model_results.txt)
    TOTAL_COUNT=$((SUCCESS_COUNT + FAILED_COUNT))

    echo "多模型对比统计:" >> $LOG_DIR/summary.txt
    echo "总计: $TOTAL_COUNT, 成功: $SUCCESS_COUNT, 失败: $FAILED_COUNT" >> $LOG_DIR/summary.txt
    if [ $TOTAL_COUNT -gt 0 ]; then
        echo "成功率: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%" >> $LOG_DIR/summary.txt
    fi
fi

echo "实验日志保存在: $LOG_DIR"
echo "结果摘要: $LOG_DIR/summary.txt"

# 显示结果摘要
if [ -f "$LOG_DIR/summary.txt" ]; then
    cat $LOG_DIR/summary.txt
fi

echo "🎉 PHM-Vibench少样本学习实验脚本执行完成！"