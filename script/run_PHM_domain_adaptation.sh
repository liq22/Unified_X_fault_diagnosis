#!/bin/bash
# PHM-Vibench域自适应实验脚本
# 使用留一法进行跨数据集泛化实验

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建日志目录
LOG_DIR="logs/PHM_Domain_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 数据集ID到名称的映射
declare -A DATASET_NAMES=(
    [1]="CWRU"
    [2]="XJTU"
    [3]="FEMTO"
    [6]="THU"
    [7]="MFPT"
    [8]="UNSW"
)

# 数据集ID列表
DATASET_IDS=(1 2 3 6 7 8)

# 基础配置模板
BASE_CONFIG="configs/PHM_Vibench/config_TSPN_test.yaml"

echo "=========================================="
echo "PHM-Vibench域自适应实验 (留一法)"
echo "时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 记录实验开始
echo "PHM-Vibench域自适应实验开始" > $LOG_DIR/experiment.log
echo "开始时间: $(date)" >> $LOG_DIR/experiment.log

# 对每个数据集作为目标域进行实验
for TARGET_DATASET_ID in "${DATASET_IDS[@]}"; do
    TARGET_DATASET_NAME="${DATASET_NAMES[$TARGET_DATASET_ID]}"

    echo "----------------------------------------"
    echo "目标域: $TARGET_DATASET_NAME (ID: $TARGET_DATASET_ID)"
    echo "开始时间: $(date)"
    echo "----------------------------------------"

    # 创建临时配置文件
    TEMP_CONFIG="$LOG_DIR/domain_config_${TARGET_DATASET_ID}.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"

    # 修改配置文件以启用域自适应
    python << EOF
import yaml

# 读取配置
with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# 修改为域自适应设置
config['vbench_config']['domain_config']['leave_one_domain_out'] = True
config['vbench_config']['domain_config']['target_domain'] = $TARGET_DATASET_ID

# 使用更多数据集进行训练
config['vbench_config']['dataset_ids'] = [1, 2, 3, 6, 7, 8]

# 减少训练样本以加快速度
config['vbench_config']['sampling_config']['target_per_class'] = 100

# 修改实验标识
config['args']['experiment_type'] = 'PHM_domain_adaptation'
config['args']['description'] = f"域自适应实验: 目标域 {config['vbench_config']['dataset_ids']} -> {target_name}"
config['args']['wandb_group'] = f'Domain_Adapt_{target_name}'

# 保存修改后的配置
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"域自适应配置已保存: $TEMP_CONFIG")
EOF

    # 创建目标域专用日志文件
    DOMAIN_LOG="$LOG_DIR/domain_${TARGET_DATASET_NAME}_$(date +%H%M%S).log"

    # 运行域自适应实验
    echo "开始域自适应实验: $TARGET_DATASET_NAME 作为目标域" | tee -a $DOMAIN_LOG

    source activate LQ_signal && {
        python main.py \
            --config_file "$TEMP_CONFIG" \
            2>&1 | tee -a $DOMAIN_LOG

        # 检查实验结果
        if [ $? -eq 0 ]; then
            echo "✅ 域自适应实验 $TARGET_DATASET_NAME 完成" | tee -a $DOMAIN_LOG
            echo "Domain_Adapt_${TARGET_DATASET_NAME}: SUCCESS" >> $LOG_DIR/results.txt
        else
            echo "❌ 域自适应实验 $TARGET_DATASET_NAME 失败" | tee -a $DOMAIN_LOG
            echo "Domain_Adapt_${TARGET_DATASET_NAME}: FAILED" >> $LOG_DIR/results.txt
        fi
    }

    echo "域 $TARGET_DATASET_NAME 实验完成时间: $(date)" >> $DOMAIN_LOG
    echo ""

    # 清理临时配置文件
    rm -f "$TEMP_CONFIG"

    # 短暂休息
    sleep 10
done

# 记录实验结束
echo "----------------------------------------"
echo "域自适应实验完成"
echo "结束时间: $(date)"
echo "----------------------------------------"

echo "域自适应实验结束时间: $(date)" >> $LOG_DIR/experiment.log

# 生成结果摘要
echo "========================================" >> $LOG_DIR/summary.txt
echo "PHM-Vibench域自适应实验结果摘要" >> $LOG_DIR/summary.txt
echo "完成时间: $(date)" >> $LOG_DIR/summary.txt
echo "========================================" >> $LOG_DIR/summary.txt

echo "域自适应实验设置:" >> $LOG_DIR/summary.txt
echo "- 留一法: 使用其他数据集训练，目标数据集测试" >> $LOG_DIR/summary.txt
echo "- 数据集: CWRU(1), XJTU(2), FEMTO(3), THU(6), MFPT(7), UNSW(8)" >> $LOG_DIR/summary.txt
echo "" >> $LOG_DIR/summary.txt

if [ -f "$LOG_DIR/results.txt" ]; then
    echo "详细结果:" >> $LOG_DIR/summary.txt
    cat "$LOG_DIR/results.txt" >> $LOG_DIR/summary.txt

    # 统计成功和失败的数量
    SUCCESS_COUNT=$(grep -c "SUCCESS" $LOG_DIR/results.txt)
    FAILED_COUNT=$(grep -c "FAILED" $LOG_DIR/results.txt)
    TOTAL_COUNT=$((SUCCESS_COUNT + FAILED_COUNT))

    echo "" >> $LOG_DIR/summary.txt
    echo "统计:" >> $LOG_DIR/summary.txt
    echo "总计: $TOTAL_COUNT" >> $LOG_DIR/summary.txt
    echo "成功: $SUCCESS_COUNT" >> $LOG_DIR/summary.txt
    echo "失败: $FAILED_COUNT" >> $LOG_DIR/summary.txt
    echo "成功率: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%" >> $LOG_DIR/summary.txt
fi

echo "实验日志保存在: $LOG_DIR"
echo "结果摘要: $LOG_DIR/summary.txt"

# 显示结果摘要
if [ -f "$LOG_DIR/summary.txt" ]; then
    cat $LOG_DIR/summary.txt
fi

echo "🎉 PHM-Vibench域自适应实验脚本执行完成！"