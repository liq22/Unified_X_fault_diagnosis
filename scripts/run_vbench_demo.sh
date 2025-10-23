#!/bin/bash
# Vbench数据集演示脚本
# 快速运行故障诊断实验

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置文件
CONFIG_DIR="configs/vbench"
EXAMPLE_CONFIG="configs/vbench/config_vbench_diagnosis.yaml"
CWRU_CONFIG="configs/vbench/RM_001_CWRU/config_TSPN.yaml"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== Vbench数据集演示 =====${NC}"
echo

# 检查数据路径
if [ ! -d "/home/user/data/PHMbenchdata/PHM-Vibench" ]; then
    echo -e "${RED}错误: Vbench数据目录不存在${NC}"
    echo "请确保数据已下载到: /home/user/data/PHMbenchdata/PHM-Vibench"
    exit 1
fi

# 检查配置文件
if [ ! -f "$EXAMPLE_CONFIG" ]; then
    echo -e "${RED}错误: 配置文件不存在${NC}"
    echo "请先运行文档中的实施脚本"
    exit 1
fi

# 菜单
echo -e "${YELLOW}请选择运行模式:${NC}"
echo "1) 基础演示（使用默认配置）"
echo "2) CWRU数据集演示"
echo "3) 快速测试（小数据集）"
echo "4) 自定义配置"
echo

read -p "请输入选择 [1-4]: " choice

case $choice in
    1)
        CONFIG_FILE="$EXAMPLE_CONFIG"
        EPOCHS=50
        echo -e "${GREEN}运行基础演示...${NC}"
        ;;
    2)
        CONFIG_FILE="$CWRU_CONFIG"
        EPOCHS=100
        echo -e "${GREEN}运行CWRU演示...${NC}"
        ;;
    3)
        CONFIG_FILE="$EXAMPLE_CONFIG"
        EPOCHS=10
        QUICK="--quick"
        echo -e "${GREEN}运行快速测试...${NC}"
        ;;
    4)
        read -p "请输入配置文件路径: " custom_config
        if [ -f "$custom_config" ]; then
            CONFIG_FILE="$custom_config"
            EPOCHS=100
            echo -e "${GREEN}使用自定义配置: $CONFIG_FILE${NC}"
        else
            echo -e "${RED}配置文件不存在: $custom_config${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo
echo -e "${YELLOW}配置文件: $CONFIG_FILE${NC}"
echo -e "${YELLOW}训练轮数: $EPOCHS${NC}"
echo

# 创建日志目录
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 运行训练
echo -e "${GREEN}开始训练...${NC}"
echo -e "${YELLOW}日志保存在: $LOG_DIR${NC}"
echo

python main.py \
    --config_file "$CONFIG_FILE" \
    ${QUICK} \
    2>&1 | tee "$LOG_DIR/training.log"

# 检查结果
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}训练完成！${NC}"
    echo -e "${YELLOW}查看结果:${NC}"
    echo "  1. WandB: https://wandb.ai/"
    echo "  2. 日志: $LOG_DIR/training.log"
    echo "  3. 模型保存在: save/ 目录"

    # 显示最后几行日志
    echo
    echo -e "${GREEN}=== 训练摘要 ===${NC}"
    tail -20 "$LOG_DIR/training.log"
else
    echo -e "${RED}训练失败！${NC}"
    echo "请检查日志: $LOG_DIR/training.log"
fi

echo