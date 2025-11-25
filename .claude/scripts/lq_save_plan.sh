#!/usr/bin/env bash
set -euo pipefail

# LQ Save Plan Script -三层架构实现
# 用法: lq_save_plan.sh --slug <slug> [--subdir <codex|glm>] [--paper-root <path>] [--apply] [--title <title>] [--content <content>]

# 默认参数
slug=""
subdir="codex"
paper_root=""
apply=0
title=""
content=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) slug="$2"; shift 2 ;;
    --subdir) subdir="$2"; shift 2 ;;
    --paper-root) paper_root="$2"; shift 2 ;;
    --apply) apply=1; shift ;;
    --title) title="$2"; shift 2 ;;
    --content) content="$2"; shift 2 ;;
    --help)
      echo "用法: lq_save_plan.sh --slug <slug> [--subdir <codex|glm>] [--paper-root <path>] [--apply] [--title <title>] [--content <content>]"
      echo ""
      echo "参数说明:"
      echo "  --slug      必填，文件slug（将转换为小写和下划线）"
      echo "  --subdir    可选，子目录（codex或glm，默认codex）"
      echo "  --paper-root 可选，paper项目根目录（默认自动检测）"
      echo "  --apply     可选，实际执行操作（默认dry-run）"
      echo "  --title     可选，文档标题（默认从slug生成）"
      echo "  --content   可选，计划内容（默认使用模板）"
      echo ""
      echo "示例:"
      echo "  lq_save_plan.sh --slug isfm_prompt_unit_test --apply"
      echo "  lq_save_plan.sh --slug experiment_validation --subdir glm --title '实验验证计划'"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 检查必填参数
if [[ -z "$slug" ]]; then
  echo "❌ 错误: --slug 参数必填"
  exit 1
fi

# 验证subdir参数
if [[ "$subdir" != "codex" && "$subdir" != "glm" ]]; then
  echo "❌ 错误: --subdir 必须是 'codex' 或 'glm'"
  exit 1
fi

# 自动检测paper根目录
if [[ -z "$paper_root" ]]; then
  current_dir=$(pwd)
  search_dir="$current_dir"
  while [[ "$search_dir" != "/" ]]; do
    if [[ -d "$search_dir/paper/2025-10_foundation_model_0_metric" ]]; then
      paper_root="$search_dir/paper/2025-10_foundation_model_0_metric"
      break
    fi
    search_dir="$(dirname "$search_dir")"
  done
fi

# 检查paper根目录
if [[ -z "$paper_root" || ! -d "$paper_root" ]]; then
  echo "❌ 错误: 未找到paper项目根目录 paper/2025-10_foundation_model_0_metric"
  echo "请使用 --paper-root 参数指定路径"
  exit 1
fi

# 生成文件路径
today_dir=$(date +%m_%d)
today_date=$(date +%Y-%m-%d)
docs_dir="$paper_root/docs/$today_dir/$subdir"
clean_slug=$(echo "$slug" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g' | sed 's/__/_/g')
file_name="${clean_slug}_${today_date}.md"
target_file="$docs_dir/$file_name"

# 创建日志目录
log_dir="$paper_root/.claude/logs"
mkdir -p "$log_dir"
log_file="$log_dir/lq_save_plan_${today_date}.log"

# 日志函数
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

# 检查文件是否已存在
if [[ -f "$target_file" ]]; then
  echo "⚠️  警告: 目标文件已存在: $target_file"
  if [[ $apply -eq 1 ]]; then
    # 生成新的文件名
    counter=2
    while [[ -f "$docs_dir/${clean_slug}_${today_date}_v${counter}.md" ]]; do
      ((counter++))
    done
    file_name="${clean_slug}_${today_date}_v${counter}.md"
    target_file="$docs_dir/$file_name"
    echo "📝 将创建新文件: $target_file"
  else
    echo "💡 dry-run模式: 将显示冲突信息"
  fi
fi

# 生成标题（如果未提供）
if [[ -z "$title" ]]; then
  title=$(echo "$clean_slug" | sed 's/_/ /g' | sed 's/\b\w/\u&/g')
fi

# 生成默认内容（如果未提供）
if [[ -z "$content" ]]; then
  content="# $title

- Created at: $today_date
- Project: $(basename "$paper_root")
- Source: Claude Code (lq_save-plan)
- Status: TODO
- Category: plan

## Background

（1-3行，说明这个计划的背景、动机）

## Objectives

- [ ] 目标 1
- [ ] 目标 2
- [ ] 目标 3

## Plan

1. 步骤 1
2. 步骤 2
3. 步骤 3

## Notes

- 依赖：
- 风险：
- 相关 PR / 实验："
fi

# 显示操作预览
echo "📋 LQ Save Plan 操作预览"
echo "   Paper项目根目录: $paper_root"
echo "   目标目录: $docs_dir"
echo "   目标文件: $file_name"
echo "   Subdir: $subdir"
echo "   Title: $title"
echo "   Apply模式: $([[ $apply -eq 1 ]] && echo '是' || echo '否（dry-run）')"
echo ""

# 如果是dry-run，只显示预览
if [[ $apply -eq 0 ]]; then
  echo "🔍 Dry-run模式 - 将要创建的内容预览："
  echo "======================================"
  echo "$content"
  echo "======================================"
  echo ""
  echo "💡 要执行实际操作，请添加 --apply 参数"
  log_message "DRY-RUN: Save plan preview - $file_name"
  exit 0
fi

# 执行实际操作
mkdir -p "$docs_dir"

# 写入文件
echo "$content" > "$target_file"

echo "✅ 成功创建计划文件: $target_file"
echo "   文件大小: $(wc -c < "$target_file") 字节"
echo "   包含章节: $(grep -c '^##' "$target_file") 个"

# 记录日志
log_message "SUCCESS: Save plan - $file_name ($(wc -c < "$target_file") bytes)"

# 显示相对路径（便于复制）
relative_path=$(echo "$target_file" | sed "s|$paper_root/||")
echo "📂 相对路径: $relative_path"