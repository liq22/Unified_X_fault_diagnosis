#!/usr/bin/env bash
set -euo pipefail

# LQ Log Manager Script - 日志管理工具
# 用法: lq_log_manager.sh [--cleanup] [--paper-root <path>] [--days <N>]

# 默认参数
cleanup=0
paper_root=""
days=30

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanup) cleanup=1; shift ;;
    --paper-root) paper_root="$2"; shift 2 ;;
    --days) days="$2"; shift 2 ;;
    --help)
      echo "用法: lq_log_manager.sh [--cleanup] [--paper-root <path>] [--days <N>]"
      echo ""
      echo "参数说明:"
      echo "  --cleanup    清理过期日志文件"
      echo "  --paper-root paper项目根目录（默认自动检测）"
      echo "  --days       保留日志天数（默认30天）"
      echo ""
      echo "功能:"
      echo "  - 显示日志文件统计信息"
      echo "  - 查看最近的日志条目"
      echo "  - 清理过期的日志文件"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

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
  echo "❌ 错误: 未找到paper项目根目录"
  exit 1
fi

log_dir="$paper_root/.claude/logs"

# 检查日志目录
if [[ ! -d "$log_dir" ]]; then
  echo "ℹ️  日志目录不存在，将创建: $log_dir"
  mkdir -p "$log_dir"
fi

echo "📋 LQ日志管理报告"
echo "   Paper项目: $paper_root"
echo "   日志目录: $log_dir"
echo "   保留天数: $days"
echo ""

# 统计日志文件
declare -a log_files
total_size=0

# 检查是否有日志文件
if ! ls "$log_dir"/*.log >/dev/null 2>&1; then
  echo "ℹ️  没有找到日志文件"
  exit 0
fi

for file in "$log_dir"/*.log; do
  if [[ -f "$file" ]]; then
    log_files+=("$file")
    size=$(stat -c "%s" "$file" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
  fi
done

echo "📊 日志文件统计:"
echo "   文件数量: ${#log_files[@]}"
echo "   总大小: $(numfmt --to=iec $total_size)"
echo ""

# 按类型统计
echo "📈 按类型统计:"
for command_type in save_plan review_recent git_commit; do
  count=$(ls "$log_dir"/lq_${command_type}_*.log 2>/dev/null | wc -l)
  if [[ $count -gt 0 ]]; then
    echo "   lq_${command_type}: $count 个文件"
  fi
done

echo ""

# 显示最近的日志条目
echo "🕒 最近的日志条目:"
declare -A recent_logs
temp_file=$(mktemp)

# 收集最近的日志条目
for file in "${log_files[@]}"; do
  while IFS= read -r line; do
    if [[ -n "$line" && "$line" =~ ^\[.*\] ]]; then
      echo "$line|$file" >> "$temp_file"
    fi
  done < "$file"
done

# 按时间排序并显示前10条
if [[ -f "$temp_file" && -s "$temp_file" ]]; then
  sort "$temp_file" | tail -10 | while IFS='|' read -r line file; do
    command_type=$(basename "$file" .log | sed 's/lq_//' | sed 's/_.*//')
    timestamp=$(echo "$line" | sed 's/^\[//' | sed 's/\].*//')
    message=$(echo "$line" | sed 's/^.*\] //')
    printf "   %-15s %-20s %s\n" "$command_type" "$timestamp" "$message"
  done
else
  echo "   没有找到日志条目"
fi

rm -f "$temp_file"

echo ""

# 清理过期日志
if [[ $cleanup -eq 1 ]]; then
  echo "🧹 清理过期日志文件..."

  cutoff_date=$(date -d "$days days ago" +%Y-%m-%d)
  cleaned_count=0
  cleaned_size=0

  for file in "${log_files[@]}"; do
    # 从文件名提取日期
    filename=$(basename "$file")
    if [[ "$filename" =~ lq_.*_(2[0-9]{3}-[0-1][0-9]-[0-3][0-9])\.log$ ]]; then
      file_date="${BASH_REMATCH[1]}"
      if [[ "$file_date" < "$cutoff_date" ]]; then
        size=$(stat -c "%s" "$file" 2>/dev/null || echo 0)
        echo "   删除过期日志: $filename ($(numfmt --to=iec $size))"
        rm "$file"
        cleaned_count=$((cleaned_count + 1))
        cleaned_size=$((cleaned_size + size))
      fi
    fi
  done

  if [[ $cleaned_count -gt 0 ]]; then
    echo "   ✅ 已清理 $cleaned_count 个文件，释放 $(numfmt --to=iec $cleaned_size) 空间"
  else
    echo "   ℹ️  没有需要清理的过期文件"
  fi
else
  echo "💡 要清理过期日志，请使用 --cleanup 参数"
fi

echo ""
echo "📂 日志目录: $log_dir"