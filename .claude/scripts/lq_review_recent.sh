#!/usr/bin/env bash
set -euo pipefail

# LQ Review Recent Script -三层架构实现
# 用法: lq_review_recent.sh [--days <N>] [--subdir <codex|glm|all>] [--auto] [--apply] [--paper-root <path>]

# 默认参数
days=3
subdir="all"
auto=0
apply=0
paper_root=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --days) days="$2"; shift 2 ;;
    --subdir) subdir="$2"; shift 2 ;;
    --auto) auto=1; shift ;;
    --apply) apply=1; shift ;;
    --paper-root) paper_root="$2"; shift 2 ;;
    --help)
      echo "用法: lq_review_recent.sh [--days <N>] [--subdir <codex|glm|all>] [--auto] [--apply] [--paper-root <path>]"
      echo ""
      echo "参数说明:"
      echo "  --days      可选，回顾天数（默认3天）"
      echo "  --subdir    可选，子目录（codex、glm或all，默认all）"
      echo "  --auto      可选，自动模式，跳过交互确认"
      echo "  --apply     可选，实际执行操作（默认dry-run）"
      echo "  --paper-root 可选，paper项目根目录（默认自动检测）"
      echo ""
      echo "示例:"
      echo "  lq_review_recent.sh --days 5 --apply"
      echo "  lq_review_recent.sh --subdir codex --auto --apply"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 验证参数
if [[ ! "$days" =~ ^[0-9]+$ || "$days" -lt 1 || "$days" -gt 30 ]]; then
  echo "❌ 错误: --days 必须是1-30之间的整数"
  exit 1
fi

if [[ "$subdir" != "codex" && "$subdir" != "glm" && "$subdir" != "all" ]]; then
  echo "❌ 错误: --subdir 必须是 'codex'、'glm' 或 'all'"
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

# 创建日志目录
log_dir="$paper_root/.claude/logs"
mkdir -p "$log_dir"
today_date=$(date +%Y-%m-%d)
log_file="$log_dir/lq_review_recent_${today_date}.log"

# 日志函数
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

# 设置要扫描的子目录
declare -a subdirs_to_scan
if [[ "$subdir" == "all" ]]; then
  subdirs_to_scan=("codex" "glm")
else
  subdirs_to_scan=("$subdir")
fi

# 生成要检查的日期目录
declare -a date_dirs
for ((i=0; i<days; i++)); do
  date_dir=$(date -d "$i days ago" +%m_%d)
  if [[ -d "$paper_root/docs/$date_dir" ]]; then
    date_dirs+=("$date_dir")
  fi
done

if [[ ${#date_dirs[@]} -eq 0 ]]; then
  echo "ℹ️  在最近 $days 天内没有找到docs目录"
  log_message "INFO: No docs directories found in last $days days"
  exit 0
fi

# 扫描文档
declare -a found_files
declare -A file_status
declare -A file_paths
declare -A file_subdirs

echo "🔍 扫描文档..."
echo "   时间范围: 最近 $days 天"
echo "   日期目录: ${date_dirs[*]}"
echo "   子目录: ${subdirs_to_scan[*]}"
echo ""

for date_dir in "${date_dirs[@]}"; do
  for scan_subdir in "${subdirs_to_scan[@]}"; do
    scan_path="$paper_root/docs/$date_dir/$scan_subdir"
    if [[ -d "$scan_path" ]]; then
      for file in "$scan_path"/*.md; do
        if [[ -f "$file" ]]; then
          found_files+=("$file")
          file_paths["$file"]="$date_dir/$scan_subdir"
          file_subdirs["$file"]="$scan_subdir"

          # 检查Status字段
          status="TODO"  # 默认状态
          if grep -q "^- Status: DONE" "$file" 2>/dev/null; then
            status="DONE"
          elif grep -q "^- Status: WIP" "$file" 2>/dev/null; then
            status="WIP"
          fi
          file_status["$file"]="$status"

          filename=$(basename "$file")
          echo "  发现文件: $filename (状态: $status)"
        fi
      done
    fi
  done
done

if [[ ${#found_files[@]} -eq 0 ]]; then
  echo "ℹ️  在指定范围内没有找到markdown文件"
  log_message "INFO: No markdown files found in scan range"
  exit 0
fi

echo ""
echo "📊 扫描结果:"
echo "   总文件数: ${#found_files[@]}"
echo "   DONE: $(echo "${found_files[@]}" | tr ' ' '\n' | while read -r f; do [[ "${file_status[$f]}" == "DONE" ]] && echo "$f"; done | wc -l)"
echo "   WIP: $(echo "${found_files[@]}" | tr ' ' '\n' | while read -r f; do [[ "${file_status[$f]}" == "WIP" ]] && echo "$f"; done | wc -l)"
echo "   TODO: $(echo "${found_files[@]}" | tr ' ' '\n' | while read -r f; do [[ "${file_status[$f]}" == "TODO" ]] && echo "$f"; done | wc -l)"

# 生成聚合文档内容
aggregation_content="# Open TODOs Aggregation — $today_date

- Created at: $today_date
- Project: $(basename "$paper_root")
- Source: Claude Code (/lq_review-recent)
- Status: TODO
- Review period (days): $days
- Scan subdirs: ${subdirs_to_scan[*]}

## 1. 已归档的计划（Status: DONE）

"

# 添加DONE文件列表
done_files=()
for file in "${found_files[@]}"; do
  if [[ "${file_status[$file]}" == "DONE" ]]; then
    done_files+=("$file")
    relative_path=$(echo "$file" | sed "s|$paper_root/||")
    aggregation_content+="- \`$relative_path\`"$'\n'
  fi
done

if [[ ${#done_files[@]} -eq 0 ]]; then
  aggregation_content+="最近$days天内没有完成的计划需要归档。"$'\n'
fi

# 添加仍在进行中的计划
aggregation_content+=$'\n## 2. 仍在进行中的计划（TODO/WIP）\n\n'

# 按主题分组
declare -A theme_groups
for file in "${found_files[@]}"; do
  status="${file_status[$file]}"
  if [[ "$status" == "TODO" || "$status" == "WIP" ]]; then
    filename=$(basename "$file")

    # 简单的主题分组
    theme="其他"
    if [[ "$filename" =~ isfm|ISFM ]]; then
      theme="ISFM基础设施"
    elif [[ "$filename" =~ experiment|Experiment ]]; then
      theme="实验配置"
    elif [[ "$filename" =~ prompt|Prompt ]]; then
      theme="Prompt相关"
    elif [[ "$filename" =~ docs|docs_|validation ]]; then
      theme="文档整理"
    elif [[ "$filename" =~ refactor|refactoring ]]; then
      theme="代码重构"
    fi

    theme_groups["$theme"]+="$file|"
  fi
done

for theme in "${!theme_groups[@]}"; do
  aggregation_content+="### $theme"$'\n'$'\n'

  IFS='|' read -ra files <<< "${theme_groups[$theme]}"
  for file in "${files[@]}"; do
    if [[ -n "$file" ]]; then
      filename=$(basename "$file")
      aggregation_content+="- \`$filename\`"$'\n'
    fi
  done
  aggregation_content+=$'\n'
done

# 添加建议的时间线
aggregation_content+="## 3. 推荐时间线\n\n"
aggregation_content+="**今天可以完成：**\n"
aggregation_content+="- 检查代码提交状态，确保已完成的功能已正确提交\n"
aggregation_content+="- 更新相关文档和配置文件\n"
aggregation_content+="- 运行基础功能测试验证\n\n"
aggregation_content+="**本周内需要推进：**\n"
theme_count=$((${#theme_groups[@]}))
if [[ $theme_count -gt 0 ]]; then
  for theme in "${!theme_groups[@]}"; do
    aggregation_content+="- 完成 $theme 相关的关键任务\n"
  done
else
  aggregation_content+="- 继续当前的开发任务\n"
fi

aggregation_content+=$'\n**需要进一步思考/等待外部条件：**\n'
aggregation_content+="- 根据实验结果调整后续策略\n"
aggregation_content+="- 等待团队成员反馈或相关数据\n"
aggregation_content+="- 评估当前方案的技术可行性\n"

# 创建聚合文档
aggregation_dir="$paper_root/docs/$(date +%m_%d)/codex"
mkdir -p "$aggregation_dir"
aggregation_file="$aggregation_dir/lq_open_todos_aggregation_${today_date}.md"

echo ""
echo "📝 将生成聚合文档:"
echo "   路径: $aggregation_file"
echo "   包含: ${#theme_groups[@]} 个主题分组"

# 如果是dry-run，只显示预览
if [[ $apply -eq 0 ]]; then
  echo ""
  echo "🔍 Dry-run模式 - 聚合文档预览："
  echo "======================================"
  echo "$aggregation_content" | head -50
  if [[ ${#aggregation_content} -gt 2000 ]]; then
    echo "..."
    echo "(内容过长，只显示前50行)"
  fi
  echo "======================================"
  echo ""
  echo "💡 要执行实际操作，请添加 --apply 参数"

  # 预览文件移动操作
  if [[ ${#done_files[@]} -gt 0 ]]; then
    echo ""
    echo "📁 将移动的DONE文件（需要--apply执行）："
    for file in "${done_files[@]}"; do
      filename=$(basename "$file")
      file_subdir="${file_subdirs[$file]}"
      target_dir="$paper_root/docs/done/$(date +%m_%d)/$file_subdir"
      echo "  $filename → docs/done/$(date +%m_%d)/$file_subdir/"
    done
  fi

  log_message "DRY-RUN: Review recent preview - ${#found_files[@]} files scanned"
  exit 0
fi

# 执行实际操作
echo "$aggregation_content" > "$aggregation_file"
echo "✅ 已生成聚合文档: $aggregation_file"

# 移动DONE文件到done目录
if [[ ${#done_files[@]} -gt 0 ]]; then
  echo ""
  echo "📁 移动DONE文件..."

  for file in "${done_files[@]}"; do
    if [[ -f "$file" ]]; then
      filename=$(basename "$file")
      file_date_dir="${file_paths[$file]}"
      file_subdir="${file_subdirs[$file]}"
      target_dir="$paper_root/docs/done/$(date +%m_%d)/$file_subdir"

      # 创建目标目录
      mkdir -p "$target_dir"

      # 移动文件
      mv "$file" "$target_dir/$filename"

      # 确保有Status字段
      if ! grep -q "^- Status:" "$target_dir/$filename" 2>/dev/null; then
        temp_file=$(mktemp)
        {
          echo "---"
          echo "- Status: DONE"
          echo "- Moved at: $(date +%Y-%m-%d)"
          echo "- Original path: $file_date_dir"
          echo "---"
          echo ""
          cat "$target_dir/$filename"
        } > "$temp_file"
        mv "$temp_file" "$target_dir/$filename"
      fi

      relative_target=$(echo "$target_dir/$filename" | sed "s|$paper_root/||")
      echo "  ✅ 已归档: $relative_target"
    fi
  done
fi

# 记录日志
log_message "SUCCESS: Review recent - ${#found_files[@]} files scanned, ${#done_files[@]} moved, 1 aggregation created"

echo ""
echo "📊 操作完成:"
echo "   扫描文件: ${#found_files[@]}"
echo "   归档文件: ${#done_files[@]}"
echo "   聚合文档: $aggregation_file"

# 显示相对路径
relative_aggregation=$(echo "$aggregation_file" | sed "s|$paper_root/||")
echo "📂 聚合文档相对路径: $relative_aggregation"