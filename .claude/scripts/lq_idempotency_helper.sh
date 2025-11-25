#!/usr/bin/env bash
set -euo pipefail

# LQ Idempotency Helper Script - 幂等性检查和冲突处理工具
# 用法: lq_idempotency_helper.sh [--check <path>] [--resolve <path>]

# 默认参数
check_path=""
resolve_path=""
paper_root=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) check_path="$2"; shift 2 ;;
    --resolve) resolve_path="$2"; shift 2 ;;
    --paper-root) paper_root="$2"; shift 2 ;;
    --help)
      echo "用法: lq_idempotency_helper.sh [--check <path>] [--resolve <path>]"
      echo ""
      echo "参数说明:"
      echo "  --check <path>    检查指定路径的文件冲突"
      echo "  --resolve <path>  解决指定文件的冲突问题"
      echo "  --paper-root     paper项目根目录（默认自动检测）"
      echo ""
      echo "功能:"
      echo "  - 检查文件名冲突"
      echo "  - 生成唯一的文件名"
      echo "  - 检测重复操作"
      echo "  - 提供冲突解决方案"
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

# 生成唯一文件名的函数
generate_unique_filename() {
  local target_file="$1"
  local dir dirname filename basename extension counter

  dir=$(dirname "$target_file")
  filename=$(basename "$target_file")

  if [[ "$filename" =~ ^(.+)_(2[0-9]{3}-[0-1][0-9]-[0-3][0-9])\.([^.]+)$ ]]; then
    basename="${BASH_REMATCH[1]}"
    datestamp="${BASH_REMATCH[2]}"
    extension="${BASH_REMATCH[3]}"

    counter=2
    while [[ -f "$dir/${basename}_${datestamp}_v${counter}.${extension}" ]]; do
      ((counter++))
    done

    echo "$dir/${basename}_${datestamp}_v${counter}.${extension}"
  else
    # 如果不符合标准格式，使用简单数字后缀
    basename="${filename%.*}"
    extension="${filename##*.}"

    if [[ "$basename" == "$filename" ]]; then
      # 没有扩展名
      counter=2
      while [[ -f "$dir/${basename}_v${counter}" ]]; do
        ((counter++))
      done
      echo "$dir/${basename}_v${counter}"
    else
      # 有扩展名
      counter=2
      while [[ -f "$dir/${basename}_v${counter}.${extension}" ]]; do
        ((counter++))
      done
      echo "$dir/${basename}_v${counter}.${extension}"
    fi
  fi
}

# 检查文件是否已经在done目录
check_done_status() {
  local source_file="$1"
  local filename basename

  filename=$(basename "$source_file")
  basename="${filename%.*}"

  # 在done目录中查找同名或相似文件
  if [[ -d "$paper_root/docs/done" ]]; then
    while IFS= read -r -d '' done_file; do
      done_filename=$(basename "$done_file")
      if [[ "$done_filename" == "$filename" ]]; then
        echo "exact"
        return
      elif [[ "$done_filename" =~ ^${basename}(_v[0-9]+)?_.*\.md$ ]]; then
        echo "similar"
        return
      fi
    done < <(find "$paper_root/docs/done" -name "*.md" -print0 2>/dev/null)
  fi

  echo "not_found"
}

# 检查文件内容是否相似（简单检查）
check_content_similarity() {
  local file1="$1"
  local file2="$2"

  if [[ ! -f "$file1" || ! -f "$file2" ]]; then
    echo "different"
    return
  fi

  # 简单的内容比较：比较文件大小和前几行
  size1=$(stat -c "%s" "$file1" 2>/dev/null || echo 0)
  size2=$(stat -c "%s" "$file2" 2>/dev/null || echo 0)

  # 如果大小差异小于10%，认为可能相似
  size_diff=$((size1 - size2))
  size_avg=$(((size1 + size2) / 2))

  if [[ $size_avg -gt 0 && $((size_diff * 100 / size_avg)) -lt 10 ]]; then
    # 进一步检查标题行
    title1=$(head -1 "$file1" 2>/dev/null || echo "")
    title2=$(head -1 "$file2" 2>/dev/null || "")

    if [[ "$title1" == "$title2" ]]; then
      echo "similar"
    else
      echo "different"
    fi
  else
    echo "different"
  fi
}

# 检查模式
if [[ -n "$check_path" ]]; then
  echo "🔍 幂等性检查报告"
  echo "   检查路径: $check_path"
  echo ""

  if [[ ! -e "$check_path" ]]; then
    echo "✅ 路径不存在，可以安全创建"
    exit 0
  fi

  if [[ -f "$check_path" ]]; then
    filename=$(basename "$check_path")
    echo "⚠️  文件已存在: $filename"

    # 检查是否在done目录中
    done_status=$(check_done_status "$check_path")
    case "$done_status" in
      "exact")
        echo "❌ 文件已在done目录中，可能是重复操作"
        ;;
      "similar")
        echo "⚠️  done目录中存在相似文件，请检查是否重复"
        ;;
    esac

    # 提供解决方案
    unique_file=$(generate_unique_filename "$check_path")
    echo "💡 建议的解决方案:"
    echo "   使用唯一文件名: $(basename "$unique_file")"
    echo "   或使用 --resolve 参数自动解决冲突"

  elif [[ -d "$check_path" ]]; then
    file_count=$(find "$check_path" -name "*.md" | wc -l)
    echo "📁 目录存在，包含 $file_count 个markdown文件"

    # 检查目录中是否有冲突
    conflicts=0
    find "$check_path" -name "*_v[0-9]*.md" -print0 2>/dev/null | while IFS= read -r -d '' file; do
      conflicts=$((conflicts + 1))
    done

    if [[ $conflicts -gt 0 ]]; then
      echo "⚠️  发现 $conflicts 个可能的冲突文件（带v后缀）"
    else
      echo "✅ 目录中没有明显的文件冲突"
    fi
  fi

fi

# 解决冲突模式
if [[ -n "$resolve_path" ]]; then
  echo "🔧 冲突解决处理"
  echo "   目标路径: $resolve_path"
  echo ""

  if [[ ! -e "$resolve_path" ]]; then
    echo "ℹ️  路径不存在，无需解决冲突"
    exit 0
  fi

  if [[ -f "$resolve_path" ]]; then
    unique_file=$(generate_unique_filename "$resolve_path")

    echo "原始文件: $resolve_path"
    echo "解决方案: $unique_file"
    echo ""

    read -p "是否创建唯一文件名版本？(y/N): " confirm
    if [[ "$confirm" =~ ^[yY] ]]; then
      if cp "$resolve_path" "$unique_file"; then
        echo "✅ 已创建唯一版本: $(basename "$unique_file")"
        echo "💡 原文件保持不变，您可以继续处理"
      else
        echo "❌ 创建唯一版本失败"
        exit 1
      fi
    else
      echo "❌ 用户取消操作"
    fi

  elif [[ -d "$resolve_path" ]]; then
    echo "ℹ️  目录冲突无法自动解决"
    echo "💡 请手动检查目录内容并处理冲突文件"
  fi
fi

# 如果没有指定操作，显示使用帮助
if [[ -z "$check_path" && -z "$resolve_path" ]]; then
  echo "ℹ️  请指定操作类型："
  echo "   --check <path>   检查文件冲突"
  echo "   --resolve <path> 解决文件冲突"
  echo ""
  echo "示例："
  echo "   lq_idempotency_helper.sh --check /path/to/file.md"
  echo "   lq_idempotency_helper.sh --resolve /path/to/file.md"
fi