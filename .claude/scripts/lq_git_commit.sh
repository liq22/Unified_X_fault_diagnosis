#!/usr/bin/env bash
set -euo pipefail

# LQ Git Commit Script -三层架构实现
# 用法: lq_git_commit.sh [--path <path>] [--group "group1,group2"] [--apply] [--dry-run] [--paper-root <path>]

# 默认参数
scope_path=""
apply=0
dry_run=1
selected_groups=""
paper_root=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path) scope_path="$2"; shift 2 ;;
    --group) selected_groups="$2"; shift 2 ;;
    --apply) apply=1; dry_run=0; shift ;;
    --dry-run) dry_run=1; apply=0; shift ;;
    --paper-root) paper_root="$2"; shift 2 ;;
    --help)
      echo "用法: lq_git_commit.sh [--path <path>] [--group \"group1,group2\"] [--apply] [--dry-run] [--paper-root <path>]"
      echo ""
      echo "参数说明:"
      echo "  --path      可选，限定扫描路径（默认整个仓库）"
      echo "  --group     可选，指定要提交的分组（逗号分隔，默认全部）"
      echo "  --apply     可选，实际执行提交操作"
      echo "  --dry-run   可选，只显示预览（默认）"
      echo "  --paper-root 可选，paper项目根目录（用于特殊分组判断）"
      echo ""
      echo "分组规则:"
      echo "  data_factory    -> src/data_factory/"
      echo "  model_factory   -> src/model_factory/"
      echo "  task_factory    -> src/task_factory/"
      echo "  trainer_factory -> src/trainer_factory/"
      echo "  configs         -> configs/"
      echo "  paper_docs      -> paper/2025-10_foundation_model_0_metric/"
      echo "  utils           -> src/utils/"
      echo "  scripts         -> scripts/"
      echo "  tests           -> test*/"
      echo "  misc            -> 其他文件"
      echo ""
      echo "示例:"
      echo "  lq_git_commit.sh --dry-run"
      echo "  lq_git_commit.sh --apply --group \"model_factory,configs\""
      echo "  lq_git_commit.sh --path src/model_factory/ --apply"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 检测paper根目录（用于paper_docs分组）
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

# 获取git状态
if [[ -n "$scope_path" ]]; then
  git_status_output=$(git status --porcelain "$scope_path" 2>/dev/null)
  scope_desc="限定路径: $scope_path"
else
  git_status_output=$(git status --porcelain 2>/dev/null)
  scope_desc="整个仓库"
fi

if [[ -z "$git_status_output" ]]; then
  echo "ℹ️  没有检测到变更文件"
  exit 0
fi

# 解析文件列表
declare -a modified_files
declare -a untracked_files

while IFS= read -r line; do
  if [[ -n "$line" ]]; then
    file_status="${line:0:2}"
    file_path="${line:3}"

    if [[ "$file_status" == "??" ]]; then
      untracked_files+=("$file_path")
    else
      modified_files+=("$file_path")
    fi
  fi
done <<< "$git_status_output"

# 合并所有变更文件
all_files=("${modified_files[@]}" "${untracked_files[@]}")

# 过滤掉不需要提交的文件
declare -a filtered_files
for file in "${all_files[@]}"; do
  # 跳过临时文件和缓存文件
  if [[ "$file" =~ \.tmp$|\~$|\.log$|\.cache$|^\.git/|^\.claude/logs/ ]]; then
    continue
  fi

  # 跳过大文件（可选检查）
  if [[ -f "$file" ]]; then
    file_size=$(stat -c "%s" "$file" 2>/dev/null || echo 0)
    if [[ "$file_size" -gt 10485760 ]]; then  # 10MB
      echo "⚠️  跳过大文件: $file ($(du -h "$file" | cut -f1))"
      continue
    fi
  fi

  filtered_files+=("$file")
done

if [[ ${#filtered_files[@]} -eq 0 ]]; then
  echo "ℹ️  没有需要提交的文件"
  exit 0
fi

# 按分组规则分类文件
declare -A grouped_files
declare -A group_scopes
declare -A group_types
declare -A group_descriptions

# 初始化关联数组
for file in "${filtered_files[@]}"; do
  group=""
  scope=""
  type="chore"
  description=""

  # 按优先级匹配分组规则
  if [[ "$file" =~ ^src/data_factory/ ]]; then
    group="data_factory"
    scope="data"
    type="refactor"
    description="数据工厂组件和数据处理逻辑"
  elif [[ "$file" =~ ^src/model_factory/ ]]; then
    group="model_factory"
    scope="model"
    type="refactor"
    description="模型工厂组件和神经网络架构"
  elif [[ "$file" =~ ^src/task_factory/ ]]; then
    group="task_factory"
    scope="task"
    type="refactor"
    description="任务工厂组件和训练任务定义"
  elif [[ "$file" =~ ^src/trainer_factory/ ]]; then
    group="trainer_factory"
    scope="trainer"
    type="refactor"
    description="训练工厂组件和训练编排器"
  elif [[ "$file" =~ ^configs/ ]]; then
    group="configs"
    scope="config"
    type="chore"
    description="实验配置和参数设置"
  elif [[ -n "$paper_root" && "$file" =~ ^$paper_root/ ]]; then
    group="paper_docs"
    scope="docs"
    type="docs"
    description="Paper项目文档和实验记录"
  elif [[ "$file" =~ ^docs/ ]]; then
    group="docs"
    scope="docs"
    type="docs"
    description="项目文档和说明"
  elif [[ "$file" =~ ^src/utils/ ]]; then
    group="utils"
    scope="utils"
    type="refactor"
    description="工具函数和辅助模块"
  elif [[ "$file" =~ ^scripts/ ]]; then
    group="scripts"
    scope="scripts"
    type="chore"
    description="脚本文件和自动化工具"
  elif [[ "$file" =~ ^test|^tests ]]; then
    group="tests"
    scope="test"
    type="test"
    description="测试文件和验证代码"
  else
    group="misc"
    scope="chore"
    type="chore"
    description="其他文件和杂项更改"
  fi

  # 初始化键（如果不存在）
  if [[ -z "${grouped_files[$group]:-}" ]]; then
    grouped_files["$group"]=""
  fi
  if [[ -z "${group_scopes[$group]:-}" ]]; then
    group_scopes["$group"]="$scope"
  fi
  if [[ -z "${group_types[$group]:-}" ]]; then
    group_types["$group"]="$type"
  fi
  if [[ -z "${group_descriptions[$group]:-}" ]]; then
    group_descriptions["$group"]="$description"
  fi

  grouped_files["$group"]="${grouped_files[$group]} $file"
  group_scopes["$group"]="$scope"
  group_types["$group"]="$type"
  group_descriptions["$group"]="$description"
done

# 生成提交信息
declare -A commit_messages
for group in "${!grouped_files[@]}"; do
  files="${grouped_files[$group]}"
  scope="${group_scopes[$group]}"
  type="${group_types[$group]}"

  # 根据文件内容调整提交类型
  if [[ "$files" =~ \.(py|md)$ ]] && [[ "$type" == "refactor" ]]; then
    # 如果包含Python或Markdown文件，可能是新功能
    if [[ "$files" =~ __init__\.py$ ]]; then
      type="feat"
    fi
  fi

  # 生成简短描述
  case "$group" in
    "data_factory")
      commit_messages["$group"]="$type($scope): enhance data processing and reader components"
      ;;
    "model_factory")
      commit_messages["$group"]="$type($scope): improve model architectures and ISFM components"
      ;;
    "task_factory")
      commit_messages["$group"]="$type($scope): update task definitions and training logic"
      ;;
    "trainer_factory")
      commit_messages["$group"]="$type($scope): enhance training orchestration and pipeline"
      ;;
    "configs")
      commit_messages["$group"]="$type($scope): tune experiment configurations and parameters"
      ;;
    "paper_docs")
      commit_messages["$group"]="$type($scope): update paper documentation and experiment records"
      ;;
    "docs")
      commit_messages["$group"]="$type($scope): improve project documentation and guides"
      ;;
    "utils")
      commit_messages["$group"]="$type($scope): update utility functions and helper modules"
      ;;
    "scripts")
      commit_messages["$group"]="$type($scope): enhance automation scripts and tools"
      ;;
    "tests")
      commit_messages["$group"]="$type($scope): add and improve test coverage"
      ;;
    *)
      commit_messages["$group"]="$type($scope): miscellaneous updates and improvements"
      ;;
  esac
done

# 显示分析结果
echo "📋 LQ Git提交分析报告"
echo "   范围: $scope_desc"
echo "   检测到 ${#filtered_files[@]} 个变更文件"
echo ""

echo "📦 变更分组："
group_index=1
declare -a group_order
for group in "${!grouped_files[@]}"; do
  group_order+=("$group")
done

# 按字母顺序排序分组
IFS=$'\n' group_order=($(sort <<<"${group_order[*]}"))
unset IFS

# 如果指定了selected_groups，过滤分组
if [[ -n "$selected_groups" ]]; then
  IFS=',' read -ra requested_groups <<< "$selected_groups"
  declare -a filtered_groups
  for g in "${requested_groups[@]}"; do
    g=$(echo "$g" | xargs)  # trim whitespace
    for order_g in "${group_order[@]}"; do
      if [[ "$order_g" == "$g" ]]; then
        filtered_groups+=("$order_g")
        break
      fi
    done
  done
  group_order=("${filtered_groups[@]}")
fi

for group in "${group_order[@]}"; do
  files="${grouped_files[$group]}"
  commit_message="${commit_messages[$group]}"
  file_count=$(echo $files | wc -w)

  echo "$group_index) $commit_message"
  echo "   文件数量: $file_count"
  echo "   文件列表:"

  # 显示前几个文件作为示例
  echo "$files" | tr ' ' '\n' | head -3 | while read -r file; do
    if [[ -n "$file" ]]; then
      echo "      - $file"
    fi
  done
  if [[ $file_count -gt 3 ]]; then
    echo "      ... 还有 $((file_count - 3)) 个文件"
  fi
  echo ""
  ((group_index++))
done

# 如果没有找到分组，说明selected_groups无效
if [[ ${#group_order[@]} -eq 0 ]]; then
  echo "❌ 错误: 未找到有效的分组。可用分组："
  for group in "${!grouped_files[@]}"; do
    echo "  - $group"
  done
  exit 1
fi

# 如果是dry-run，显示执行建议
if [[ $dry_run -eq 1 ]]; then
  echo "🔍 Dry-run模式 - 将执行的操作预览："
  echo ""

  echo "# 1. 添加文件到暂存区"
  for group in "${group_order[@]}"; do
    files="${grouped_files[$group]}"
    for file in $files; do
      echo "git add \"$file\""
    done
  done

  echo ""
  echo "# 2. 提交文件"
  for group in "${group_order[@]}"; do
    commit_message="${commit_messages[$group]}"
    echo "git commit -m \"$commit_message\""
  done

  echo ""
  echo "💡 要执行实际提交，请运行："
  group_list=$(IFS=','; echo "${group_order[*]}")
  echo "lq_git_commit.sh --apply --group \"$group_list\""
  exit 0
fi

# 执行实际提交操作
echo ""
echo "🚀 开始执行提交操作..."

# 添加文件到暂存区
staged_count=0
for group in "${group_order[@]}"; do
  files="${grouped_files[$group]}"
  for file in $files; do
    if git add "$file" 2>/dev/null; then
      ((staged_count++))
    else
      echo "⚠️  警告: 无法添加文件: $file"
    fi
  done
done

echo "✅ 已暂存 $staged_count 个文件"

# 提交文件
commit_count=0
for group in "${group_order[@]}"; do
  commit_message="${commit_messages[$group]}"

  # 检查是否有属于这个分组的文件被暂存
  group_files="${grouped_files[$group]}"
  has_staged=false
  for file in $group_files; do
    if git diff --cached --quiet -- "$file" 2>/dev/null; then
      :  # 文件没有变化
    else
      has_staged=true
      break
    fi
  done

  if [[ "$has_staged" == "true" ]]; then
    if git commit -m "$commit_message" 2>/dev/null; then
      echo "✅ 提交成功: $commit_message"
      ((commit_count++))
    else
      echo "❌ 提交失败: $commit_message"
    fi
  fi
done

echo ""
echo "📊 提交完成:"
echo "   暂存文件: $staged_count"
echo "   成功提交: $commit_count"

if [[ $commit_count -gt 0 ]]; then
  echo "   最新提交: $(git log -1 --oneline)"
fi