# /lq_save-plan — lq的计划保存命令

## 功能简介

把当前会话中整理好的计划保存到paper项目的docs目录，生成标准格式的markdown文档。

## 调用方式

Claude需要调用脚本：`.claude/scripts/lq_save_plan.sh`

### 必填参数
- `--slug <slug>`: 文件标识符（必填）

### 可选参数
- `--subdir <codex|glm>`: 子目录（默认codex）
- `--paper-root <path>`: paper项目根目录（默认自动检测）
- `--apply`: 实际执行操作（默认dry-run）
- `--title <title>`: 文档标题（默认从slug生成）
- `--content <content>`: 计划内容（默认使用标准模板）

## Claude需要完成的智能部分

### 1. 生成slug
- 从计划内容中提取关键词
- 转换为小写，用下划线连接
- 示例：`ISFM prompt 单元测试计划` → `isfm_prompt_unit_test_plan`

### 2. 提取计划内容
- 识别对话中的计划段落
- 提取标题、背景、目标、具体步骤
- 格式化为标准markdown结构

### 3. 生成标题
- 基于计划内容生成简洁明了的标题
- 支持中文和英文

## 调用示例

### 基本调用
```bash
# dry-run预览
.claude/scripts/lq_save_plan.sh --slug isfm_prompt_unit_test

# 实际保存
.claude/scripts/lq_save_plan.sh --slug isfm_prompt_unit_test --apply
```

### 完整参数调用
```bash
# 保存到glm目录，自定义标题
.claude/scripts/lq_save_plan.sh \
  --slug experiment_validation \
  --subdir glm \
  --title "实验验证计划" \
  --apply
```

### 带内容的调用
```bash
# 保存完整内容
.claude/scripts/lq_save_plan.sh \
  --slug isfm_refactor \
  --title "ISFM重构计划" \
  --content "# ISFM重构计划

- Created at: 2025-11-25
- Project: 2025-10_foundation_model_0_metric
- Source: Claude Code (lq_save-plan)
- Status: TODO

## Plan
1. 重构M_02_ISFM的batch处理逻辑
2. 更新相关测试用例
3. 验证异构batch数据流稳定性" \
  --apply
```

### 可移植性调用
```bash
# 使用相对路径（推荐，跨项目兼容）
./.claude/scripts/lq_save_plan.sh --slug test_plan --apply

# 使用绝对路径（适用于脚本调用）
$(dirname $(pwd))/.claude/scripts/lq_save_plan.sh --slug test_plan --apply
```

## 输出路径格式

- 目录：`paper/2025-10_foundation_model_0_metric/docs/MM_DD/codex/` 或 `paper/2025-10_foundation_model_0_metric/docs/MM_DD/glm/`
- 文件：`<slug>_YYYY-MM-DD.md`
- 示例：`isfm_prompt_unit_test_2025-11-25.md`

## 标准模板结构

所有保存的计划都使用统一模板：
```markdown
# <标题>

- Created at: YYYY-MM-DD
- Project: 2025-10_foundation_model_0_metric
- Source: Claude Code (lq_save-plan)
- Status: TODO
- Category: plan

## Background
（1-3行背景说明）

## Objectives
- [ ] 目标1
- [ ] 目标2
- [ ] 目标3

## Plan
1. 步骤1
2. 步骤2
3. 步骤3

## Notes
- 依赖：
- 风险：
- 相关PR/实验：
```

## 安全特性

- **默认dry-run**: 不加`--apply`只显示预览
- **文件冲突检查**: 自动处理重名文件（添加v2后缀）
- **目录自动创建**: 自动创建所需的目录结构
- **路径验证**: 自动检测paper项目根目录

## 错误处理

常见错误和解决方案：
- `--slug 参数必填`: 必须提供slug参数
- `未找到paper项目根目录`: 检查当前目录是否在正确的项目结构下
- `--subdir 必须是 'codex' 或 'glm'`: 使用正确的子目录名称

## 相关命令

- `/lq_review-recent`: 查看和管理已保存的计划
- `/lq_git-commit`: 提交相关的代码更改