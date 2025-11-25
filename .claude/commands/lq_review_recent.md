# /lq_review-recent — lq的文档整理命令

## 功能简介

回顾paper项目最近N天的docs，区分DONE/TODO/WIP状态，自动归档DONE文件并生成TODO聚合文档。

## 调用方式

Claude需要调用脚本：`.claude/scripts/lq_review_recent.sh`

### 可选参数
- `--days <N>`: 回顾天数（默认3天，范围1-30）
- `--subdir <codex|glm|all>`: 子目录（默认all）
- `--auto`: 自动模式，跳过交互确认
- `--apply`: 实际执行操作（默认dry-run）
- `--paper-root <path>`: paper项目根目录（默认自动检测）

## Claude需要完成的智能部分

### 1. 生成时间范围建议
- 根据用户工作节奏建议合适的天数
- 周末/工作日模式调整

### 2. 选择子目录
- `codex`: 技术实现相关的文档
- `glm`: 管理和流程相关的文档
- `all`: 扫描所有子目录（推荐）

### 3. 分析聚合结果
- 解释主题分组的含义
- 提供时间线建议
- 识别紧急任务

## 调用示例

### 基本调用
```bash
# 预览最近3天的文档
.claude/scripts/lq_review_recent.sh

# 实际整理最近5天的文档
.claude/scripts/lq_review_recent.sh --days 5 --apply
```

### 指定子目录
```bash
# 只整理codex目录
.claude/scripts/lq_review_recent.sh --subdir codex --apply

# 整理所有子目录
.claude/scripts/lq_review_recent.sh --subdir all --apply
```

### 自动模式
```bash
# 自动整理最近一周，不询问确认
.claude/scripts/lq_review_recent.sh --days 7 --auto --apply
```

### 可移植性调用
```bash
# 使用相对路径（推荐，跨项目兼容）
./.claude/scripts/lq_review_recent.sh --days 3 --apply

# 自动检测paper项目（适用于任何包含paper的项目）
./.claude/scripts/lq_review_recent.sh --paper-root ./paper/project_name --apply
```

## 扫描规则

### 1. 时间范围
- 基于文件mtime（修改时间）
- 扫描`docs/MM_DD/`目录结构
- 只处理存在markdown文件的日期目录

### 2. 状态识别（固定规则）
- **DONE**: 文件包含 `- Status: DONE`
- **WIP**: 文件包含 `- Status: WIP`
- **TODO**: 其他所有情况（默认状态）

### 3. 主题分组
- **ISFM基础设施**: 文件名包含`isfm|ISFM`
- **实验配置**: 文件名包含`experiment|Experiment`
- **Prompt相关**: 文件名包含`prompt|Prompt`
- **文档整理**: 文件名包含`docs|docs_|validation`
- **代码重构**: 文件名包含`refactor|refactoring`
- **其他**: 不符合以上规则的文件

## 操作流程

### 1. 扫描阶段
- 检测指定天数内的日期目录
- 扫描每个目录下的markdown文件
- 读取Status字段确定文件状态

### 2. 分析阶段
- 统计各状态的文件数量
- 按主题对TODO/WIP文件分组
- 生成聚合文档内容

### 3. 执行阶段（apply模式）
- 创建聚合文档：`docs/MM_DD/codex/lq_open_todos_aggregation_YYYY-MM-DD.md`
- 移动DONE文件到：`docs/done/MM_DD/{subdir}/`
- 为移动的文件添加`Moved at`元数据

## 输出格式

### 聚合文档结构
```markdown
# Open TODOs Aggregation — YYYY-MM-DD

- Created at: YYYY-MM-DD
- Project: 2025-10_foundation_model_0_metric
- Source: Claude Code (/lq_review-recent)
- Status: TODO
- Review period (days): N
- Scan subdirs: [codex, glm]

## 1. 已归档的计划（Status: DONE）
- `docs/11_23/codex/xxx_2025-11-23.md`

## 2. 仍在进行中的计划（TODO/WIP）
### 2.1 ISFM基础设施
- `isfm_prompt_unit_test_plan_2025-11-24.md`

## 3. 推荐时间线
**今天可以完成：**
- 检查代码提交状态
- 更新相关文档

**本周内需要推进：**
- 完成ISFM相关任务
- 推进实验配置工作
```

## 安全特性

- **默认dry-run**: 不加`--apply`只显示分析和预览
- **文件移动确认**: apply模式才实际移动文件
- **目录自动创建**: 自动创建done目录结构
- **元数据保护**: 移动文件时保留Status字段并添加移动信息

## 错误处理

常见错误和解决方案：
- `--days 必须是1-30之间的整数`: 使用有效的天数范围
- `--subdir 必须是 'codex'、'glm' 或 'all'`: 使用正确的子目录名称
- `未找到paper项目根目录`: 检查当前目录是否在正确的项目结构下

## 日志记录

操作记录保存在：`paper/2025-10_foundation_model_0_metric/.claude/logs/lq_review_recent_YYYY-MM-DD.log`

## 相关命令

- `/lq_save-plan`: 创建新的计划文档
- `/lq_git-commit`: 提交相关的代码更改

## 最佳实践

1. **定期整理**: 建议每天或每两天运行一次
2. **Status维护**: 手动更新文档的Status字段
3. **主题一致性**: 使用一致的文件命名便于分组
4. **归档策略**: DONE文件移动后定期检查是否需要进一步处理