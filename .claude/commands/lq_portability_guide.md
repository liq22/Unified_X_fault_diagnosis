# LQ命令系统可移植性使用指南

## 概述

LQ命令系统设计为高度可移植，可以轻松迁移到任何项目中使用。本指南说明如何在不同的项目和环境中部署和使用LQ命令系统。

## 系统架构

LQ命令系统采用三层架构设计：

```
项目根目录/
├── .claude/
│   ├── commands/              # Claude命令说明层
│   │   ├── lq_save_plan.md
│   │   ├── lq_review_recent.md
│   │   ├── lq_git_commit.md
│   │   └── lq_portability_guide.md
│   ├── scripts/               # CLI脚本层（核心执行逻辑）
│   │   ├── lq_save_plan.sh
│   │   ├── lq_review_recent.sh
│   │   ├── lq_git_commit.sh
│   │   ├── lq_log_manager.sh
│   │   └── lq_idempotency_helper.sh
│   └── logs/                  # 操作日志（运行时生成）
├── paper/                     # 文档输出层（可选）
│   └── your_project/
│       ├── docs/
│       └── .claude/logs/
└── [其他项目文件]
```

## 快速部署到新项目

### 1. 复制LQ命令系统

```bash
# 复制整个.claude目录到新项目
cp -r /path/to/original/project/.claude /path/to/new/project/

# 或者只复制commands和scripts目录
mkdir -p /path/to/new/project/.claude/{commands,scripts}
cp /path/to/original/project/.claude/commands/* /path/to/new/project/.claude/commands/
cp /path/to/original/project/.claude/scripts/* /path/to/new/project/.claude/scripts/

# 设置脚本执行权限
chmod +x /path/to/new/project/.claude/scripts/*.sh
```

### 2. 基本配置验证

```bash
cd /path/to/new/project

# 验证脚本路径
./.claude/scripts/lq_save_plan.sh --help
./.claude/scripts/lq_review_recent.sh --help
./.claude/scripts/lq_git_commit.sh --help
```

## 核心命令使用

### 计划保存命令

```bash
# 基本使用
./.claude/scripts/lq_save_plan.sh --slug project_plan --apply

# 完整参数
./.claude/scripts/lq_save_plan.sh \
  --slug feature_development \
  --subdir codex \
  --title "新功能开发计划" \
  --apply

# 自定义内容
./.claude/scripts/lq_save_plan.sh \
  --slug api_design \
  --title "API设计文档" \
  --content "# API设计文档
- Created at: $(date +%Y-%m-%d)
- Project: my_awesome_project
- Source: Claude Code (lq_save-plan)
- Status: TODO

## API设计要点
1. RESTful接口规范
2. 数据格式定义
3. 错误处理机制" \
  --apply
```

### 文档整理命令

```bash
# 基本使用
./.claude/scripts/lq_review_recent.sh --days 3 --apply

# 指定子目录
./.claude/scripts/lq_review_recent.sh --subdir codex --apply
./.claude/scripts/lq_review_recent.sh --subdir glm --apply

# 自动模式
./.claude/scripts/lq_review_recent.sh --days 7 --auto --apply

# 指定paper项目路径
./.claude/scripts/lq_review_recent.sh \
  --paper-root ./paper/my_project \
  --days 5 \
  --apply
```

### Git提交命令

```bash
# 预览所有变更
./.claude/scripts/lq_git_commit.sh --dry-run

# 提交特定分组
./.claude/scripts/lq_git_commit.sh --apply --group "model_factory,configs"

# 限定路径范围
./.claude/scripts/lq_git_commit.sh --path src/ --apply

# 完整示例
./.claude/scripts/lq_git_commit.sh \
  --path src/ \
  --group "model_factory,task_factory" \
  --apply
```

## 项目适配指南

### 1. Paper项目结构适配

如果您的项目没有paper目录，可以：

```bash
# 方案A：创建标准paper目录
mkdir -p paper/your_project/{docs/{MM_DD/{codex,glm}},done,.claude/logs}

# 方案B：使用自定义路径
./.claude/scripts/lq_save_plan.sh \
  --paper-root ./docs \
  --slug my_plan \
  --apply

# 方案C：不使用paper结构（直接保存到当前目录）
./.claude/scripts/lq_save_plan.sh \
  --slug my_plan \
  --paper-root . \
  --apply
```

### 2. 工厂架构适配

如果您不使用PHM-Vibench的工厂架构，可以修改`lq_git_commit.sh`中的分组规则：

```bash
# 查看当前分组规则
grep -n "分组规则" .claude/scripts/lq_git_commit.sh

# 根据需要修改分组映射
# 例如：web项目常见分组
frontend        -> src/frontend/           # 前端代码
backend         -> src/backend/            # 后端代码
api             -> src/api/                 # API接口
config          -> config/                  # 配置文件
docs            -> docs/                    # 项目文档
```

### 3. 状态管理适配

LQ命令系统支持灵活的状态管理：

```markdown
# 标准状态字段
- Status: TODO    # 待完成
- Status: WIP     # 进行中
- Status: DONE    # 已完成

# 可选分类字段
- Category: plan
- Category: refactor
- Category: feature
- Category: bugfix
- Category: docs
```

## 高级配置

### 1. 环境变量支持

```bash
# 设置默认paper项目根目录
export LQ_PAPER_ROOT="$HOME/projects/my_project"

# 设置默认日志级别
export LQ_LOG_LEVEL="info"

# 设置默认子目录
export LQ_DEFAULT_SUBDIR="codex"
```

### 2. 脚本别名配置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias lq-plan='./.claude/scripts/lq_save_plan.sh'
alias lq-review='./.claude/scripts/lq_review_recent.sh'
alias lq-commit='./.claude/scripts/lq_git_commit.sh'
alias lq-log='./.claude/scripts/lq_log_manager.sh'

# 重新加载配置
source ~/.bashrc  # 或 source ~/.zshrc
```

### 3. 集成到IDE/编辑器

#### VS Code配置
```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "LQ Save Plan",
            "type": "shell",
            "command": "./.claude/scripts/lq_save_plan.sh",
            "args": ["--slug", "${input:slug}", "--apply"],
            "group": "build"
        },
        {
            "label": "LQ Review Recent",
            "type": "shell",
            "command": "./.claude/scripts/lq_review_recent.sh",
            "args": ["--apply"],
            "group": "build"
        }
    ],
    "inputs": [
        {
            "id": "slug",
            "description": "Plan slug",
            "default": "development_plan",
            "type": "promptString"
        }
    ]
}
```

## 故障排除

### 常见问题

1. **权限错误**
```bash
# 解决方案：设置执行权限
chmod +x .claude/scripts/*.sh
```

2. **路径找不到**
```bash
# 检查当前目录结构
ls -la .claude/
ls -la .claude/scripts/
ls -la .claude/commands/

# 使用绝对路径测试
/path/to/project/.claude/scripts/lq_save_plan.sh --help
```

3. **Paper项目检测失败**
```bash
# 手动指定paper项目路径
./.claude/scripts/lq_save_plan.sh \
  --paper-root ./my_docs \
  --slug test_plan \
  --apply
```

4. **Git操作失败**
```bash
# 检查git仓库状态
git status
git remote -v

# 确保有提交权限
git config user.name
git config user.email
```

### 调试模式

```bash
# 启用详细日志
export LQ_DEBUG=1

# 查看脚本执行过程
bash -x .claude/scripts/lq_save_plan.sh --slug test

# 检查日志文件
./.claude/scripts/lq_log_manager.sh --cleanup
```

## 最佳实践

### 1. 工作流集成

```bash
# 完整的开发工作流
# 1. 开始开发前制定计划
./.claude/scripts/lq_save_plan.sh --slug feature_x --apply

# 2. 开发过程中定期整理
./.claude/scripts/lq_review_recent.sh --days 1 --apply

# 3. 开发完成后提交代码
./.claude/scripts/lq_git_commit.sh --dry-run
./.claude/scripts/lq_git_commit.sh --apply --group "frontend,backend"
```

### 2. 团队协作

```bash
# 1. 统一命名规范
# 使用一致的项目slug命名：feature_name, bugfix_id, etc.

# 2. 定期文档整理
# 建议每天运行一次lq_review_recent

# 3. 代码提交规范
# 使用lq_git_commit确保提交信息的一致性
```

### 3. 维护和升级

```bash
# 定期清理日志
./.claude/scripts/lq_log_manager.sh --cleanup --days 30

# 检查文件冲突
./.claude/scripts/lq_idempotency_helper.sh --check ./docs/

# 备份重要配置
cp -r .claude ../lq_system_backup_$(date +%Y%m%d)
```

## 版本兼容性

- **Bash版本**: 需要 bash 4.0+ (关联数组支持)
- **Git版本**: 需要 git 2.0+
- **操作系统**: Linux, macOS, Windows (WSL)
- **依赖工具**: 标准Unix工具 (find, grep, sed, awk)

## 支持和贡献

如果您在使用过程中遇到问题或有改进建议，请：

1. 检查日志文件获取详细错误信息
2. 查看本文档的故障排除部分
3. 提交issue或贡献代码改进

---

**最后更新**: 2025-11-25
**版本**: 2.0 (可移植性增强版)
**兼容性**: 跨项目、跨平台