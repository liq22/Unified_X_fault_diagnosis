# /lq_git-commit — lq的智能提交命令

## 功能简介

分析git变更，按PHM-Vibench工厂架构分组文件，生成Conventional Commits标准提交信息并执行批量提交。

## 调用方式

Claude需要调用脚本：`.claude/scripts/lq_git_commit.sh`

### 可选参数
- `--path <path>`: 限定扫描路径（默认整个仓库）
- `--group "group1,group2"`: 指定要提交的分组（逗号分隔，默认全部）
- `--apply`: 实际执行提交操作
- `--dry-run`: 只显示预览（默认）
- `--paper-root <path>`: paper项目根目录（用于paper_docs分组判断）

## Claude需要完成的智能部分

### 1. 分析分组结果
- 解释每个分组包含的文件类型
- 评估提交的影响范围
- 建议合理的提交顺序

### 2. 选择提交分组
- 根据工作内容选择相关分组
- 避免混合不相关的更改
- 考虑依赖关系和先后顺序

### 3. 验证提交信息
- 检查生成的提交信息是否准确
- 确认符合Conventional Commits标准
- 调整描述内容使其更清晰

## 分组规则（按4层工厂架构）

### 工厂模块组（最高优先级）
```bash
data_factory    -> src/data_factory/           # 数据工厂：数据处理、读取器
model_factory   -> src/model_factory/          # 模型工厂：神经网络、ISFM
task_factory    -> src/task_factory/           # 任务工厂：任务定义、训练逻辑
trainer_factory -> src/trainer_factory/        # 训练工厂：训练编排、流水线
```

### 配置和文档组
```bash
configs         -> configs/                    # 实验配置、参数设置
paper_docs      -> paper/2025-10_foundation_model_0_metric/  # Paper项目文档
docs            -> docs/                       # 项目文档、说明
```

### 工具和测试组
```bash
utils           -> src/utils/                  # 工具函数、辅助模块
scripts         -> scripts/                    # 脚本文件、自动化工具
tests           -> test*/                      # 测试文件、验证代码
```

### 其他
```bash
misc            -> 其他文件                    # 杂项更改、根目录文件
```

## 调用示例

### 基本调用
```bash
# 预览所有变更
.claude/scripts/lq_git_commit.sh --dry-run

# 提交所有变更
.claude/scripts/lq_git_commit.sh --apply
```

### 指定分组
```bash
# 只提交模型工厂和配置更改
.claude/scripts/lq_git_commit.sh --apply --group "model_factory,configs"

# 提交特定路径的变更
.claude/scripts/lq_git_commit.sh --path src/model_factory/ --apply
```

### 组合使用
```bash
# 预览paper项目相关变更
.claude/scripts/lq_git_commit.sh --path paper/2025-10_foundation_model_0_metric/ --dry-run

# 提交工厂架构相关的更改
.claude/scripts/lq_git_commit.sh --apply --group "data_factory,model_factory,task_factory,trainer_factory"
```

### 可移植性调用
```bash
# 使用相对路径（推荐，跨项目兼容）
./.claude/scripts/lq_git_commit.sh --dry-run

# 适用于任何项目的工厂架构
./.claude/scripts/lq_git_commit.sh --apply --group "model_factory,configs"

# 自动检测paper项目（如果有paper目录）
./.claude/scripts/lq_git_commit.sh --paper-root ./paper/my_project --apply
```

## 提交信息格式

遵循Conventional Commits标准：
```bash
<type>(<scope>): <short description>

# 示例
refactor(model): improve heterogeneous batch handling in M_02_ISFM
feat(data): add new dataset reader for THU data
docs(paper): update experiment validation results
chore(config): tune hyperparameters for experiment 2
test(utils): add unit tests for data processing functions
```

### 提交类型说明
- `feat`: 新功能（新的读取器、模型组件等）
- `refactor`: 代码重构（优化、结构调整等）
- `fix`: 错误修复
- `docs`: 文档更新
- `test`: 测试相关
- `chore`: 构建工具、配置更改

## 操作流程

### 1. 分析阶段
- 扫描git状态获取变更文件
- 按分组规则对文件分类
- 生成符合规范的提交信息

### 2. 预览阶段（dry-run模式）
- 显示所有分组和文件列表
- 展示生成的提交信息
- 提供执行建议

### 3. 执行阶段（apply模式）
- 添加指定分组的文件到暂存区
- 按分组顺序执行提交
- 显示提交结果和统计信息

## 输出示例

### 预览输出
```bash
📋 LQ Git提交分析报告
   范围: 整个仓库
   检测到 8 个变更文件

📦 变更分组：
1) refactor(model): improve model architectures and ISFM components
   文件数量: 3
   文件列表:
      - src/model_factory/ISFM/M_02_ISFM.py
      - src/model_factory/ISFM/embedding/E_02_HSE_rec.py
      - src/model_factory/ISFM_Prompt/M_02_ISFM_Prompt.py

2) docs(paper): update paper documentation and experiment records
   文件数量: 2
   文件列表:
      - paper/2025-10_foundation_model_0_metric/docs/11_24/codex/plan.md

💡 要执行实际提交，请运行：
.claude/scripts/lq_git_commit.sh --apply --group "model_factory,paper_docs"
```

## 安全特性

- **默认dry-run**: 不加`--apply`只显示分析和预览
- **分组过滤**: 支持选择特定分组进行提交
- **路径限制**: 支持`--path`限定扫描范围
- **大文件检查**: 自动跳过过大的文件（>10MB）

## 错误处理

常见错误和解决方案：
- `未找到有效的分组`: 检查`--group`参数中的分组名称
- `没有检测到变更文件`: 检查是否有未提交的更改
- `无法添加文件`: 检查文件权限和git状态

## 最佳实践

### 1. 分组选择策略
- **逻辑相关**: 同一分组内的文件应该逻辑相关
- **粒度适中**: 避免单个提交包含过多文件
- **依赖考虑**: 考虑文件间的依赖关系

### 2. 提交频率
- **原子性**: 每个提交应该是一个完整的功能单元
- **及时性**: 完成一个功能单元后立即提交
- **描述性**: 提交信息应该清楚描述变更内容

### 3. 工作流集成
- **开发完成**: 运行`--dry-run`预览变更
- **确认分组**: 根据预览结果选择分组
- **执行提交**: 使用`--apply`和`--group`执行提交

## 相关命令

- `/lq_save-plan`: 保存相关的开发计划
- `/lq_review-recent`: 整理项目文档