# Claude Code AI研究助手代理索引 (18个专业代理)

## 🎨 颜色标识系统
- **📚 Research类** (蓝色) - 学术研究、文献分析、知识发现
- **✍️ Writer类** (绿色) - 论文撰写、格式优化、质量控制  
- **💻 Coder类** (橙色) - 代码开发、调试优化、工业部署

### 查看彩色Agent列表
```bash
# 按类别显示（推荐）
python scripts/show_agents_colored.py

# 列表形式显示
python scripts/show_agents_colored.py --list

# 显示描述
python scripts/show_agents_colored.py --description

# 只显示特定类别
python scripts/show_agents_colored.py --category research
python scripts/show_agents_colored.py --category writer  
python scripts/show_agents_colored.py --category coder

# 显示统计摘要
python scripts/show_agents_colored.py --summary
```

---

## 📚 Research类代理 (7个) - 🔵 蓝色标识
专注于学术研究、文献分析、知识发现

### 文献与知识管理
- **`research-academic`** - 学术文献搜索与分析专家，整合多数据库搜索能力
- **`research-literature`** - 文献协调器，系统性文献综述与证据合成
- **`research-knowledge-graph`** - 知识图谱构建，可视化研究领域连接关系
- **`research-semantic-scholar`** - Semantic Scholar API集成，访问200M+学术论文数据库

### 研究策略与创新
- **`research-gap-identifier`** - 研究空白识别，发现未探索的研究机会
- **`research-hypothesis`** - AI驱动假设生成，基于文献分析产生创新假设
- **`research-trends`** - 研究趋势分析与未来方向预测

## ✍️ Writer类代理 (8个) - 🟢 绿色标识
专注于论文撰写、格式优化、质量控制

### 集成写作集群 (5-in-1 Clusters)
- **`writer-intro-cluster`** - 引言集群：背景+文献+问题+贡献+预告
- **`writer-method-cluster`** - 方法集群：概述+算法+数学+实现+复杂度
- **`writer-results-cluster`** - 结果集群：实验+数据+图表+对比+显著性
- **`writer-discussion-cluster`** - 讨论集群：发现+理论+局限+影响+未来
- **`writer-format-cluster`** - 格式集群：摘要+标题+结构+语言+声明

### 质量控制与优化
- **`writer-cache-manager`** - 智能缓存管理，加速研究工作流程
- **`writer-quality-controller`** - Nature级质量控制，四重门控验证系统
- **`writer-style-formatter`** - 期刊特定格式化，支持多种顶级期刊风格

## 💻 Coder类代理 (3个) - 🟠 橙色标识
专注于代码开发、调试优化、工业部署

- **`coder-reviewer`** - 代码审查专家，确保代码质量和安全性
- **`coder-debugger`** - 调试与错误诊断专家，快速定位和修复问题
- **`coder-industrial-ai`** - 工业AI专家，PyTorch/JAX边缘部署和优化

## 🎯 使用指南

### 按类别调用代理
- **Research类**: `/agent research-[name]` - 学术研究和文献分析
- **Writer类**: `/agent writer-[name]` - 论文撰写和质量控制  
- **Coder类**: `/agent coder-[name]` - 代码开发和工业部署

### Research类工作流
```bash
# 完整研究发现流程
/agent research-literature: "搜索和综合[主题]文献"
/agent research-knowledge-graph: "构建[领域]知识图谱" 
/agent research-gap-identifier: "识别[领域]研究空白"
/agent research-hypothesis: "生成创新研究假设"
/agent research-trends: "分析研究趋势预测"
```

### Writer类工作流
```bash
# D1→D2→D3→D4→D5 顺序写作
/agent writer-intro-cluster --task background: "构建研究背景"
/agent writer-method-cluster --task overview: "设计方法架构"
/agent writer-results-cluster --task experiment: "设计实验方案"
/agent writer-discussion-cluster --task findings: "总结核心发现"
/agent writer-format-cluster --task abstract: "创建结构化摘要"
```

### Coder类工作流
```bash
# 代码开发与部署流程
/agent coder-industrial-ai: "用JAX实现高性能算法"
/agent coder-reviewer: "审查代码质量和安全性"
/agent coder-debugger: "调试性能和错误问题"
```

### 质量保证工作流
```bash
# 四重质量验证
/agent writer-quality-controller: "执行完整质量评估"
/agent writer-style-formatter: "应用[Nature/Science]格式"
/agent writer-cache-manager: "优化工作流程性能"
```

## 📊 代理能力矩阵

| 代理类型 | 研究发现 | 写作专业 | 代码开发 | 质量控制 |
|---------|---------|---------|---------|----------|
| Research类 | ✅ | ⭐ | ⭐ | ⭐ |
| Writer类 | ⭐ | ✅ | ⭐ | ✅ |
| Coder类 | ⭐ | ⭐ | ✅ | ✅ |

**图例：** ✅ 核心能力 | ⭐ 辅助能力

## 🔗 协作模式

### 三类代理协作模式

#### 研究→写作→代码 完整流程
```
Research: literature → knowledge-graph → gap-analysis → hypothesis
    ↓
Writer: intro-cluster → method-cluster → results-cluster → discussion-cluster
    ↓
Coder: industrial-ai → reviewer → debugger
```

#### 并行协作
```
Research: academic ‖ literature ‖ knowledge-graph ‖ trends
Writer: D1 ‖ D2 ‖ D3 ‖ D4 ‖ D5
Coder: industrial-ai ‖ reviewer ‖ debugger
```

#### 质量保证循环
```
Draft → Code Review → Quality Control → Style Format → Cache Optimize → Refine
```

---

## 🚀 三类协作工作流示例

### 完整AI研究项目流程
```bash
# 1. Research阶段：发现与分析
/agent research-literature: "搜索强化学习最新进展"
/agent research-knowledge-graph: "构建RL研究网络"
/agent research-gap-identifier: "识别RL研究空白"

# 2. Writer阶段：论文撰写
/agent writer-intro-cluster --task background: "撰写RL背景"
/agent writer-method-cluster --task algorithm: "描述算法细节"
/agent writer-results-cluster --task experiment: "设计实验方案"

# 3. Coder阶段：实现与优化
/agent coder-industrial-ai: "用JAX实现高性能RL算法"
/agent coder-reviewer: "审查JAX实现代码"
/agent coder-debugger: "调试性能问题"

# 4. 质量保证与发布
/agent writer-quality-controller: "执行四重质量验证"
/agent writer-style-formatter: "应用Nature格式"
```

---

**总计：** 18个专业代理 | Research(7) + Writer(8) + Coder(3)
**目标：** 从研究发现到工业部署的完整AI研究链条 | Nature级论文质量 | 25倍效率提升