# 7个 Paper 公共代码上收与统一规划（2024-11-26）

> 目标：将目前散落在各 `Paper/*` 子项目中的**通用代码**逐步上收/统一到父项目（根仓库），形成可复用的基础设施层，减少重复实现和维护成本。  
> 范围：数据加载、可解释性核心、算子/模块实现、训练与评估工具、LLM 集成等与具体论文弱耦合的代码。

---

## 一、现状与原则

### 1.1 当前常见问题（来自 7 个 proposal 与 README）

- 不同 Paper 子目录中逐渐出现：  
  - 自己的 `datasets` / `data_utils`；  
  - 自己的 explainers / metrics / visualization；  
  - 自己的算子实现（如 MoE 专家、Operator Attention 层、1D-2D 对齐模块等）；  
  - 各自的训练脚本与评估脚本。  
- 根仓库已经存在的通用模块：  
  - `data/vbench_dataset.py`, `data/vbench_utils.py`；  
  - `explainability/`（与 Toolkit 内 `toolkit_integration/explainability` 结构高度对应）；  
  - `model/` 下的信号处理、特征提取等基础组件；  
  - `trainer/` 中统一训练循环与配置。

### 1.2 上收与统一的基本原则

1. **能复用就不复制**：  
   - 底层通用功能（数据加载、可解释性核心、算子模块等）优先放在根仓库的公共目录中。  
2. **方法逻辑留在 Paper，基础设施回归父项目**：  
   - 例如 MoE 的“物理专家路由思想”留在 `Paper/MOE_explainable`，但专家算子实现可抽象为 `model/experts/` 模块。  
3. **先约定接口，再迁移代码**：  
   - 在公共模块中先定义清晰接口/抽象类，然后逐步把 Paper 子项目中的重复实现迁移过来。  
4. **优先不破坏现有功能**：  
   - 迁移过程中先做“上收 + 复用”，暂不彻底删除旧代码，仅在适当时机标记为 deprecated。

---

## 二、统一目标：公共模块的目标形态

### 2.1 数据层（Data Layer）

- **目标**：所有新实验优先通过 `data/vbench_dataset.py` + `data/vbench_utils.py` 访问数据。  
- 统一职责：  
  - 数据集加载与切分（train/val/test、few-shot 等）；  
  - 常用预处理（归一化、切片、增强等）；  
  - 统一 `DatasetTask`/`SignalData` 结构（可与 Explainable_FD_Toolkit 中的 `SignalData` 对齐）。

### 2.2 可解释性层（Explainability Layer）

- **目标**：根目录 `explainability/` 成为所有可解释性功能的唯一底层实现，Toolkit 与各 Paper 通过此模块访问解释能力。  
- 统一职责：  
  - `core/`：解释器抽象、解释结果数据结构；  
  - `methods/`：本征/事后解释方法集合；  
  - `knowledge/`：知识图谱、术语映射；  
  - `llm/`：LLM 接口与提示管理（底层共用，Paper 级包装各自实现）。

### 2.3 模型与算子层（Model & Operators）

- **目标**：将“可复用”的结构算子与模块集中到 `model/` 下：  
  - 信号处理算子（已在 `Signal_processing.py` 中）；  
  - MoE 专家模块与路由器骨架（可新建 `model/moe_experts.py` 等）；  
  - Operator Attention 层（可新建 `model/operator_attention.py`）；  
  - 1D-2D 融合基础模块（可新建 `model/multimodal_fusion.py`）。

### 2.4 训练与评估工具层（Trainer & Eval）

- **目标**：统一使用 `trainer/` 下的训练 loop 与配置系统，各 Paper 不再在子目录重复实现完整训练器。  
- 统一职责：  
  - 标准训练/验证/测试循环；  
  - 通用早停、日志记录、模型保存；  
  - 统一评估指标与日志输出格式。

### 2.5 LLM 与交互层（LLM & Interface）

- **目标**：在根仓库维护统一的 LLM Provider 抽象与基础工具，由 LLM_Explainable_FD_Toolkit 在其上做专业封装。  
- 统一职责：  
  - LLM Provider 接口（OpenAI、本地模型等）；  
  - Prompt 模板管理、输出解析基础组件；  
  - 基础安全/鲁棒性处理（超时、重试、简单输出检查）。

---

## 三、分阶段执行计划（仓库级）

> 时间尺度参考前面“Week 1–8” 总体计划，可按实际进度调整。

### 阶段 1：清点与标记（Week 1）

**目标**：弄清楚 7 个 Paper 中已经开始“局部实现”的公共功能，标记潜在上收对象。

- [ ] 在 `docs/11_26` 下新增一个清单文件（例如 `COMMON_COMPONENTS_INVENTORY.md`），记录：  
  - 每个 Paper 子目录中出现的：  
    - 数据加载/预处理代码；  
    - explainers/metrics/visualization；  
    - 算子/模块（MoE 专家、算子注意力、1D-2D 对齐等）；  
    - 训练/评估脚本。  
- [ ] 用简单表格标记：  
  - 是否已经在根仓库有类似功能；  
  - 是否适合上收为公共组件；  
  - 当前使用范围（仅本 Paper / 多个 Paper）。

产出：  
- 1 份公共组件清单，为后续迁移提供依据。

### 阶段 2：确定“公共模块归属地”（Week 1–2）

**目标**：将清单中的上收对象映射到根仓库中的目标模块位置。

- [ ] 针对每类组件（数据、解释、算子、训练、LLM），在本计划中补充“目标模块位置”字段：  
  - 例如：  
    - `Paper/MOE_explainable` 中的 `experts/` → 上收部分到 `model/moe_experts.py`；  
    - `Paper/Explainable_FD_Toolkit` 中的 explainability 子模块 → 对齐并复用根目录 `explainability/`；  
    - `Paper/LLM_Explainable_FD_Toolkit` 中的 LLM 通用工具 → 上收到根级 LLM Provider 抽象。
- [ ] 在各项目的 plan 文件中增加简短备注：  
  - “XXX 模块的底层实现应放在父项目 YYY 处，仅在本项目做调用和轻量包装。”

产出：  
- 一份“组件→根模块”的映射关系草案，达成 conceptual agreement。

### 阶段 3：自上而下定义统一接口（Week 2–3）

**目标**：先在根仓库创建/整理统一接口，再让各 Paper 按接口进行改造。

- [ ] 在根仓库中完成/完善以下接口：  
  - `data/vbench_dataset.py` + `vbench_utils.py`：统一数据加载与切分接口；  
  - `explainability/core` 中的解释器/解释结果接口（与 Toolkit proposal 中定义的接口对齐）；  
  - `model/` 中的可复用模块接口（如 MoE 专家、Operator Attention 层、1D-2D 融合层的抽象类或基础实现）；  
  - LLM Provider 抽象接口（若需要，可放在 `explainability/llm` 或 `utils/llm`）。
- [ ] 在 `docs/11_26` 下撰写一份“统一接口规范”说明：  
  - 强调新代码/新项目应优先依赖这些接口，而非在 Paper 子目录自己再造一份。

产出：  
- 根仓库统一接口基本就绪，可以作为各项目的“上游依赖”。

### 阶段 4：Paper 子项目逐步改造为“轻量包装层”（Week 3–6）

**目标**：让各 Paper 子项目只保留“方法/论文相关逻辑”，将基础设施依赖迁移到根模块。

- [ ] 依次处理 7 个 Paper：  
  - 在其 plan 中已有“阶段 1 Demo + 阶段 2 集成”任务的基础上：  
    - 优先修改 import，使其从根仓库公共模块导入（如 `from data.vbench_dataset import ...`、`from explainability.core import ...`）；  
    - 把可以迁移的工具代码从 Paper 子目录移动/重写到根模块中。  
- [ ] 为每一次改造添加简单的回归测试（可以是最小 demo 脚本），确保迁移不破坏现有功能。

产出：  
- 各 Paper 子目录中减少重复基础设施代码，更多只是对公共模块的调用与配置。

### 阶段 5：清理遗留与文档同步（Week 6–8）

**目标**：清理已废弃的旧实现，更新文档，让所有人都知道“新规范”。

- [ ] 为已迁移到根模块的旧代码添加明确的 deprecation 标记，并在合适时候删除或重定向。  
- [ ] 更新以下文档：  
  - 根目录 `readme.md` 中的数据/解释/训练模块介绍；  
  - `Paper/doc/README_11_25.md` 中关于基础设施层的描述；  
  - 各 Paper 的 README 与 proposal，确保不再推荐使用旧路径/旧模块。  
- [ ] 在 `docs` 中新增一份简短“开发者指南”，专门给后来者说明：  
  - 代码应该放在哪一层；  
  - Paper 子目录下应避免包含哪些类型的“公共代码”。

产出：  
- 一个结构清晰、职责分明的仓库布局；  
- 文档与实际代码状态保持一致。

---

## 四、与 7 个子项目阶段计划的关系

- Explainable_FD_Toolkit 🟢：  
  - 将作为可解释性基础设施的集中入口，其自身也应尽量重用 `explainability/` 根模块。  
- Neuralsymbolic-XFD 🟦：  
  - 理论层可在文档中正式引入“基础设施层 vs 方法层 vs 应用层”的概念，巩固这一上收策略。  
- Operator Attention 🔴 / MoE 🟠 / 1D-2D 📘 / Fuzzy 🩷：  
  - 在模型/算子层上收部分通用模块，Paper 子目录只保留特定实验配置与论文逻辑。  
- LLM Interface 🟣：  
  - 依托统一的 LLM Provider 与解释中间表示接口，减少在各处重复实现 LLM 相关工具。

---

## 五、后续 Codex/Agent 使用建议

1. 若任务是“开发新基础设施” → 优先修改根目录（`data/`, `explainability/`, `model/`, `trainer/`），并在本计划中登记。  
2. 若任务是“实现某篇 Paper 的方法” → 先查本计划与统一接口规范，尽量只在 `Paper/*` 目录中写方法特有逻辑与配置。  
3. 若发现跨多个 Paper 重复的代码 → 先在 `docs/11_26/COMMON_COMPONENTS_INVENTORY.md` 记录，再提议/执行上收。  

