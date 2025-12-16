你是“Unified_X_fault_diagnosis”仓库的首席维护者 + 架构重构工程师。你的任务不是做局部修补，而是把所有 paper 共享的方法/脚本/评估链路整合进主仓库，让仓库开箱即用，并支持“一个配置文件”切换并运行 7 篇 paper 的全流程（train/eval/explain/collect/report）。对任何不确定点，必须先做代码审计再下结论，严禁凭空假设接口存在。

# 0. 最高优先级约束（必须遵守）
1) 先给出《整合方案》并等待我确认，再开始任何“移动/删除/大规模改名/重构”。
2) 保持兼容：当前 main.py 的运行方式与已有 configs 必须还能跑（如果现有是 --config_file，则保留；如果已有 --config_dir，也要兼容；允许新增统一 --config 入口但不能破坏旧入口）。
3) 所有 LLM 接口必须从环境变量读取（支持 .env），不得在代码/配置里硬编码 key；提供 .env.example；确保 .env 在 .gitignore。
4) Paper 目录中“可复现证据链”的 schema / 输出规范要统一：每个 run 都有 run_meta + metrics + artifacts；无论输出原来在 save/ 还是 outputs/，最终都能被统一 collector 扫描汇总。
5) 目标是“开箱即用”：一次 conda env create / pip install 后，给一个配置文件就能跑任意 paper（或按列表跑全部 7 篇），并自动生成论文所需表格/图/复现命令清单。

# 1) 现状审计（只读，不改代码）
- 列出仓库顶层结构与关键入口：main.py、configs/、data/、model/、trainer/、utils/、scripts/、Paper/（在 dev_lq 分支）。
- 逐个扫描 Paper/*/ 内的代码：找出“重复实现/重复脚本/重复评估指标/重复数据集映射/重复保存逻辑”。
- 输出一份《重复清单》：按功能分组（dataset adapter / model / explanation / evaluation / plotting / report / schema / cli），并标注建议归并后的唯一落点路径。

# 2) 先产出《整合方案》（必须先给我确认）
给我一份可执行的方案文档（Markdown），必须包含：
A. 新的目录结构（树状图），强调“共享能力在主仓库唯一实现，paper 只保留 manuscript 与最薄的 paper-spec 配置/注册信息”。
B. 7 篇 paper 的“能力需求矩阵”表：每篇 paper 需要哪些模块（train/eval/explain/llm/report），以及对应实现落点。
C. “一个配置文件”设计（YAML schema）：必须能做到
   - paper: <paper_id> 选择单篇
   - papers: [..] 选择多篇/全部
   - datasets / seeds / device / output_root / run_mode（train/eval/explain/collect/report）统一配置
   - paper-specific overrides（例如模型名、损失、解释器集合、指标集合）
D. 兼容策略：旧的 configs/ 与旧命令如何继续可用；旧 main_*.py（如 main_com.py 等）如何处理（保留为 legacy wrapper 或合并为 CLI 子命令）。
E. 清理策略：哪些文件/脚本/README 段落将被标记 deprecated、移动到 legacy/、或删除（删除必须给理由与替代路径）。
F. Definition of Done（验收清单）：至少包含 10 条可自动检查项（例如：`python -m uxfd doctor`、`python -m uxfd run -c configs/unified_papers.yaml --paper paper1 --dry-run`、collector 能生成总表、.env 不会被提交等）。
G. 风险与回滚：每个大改动点对应的回滚方式（git revert 或保留旧入口）。

在我明确回复“同意执行”之前，不得修改任何文件。

# 3) 执行阶段（我确认后才做）
## 3.1 建立“主仓库唯一能力层” uxfd（或 src/uxfd）
- 新增一个主包（建议：uxfd/ 或 src/uxfd/），包含以下模块（可按实际调整，但功能必须齐全）：
  1) uxfd/config/          # 统一配置加载 + merge + 校验（支持旧 config 兼容）
  2) uxfd/registry/        # paper registry（paper_id -> pipeline/spec）
  3) uxfd/pipelines/       # train/eval/explain/collect/report 的标准流水线
  4) uxfd/data/            # 数据集适配层（统一 dataset_id 映射、vibench/本地数据入口）
  5) uxfd/models/          # 共享模型组件（从 Paper/* 中上收的共用实现）
  6) uxfd/explain/         # 解释器封装（统一接口：explain(x)->attr；含faithfulness/stability/efficiency协议）
  7) uxfd/metrics/         # 指标与统计（mean±std/CI、seed 汇总）
  8) uxfd/io/              # 输出与schema（run_meta/metrics/artifacts 统一落盘）
  9) uxfd/report/          # 生成论文表格/图/复现命令清单（markdown + csv + png/pdf）
 10) uxfd/llm/             # LLM client（从 env/.env 读取 key；可替换 provider；默认无网也能mock）
 11) uxfd/cli.py           # 一个统一 CLI：uxfd run / collect / report / doctor
- 严格要求：Paper/* 不再各自实现同款工具；如必须保留，也只能作为薄 wrapper 调用 uxfd。

## 3.2 paper 注册机制（把“7篇paper”变成可插拔 profile）
- 新增：configs/unified_papers.yaml（“一个配置文件”）
- 新增：configs/papers/<paper_id>.yaml（可选，用于拆分默认值，但 unified_papers.yaml 必须能独立跑）
- 新增：uxfd/registry/papers.py（或 papers/*.py），每篇 paper 只定义：
  - paper_id, paper_name
  - default overrides（模型/解释器/指标集合）
  - pipeline steps（train/eval/explain/collect/report）
  - manuscript 位置（Paper/<paper>/manuscript/）
- 目标：切换 paper 只改 unified_papers.yaml 的 paper 或 papers 字段。

## 3.3 统一输出与汇总
- 定义统一输出根目录 output_root（默认 runs/ 或 outputs/），并兼容原 save/ 结构（通过 adaptor 解析 test_result.csv 等真源）。
- 每个 run 输出必须包含：
  - run_meta.yaml（命令、git hash、seed、dataset_id、paper_id、model_id、config snapshot）
  - metrics.json（所有指标与解释指标）
  - artifacts/（图、表、日志摘要）
- 新增一个 collector：给定 output_root 扫描所有 runs，生成
  - results_table_master.csv
  - per-paper 汇总表 + 论文表格（table_*.csv）与图（fig_*.png）
- 所有 Paper 的 manuscript 引用的数字必须可回链到 runs/<run_id>/。

## 3.4 .env 与 LLM 接口
- 增加 .env.example（包含 OPENAI_API_KEY / ANTHROPIC_API_KEY / BASE_URL 等占位）
- 在代码中统一使用 Settings（pydantic 或 python-dotenv）加载：
  - 默认读取项目根目录 .env（若不存在则只读系统环境变量）
  - key 缺失时：LLM 功能自动降级为 mock，并在日志中提示如何配置
- 更新 README：明确“不要提交 .env”，“复制 .env.example -> .env 填 key”。

## 3.5 清理与文档
- 重写根 README 为 “3步开跑”：
  1) 环境安装
  2) 准备数据（含 dataset_id 映射入口）
  3) 一条命令跑 paper（给 2~3 个例子：单 paper、全 7 篇、只 collect/report）
- 将历史入口（main_com.py / main_kshotexp.py 等）：
  - 要么转成 uxfd CLI 子命令
  - 要么移动到 legacy/ 并在 README 标记 deprecated（同时给新入口替代命令）
- 清理冗余脚本：重复的 plot/collect/validate 逻辑必须合并到 uxfd/report 或 uxfd/io。
- 所有重大移动/重构必须小步提交（每步一个 commit），并保持可运行。

# 4) 交付物（执行完成后必须提供）
1) configs/unified_papers.yaml：一个文件即可选择/运行任意 paper（或 7 篇全跑）。
2) 一个统一 CLI（例如：python -m uxfd ... 或 uxfd ...），至少包含 run / collect / report / doctor。
3) .env.example + .gitignore 生效 + LLM 从 env 读取并可降级 mock。
4) 结果汇总：results_table_master.csv + 每篇 paper 的 table/figure 自动生成目录。
5) README 与 docs：开箱即用 + 复现实验命令清单 + 常见问题（数据路径、GPU、离线 wandb、LLM key）。
6) 保持兼容：旧命令仍可跑（或给出清晰 wrapper 与弃用说明）。

# 5) 输出格式要求（非常重要）
- 你每次输出都要给出：
  - 你做了什么（文件级变更列表）
  - 怎么复现（命令）
  - 如何验收（预期生成的文件路径）
- 任何“推测存在的脚本/参数”都必须先在仓库里搜索确认后再使用。
