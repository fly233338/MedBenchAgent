# MedBenchAgent 端到端自主研究产品与实施路线图

状态：提案

日期：2026-08-19

适用范围：从数据集和医学资料发现、摄取、规范化、研究到 benchmark 构建、审计和交付的完整流程

## 1. 执行结论

MedBenchAgent 不应只把现有 `CapabilityAgent` 和 `ClinicalDesignAgent` 改造成 ReAct。更有产品价值的
方向，是把当前“用户预先准备好规范数据集和 PDF 后才能运行”的 benchmark builder，扩展为一个
**Medical VLM Benchmark Research Workspace**：用户只需要给出 benchmark 目标和资源策略，系统可以
在权限和预算内发现候选数据集、筛选和下载资源、把原始标注规范化为项目的 annotation v2、收集医学
资料、执行 Agentic RAG、生成可审核的 benchmark 方案，最后复用现有确定性生成和 Audit 交付结果。

目标闭环是：

```text
自然语言目标
  → 资源规划
  → 数据集发现 / 筛选 / 获取
  → annotation v2 与图像资产规范化
  → 数据集能力研究
  → 医学资料发现 / 筛选 / 获取
  → Agentic RAG 与题型设计
  → 确定性题目生成
  → 硬审计与交付
```

这不是“所有步骤都交给 Agent”。可落地的职责划分是：

- Agent 负责开放且需要观察反馈的决策：搜索词、候选资源比较、字段和 join 语义、检索纠正、证据
  充分性、何时请求人工判断；
- 受控代码负责下载、解压、hash、文件清单和注册等副作用；确定性代码负责行级转换、图像转换、
  schema validation、答案构造、provenance 和 Audit；
- policy engine 决定哪些下载或转换可以自动执行，哪些必须先取得用户批准；
- 人工确认保留在许可/体量、不可逆的映射歧义、医学图像编码和最终方案等高风险边界。

ReAct 的引入必须对应实际功能。本计划认可四个 Agent 决策区：

1. **Dataset Discovery**：根据搜索结果和 dataset card 继续搜索、比较或停止；
2. **Dataset Onboarding**：根据 source inventory 和 join/cardinality observation 修订声明式映射；
3. **Capability Research**：按需深查字段语义和值分布，再决定任务是否可支持；
4. **Agentic RAG**：根据 passage 相关性和 claim coverage 改写查询、切换检索策略、补充资料或声明证据
   不足。

下载、转换、注册、样本分配、逐题生成和硬审计不是 ReAct 场景。模型可以提议或触发一个已批准的
确定性 job，但不能自行实现这些操作。

## 2. 当前不优雅之处与改造必要性

| 当前情况 | 实际影响 | 是否有必要改 |
|---|---|---|
| `dataset_catalog.json` 只列人工准备好的 `annotation.jsonl` | 新数据集发现、下载、处理和注册完全在系统外 | 必须 |
| CSV/XLSX/JSON 原始标注不能直接进入项目 | 每个数据集依赖临时人工清洗，难以复现 | 必须 |
| dataset catalog 只保存路径和描述 | 缺少来源、许可、版本、下载 hash、转换计划和图像资产的统一 provenance | 必须 |
| 旧 `normalize-medical-image-dataset` skill 仍使用 annotation v1 | 与当前代码的四实体 annotation v2 构成双真相源 | 必须首先修复 |
| 当前资料选择只扫描用户预放入 `materials/` 的 PDF | 系统不能发现缺失指南、判断版本或补充资料 | 必须 |
| `LocalPdfRag` 对每个任务执行一次固定检索 | 低相关结果不会触发查询改写、检索策略切换或 corpus gap 处理 | 必须 |
| 检索、设计生成和引用验证耦合在 `design_task` | 难以独立评测 retriever、evidence grader 和 designer | 必须 |
| `complete_json()` 是唯一模型接口 | 无法稳定处理多轮 tool call/result | 必须 |
| CLI 使用进程内 `MemorySaver` | 长下载、索引、转换和研究任务退出后不能恢复 | 必须 |
| Blueprint 在部分流程中被逐步直接写入 | Agent 化后会放大半成品和重放风险 | 应改为 candidate artifact 后原子 apply |
| 五个角色名称存在，但根图完全固定 | 对确定性依赖是优点；对资源缺失分支不够 | 只增加代码控制的资源路由，不改成自由 supervisor |

这里最重要的清理不是增加 Agent 类，而是建立一条正式的 **resource plane**：所有外部数据集和资料都
先成为有状态、有 provenance、可验证、可恢复的资源 job，准备完成后才进入现有 benchmark control
plane。

### 2.1 扩展优先级

| 优先级 | 能力 | 原因 |
|---|---|---|
| 必须先做 | annotation v2 单一真相源、durable jobs、resource manifests、tool-call runtime | 其他自治能力的可靠性基础 |
| 必须做 | Dataset Discovery/Acquisition、Tier A/B Onboarding、catalog v2 | 直接解决数据必须由用户提前准备的问题 |
| 必须做 | Reference Curation、corpus version、corrective Agentic RAG、evidence ledger | 直接解决资料缺失和固定单次检索的问题 |
| 应做 | Capability 按需字段研究、validated memory、resource 快速复用 | 能减少错误任务和重复工作，但必须用 eval 证明收益 |
| 条件式 | 新医学数据源 connector、hybrid retrieval、并行 task workers | 只在现有来源/检索/吞吐确有瓶颈时引入 |
| 暂缓 | 自由 supervisor、角色型多 Agent、任意 Web/代码执行、自动生成并执行 custom adapter | 风险和复杂度高，当前没有必要 |

## 3. 参考项目、标准和研究经验

本计划只迁移与 Medical VLM benchmark 生产问题相符的模式。

| 来源 | 可迁移经验 | 本项目决策 |
|---|---|---|
| [ReAct](https://arxiv.org/abs/2210.03629) | 推理、行动、观察交替，外部 observation 可以修正计划 | 用于资源搜索、映射核查和纠正检索，不用于确定性转换 |
| [Open Deep Research](https://github.com/langchain-ai/open_deep_research) | research brief、显式完成工具、迭代和并发预算 | 每个研究 loop 有 brief、工具集、完成动作和硬预算 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | planner、information gathering、publisher 分离，来源贯穿全流程 | 分离资源研究、证据研究和 benchmark 构建，不让生成阶段重新找来源 |
| [STORM / Co-STORM](https://github.com/stanford-oval/storm) | 先做知识整理和提纲，再生成；人可以中途调整研究方向 | 先形成 Evidence-backed Proposal，再生成题目 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 检索不是固定必选动作，需要判断 passage relevance 和输出支持度 | 采用按需检索和证据判断思想，不训练 reflection-token 模型 |
| [Corrective RAG](https://arxiv.org/abs/2401.15884) | 检索质量差时评估并触发纠正动作 | 低相关、冲突或 coverage 不足时改写查询、换策略或补 corpus |
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 按问题复杂度选择简单或多步检索 | 直接 task context 可单次检索，复杂/失败 task 才进入多轮循环 |
| [RAGChecker](https://arxiv.org/abs/2408.08067) | retriever 与 generator 应分别诊断，不能只看最终答案 | 分开报告 retrieval recall/precision、claim support 和 design validity |
| [MLCommons Croissant](https://mlcommons.org/working-groups/data/croissant/) | 数据集元数据覆盖 provenance、license、resources、structure 和 ML semantics | 能读取 Croissant 时复用元数据；内部仍使用更小的 DatasetCandidate contract |
| [Hugging Face Dataset Cards](https://huggingface.co/docs/hub/main/datasets-cards) | license、语言、体量、任务和限制是发现与负责使用的基础 | dataset card 是候选证据，不完整字段必须标记 unknown |
| [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | 优先简单组合，只在可测收益时增加自治 | 固定外层、局部 Agent；每阶段有停止实施门 |
| [Anthropic: writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | 工具少而明确、按资源命名、响应高信号、通过 eval 优化 | 每个 Agent 只动态加载本阶段的 3–5 个工具 |
| [Anthropic: multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system) | 多 Agent 适合可并行、高价值、超出单上下文的任务，但显著增加 token 和协调成本 | 首版不用角色型多 Agent；成熟后只并行独立 task jobs |
| [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) 与 [vLLM Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/) | 模型能力不等于本地 serving 已正确配置 tool parser、template 和多轮消息 | Phase 0 实测本地 endpoint，失败即阻断 Agent 实施 |

## 4. 可行性反思与架构决策

### 4.1 Agent 能否把任意下载数据集自动变成 annotation v2

**初步目标**：下载任意医学影像数据集后，全自动生成合法 `annotation.jsonl`。

**反思**：任意数据集可能包含多表多对多 join、不明 entity scope、病灶级重复行、非标准 bbox、mask、
压缩或加密图像、DICOM series、WSI、许可限制和文档缺失。模型无法从列名或数值可靠推断这些医学与
数据语义。承诺“任意数据集全自动”不可行，也不安全。

**决策**：采用三级摄取能力，而不是任意代码生成。

| 层级 | 支持范围 | 自治程度 |
|---|---|---|
| A：已知 adapter | 已有测试的 dataset/version adapter | policy 允许时全自动下载、转换、验证和注册 |
| B：声明式 mapping | 常见 CSV/TSV/XLSX/JSON/Parquet，join 和 grouping 可由 mapping contract 表达 | Agent 提出映射，代码验证，用户首次确认后由通用 engine 执行 |
| C：custom adapter required | mapping contract 无法表达、文档/坐标/实体语义不明确、复杂医学格式 | Agent 输出 inventory 和阻断原因，由工程师实现并审核 adapter |

首版产品完成 A 和 B。C 不是失败隐藏项，而是明确的 `adapter_required` 终止状态。Agent 不生成并直接
执行任意 Python。后续可以让 Agent 起草 adapter patch，但仍需代码审查、测试和独立批准后才能运行。

### 4.2 数据集能否自主搜索和下载

**可行部分**：Agent 可以根据 BenchmarkInfo 生成查询，跨 ModelScope、OpenDataLab/OpenXLab、Hugging
Face 搜索候选，读取 dataset card/metadata，比较模态、疾病、annotation 类型、许可、版本和体量，并
继续细化查询。给定 URL 或 repository ID 时可以直接识别来源。

**不能省略的边界**：下载可能消耗大量磁盘、要求登录/同意协议、包含未知许可或受控医疗数据。模型
不能代表用户接受条款、绕过访问控制或决定 PHI 使用权限。

**决策**：Agent 自主提出并执行下载，但必须经过 acquisition policy：

- `manual`：每次下载都确认；
- `open_under_limit`：仅对许可 allowlist、无需登录、大小低于配置阈值的资源自动下载；
- gated、unknown license、unknown size、付费或需要同意条款一律中断；
- 用户只批准 `AcquisitionPlan`，下载工具接收代码生成的 scoped approval token；
- credentials 由运行时注入，不进入模型 prompt、tool arguments 或 artifact。

名称搜索按本地环境约定优先 ModelScope，其次 OpenDataLab/OpenXLab，再到 Hugging Face。医学专用来源
如 TCIA 或 Grand Challenge 在有实际数据需求和稳定 API 后作为独立 connector 增加，不以通用浏览器
下载代替 connector。

### 4.3 annotation schema 是否有唯一真相源

当前 `src/medbenchagent/schemas/annotation.v2.schema.json` 要求 `patient`、`case`、`study`、`image` 四个
实体；较新的 `prepare-medical-annotations` skill 也遵循 v2。旧
`normalize-medical-image-dataset/schema.json` 仍是六个顶层字段的 v1。

**决策**：实现任何自动摄取前必须消除 schema drift：

- `src/medbenchagent/schemas/annotation.v2.schema.json` 是唯一运行时事实源；
- 旧 v1 normalization skill 标记 deprecated 或升级到 v2；
- skills、converter、validator 和测试不再维护独立手写 schema 副本，必须引用或校验同一 hash；
- `dataset_bundle.json` 记录 annotation schema ID/hash 和 converter version；
- 未通过 v2 schema、join reconciliation 和 determinism check 的数据集不能注册为 available。

### 4.4 行级转换应该由 Agent 还是代码完成

**反思**：模型逐行转换不可复现、昂贵，也容易添加源数据没有的医学解释。现有数据准备 skill 已明确
要求 deterministic converter。

**决策**：Agent 只读取 bounded inventory 和文档摘要，输出 `ConversionPlanV2`；通用 mapping engine
根据已批准 plan 生成字节稳定的 annotation v2。重复运行必须 byte-identical。原始表、完整记录、患者
标识符和图像不会发送给外部模型。

### 4.5 Agentic RAG 是否只是给检索加 ReAct

**反思**：如果仍然只有单一 vector query，只是让模型多调用几次 `retrieve`，系统并没有解决索引、
来源质量、相关性判断和 claim grounding 问题。

**决策**：Agentic RAG 必须包含两层：

1. **Reference Curation Agent**：发现、筛选、版本核验、下载和建立可检索 corpus；
2. **Task Evidence Agent**：query planning → retrieval → evidence grading → corrective action → claim ledger →
   validated design。

检索层支持 metadata filter、lexical、semantic 和 hybrid strategy，再经过 reranker。Agent 可以选择或
改写 query，但不能修改索引分数。Evidence grader 输出结构化相关性和覆盖判断；最终 citation validity
由代码根据 chunk ID、PDF hash 和页码回放。

本项目不复制 Self-RAG 的模型训练，也不把模型自评当作事实。采用的是 adaptive/corrective control
flow，并用独立 eval 校验 retriever 和 claim support。

### 4.6 Agentic RAG 是否可以在 corpus 缺失时自动上网

**决策**：Task Evidence Agent 不直接开放任意 Web。它可以返回 `corpus_gap`，由 ResourceCoordinator
调用 Reference Curation Agent，在政府、WHO、专业学会、PubMed/PMC 和用户 allowlist 中搜索。资料经过
版本筛选、下载 hash、文本可提取性和人工/policy 准入后进入新 corpus version，然后原 task 重新检索。

Web 内容里的指令一律是不可信数据。不可下载或受限来源保留 landing link 和原因，不能伪造本地引用。

### 4.7 是否需要自由 supervisor 或多 Agent

资源是否就绪产生了真实分支，但不需要让一个 LLM 自由跳过安全阶段。

**决策**：根图增加代码控制的 `ResourceCoordinator`：

- catalog 有合适且 ready 的数据集时走现有选择；
- 没有合适数据集或用户要求新数据时进入 Dataset Discovery/Onboarding；
- materials corpus 满足任务 evidence policy 时直接进入 Task Evidence Agent；
- corpus gap 时进入 Reference Curation；
- Generation、Audit 和 finalize 永远不能被模型跳过。

首版每个 research job 是单 Agent。未来并行只使用同一 Evidence subgraph 按独立 task map 执行，不先
引入不同人格的角色群。多 Agent 仍受本文件后续量化门控制。

### 4.8 记忆是否等于保存所有研究消息

**决策**：分离 checkpoint、job artifact、validated memory 和 immutable benchmark artifacts。不保存
raw chain-of-thought；未确认的 dataset interpretation、mapping 或医学 claim 不进入长期记忆。成功的
mapping plan 可以在 source revision/hash 相同时复用，但必须重新运行 deterministic validation。

## 5. 最终产品形态

### 5.1 产品定位

产品名称暂定 **MedBench Research Workspace**。用户可以从三种入口开始：

1. 使用已注册 dataset 和 materials；
2. 提供本地原始数据集，由系统 inventory、映射、规范化和注册；
3. 只描述 benchmark 目标，由系统搜索并提出数据集和医学资料获取方案。

所有用户交互继续只接收自然语言。模型把 resource policy、acquisition approval、mapping review、资料
选择和 proposal review 解析成受限 JSON action，代码再校验 candidate ID、approval scope 和允许修改项；
不能让自由文本直接触发下载、转换或注册。

首个可发布产品不追求任意疾病、任意格式的完全无人值守，而是让支持范围内的资源准备形成可重复、
可审核的自动化路径；遇到未知格式或语义歧义时产生高质量阻断 artifact，而不是回退到隐式猜测。

### 5.2 用户流程

```text
1. 用户描述 benchmark 目标和 resource policy
2. SpecificationAgent 提取 BenchmarkInfo，用户确认
3. ResourceCoordinator 检查现有资源
4. 如果需要新数据集：
   4.1 Dataset Discovery Agent 搜索和比较候选
   4.2 用户/policy 批准 AcquisitionPlan
   4.3 确定性 downloader 获取并生成 source manifest
   4.4 Dataset Onboarding Agent 读取 inventory，提出 ConversionPlanV2
   4.5 用户确认存在语义影响的映射和图像编码
   4.6 确定性 engine 生成 annotation v2、图像资产和 bundle manifest
   4.7 validator 通过后注册 catalog
5. Capability Research Agent 提出 Task Plan，用户确认
6. Reference Curation Agent 检查/补充医学资料，用户或 policy 确认来源范围
7. Task Evidence Agent 执行 Agentic RAG，生成 Evidence-backed Design Proposal
8. 用户确认或定向修改设计
9. GenerationAgent 确定性生成题目
10. AuditAgent 审计并交付 benchmark、manifest 和研究报告
```

### 5.3 目标架构

```text
                         ┌──────────────── Resource Plane ────────────────┐
BenchmarkInfo ──────────→│ ResourceCoordinator                            │
                         │   ├─ Dataset Discovery Agent ↔ source tools     │
                         │   ├─ Acquisition Job → source manifest          │
                         │   ├─ Dataset Onboarding Agent ↔ inventory tools │
                         │   ├─ Deterministic Normalization/Image Jobs     │
                         │   └─ Reference Curation Agent ↔ source tools    │
                         └────────────────────┬────────────────────────────┘
                                              ↓ ready resource manifests
                         ┌──────────────── Benchmark Control Plane ───────┐
                         │ Capability Research Agent                      │
                         │   ↓ confirmed Task Plan                        │
                         │ Task Evidence Agent ↔ Agentic RAG tools        │
                         │   ↓ confirmed Evidence-backed Design           │
                         │ GenerationAgent → AuditAgent → finalize        │
                         └─────────────────────────────────────────────────┘
```

Resource Plane 的所有大文件和转换结果只以 manifest/path/hash 跨入 control plane。模型上下文不传原始数据
行、完整 archive、图像像素或任意本地路径。

### 5.4 运行模式

- `workflow`：只使用已准备资源和当前固定 RAG，作为基线和回滚；
- `research_local`：允许本地原始 dataset onboarding 和 Agentic RAG，不访问网络；
- `research_connected`：按 connector、许可、磁盘和域名 policy 进行资源发现与获取。

模式、policy、connector 版本、toolset version 和所有预算写入 `effective_config.json` 和最终 manifest。
不能由模型在运行中隐式提高自治等级。

## 6. Resource Plane 设计

### 6.1 DatasetCandidate

不同来源的 metadata 先标准化为内部 candidate，不把 provider 原始响应直接交给模型：

```json
{
  "candidate_id": "provider:owner/repo@revision",
  "provider": "modelscope | opendatalab | huggingface | local",
  "repository_id": "owner/repo",
  "revision": "immutable revision or null",
  "name": "dataset name",
  "description": "bounded description",
  "modalities": [],
  "anatomies": [],
  "diseases": [],
  "annotation_types": [],
  "license": "SPDX/URL/unknown",
  "access": "open | gated | restricted | unknown",
  "size_bytes": null,
  "source_url": "https://...",
  "documentation_urls": [],
  "metadata_provenance": []
}
```

Dataset card、Croissant 或 provider metadata 只支持它明确声明的字段。Agent 不得从 repository 名称推断
许可、疾病、模态或 annotation 存在。

### 6.2 AcquisitionPlan 与下载 job

Agent 提交 `AcquisitionPlan`，代码检查：

- provider/repository/revision 是否来自当前 search result；
- license/access/size 是否满足 policy；
- 目标目录是否位于配置的 source root；
- 预计磁盘空间、archive 数量和认证要求；
- downloader 命令模板和 connector version；
- approval token 的 scope、过期时间和一次性语义。

下载优先级和命令实现遵循 `research-downloads` 的环境约定。每次 job 输出：

```text
<dataset_source_job>/
├── acquisition_plan.json
├── source_manifest.jsonl
├── source_metadata.json
├── downloads/
└── acquisition_report.json
```

`source_manifest.jsonl` 记录每个文件的 provider path、local relative path、size、sha256 和 fetch status。
下载到 staging/quarantine，安全检查 archive path traversal、symlink、重复路径和体量后再发布。失败重试以
revision 和 hash 幂等，不覆盖不同内容，也不自动删除用户数据。

### 6.3 SourceInventory 与 ConversionPlanV2

确定性 inventory job 读取 source manifest，输出：

- annotation tables/sheets/JSON 的行数、列、类型、null/unique counts；
- candidate image/patient/case/study identifiers；
- candidate join keys 及实测 cardinality；
- source-row granularity；
- 图像文件内容格式、尺寸、帧/volume/WSI 状态；
- annotation reference 与 image path 的 exact/missing/ambiguous matches；
- dataset documentation 和已提取的 mapping statements；
- 不含完整原始行或不必要 PHI 的 bounded examples。

Dataset Onboarding Agent 根据 inventory 和来源文档提出 `ConversionPlanV2`：

- authoritative image population 和稳定顺序；
- image/patient/case/study ID 与 `image.image_path` 规则；
- joins、cardinalities、grouping 和 entity scope；
- source column → annotation v2 field mappings；
- modality/anatomy provenance；
- bbox source convention 和精确转换公式；
- included/excluded fields、missing/orphan/duplicate/conflict policy；
- image preparation policy；
- expected input/output counts；
- unresolved decisions 和所需 approval。

Validator 不只检查 JSON 结构，还要把 plan 的 join 和 field references 对照 inventory。存在无法由来源文档
确定的 identity、entity scope、bbox 约定、mask 处理或 image encoding 时返回 `needs_human` 或
`adapter_required`。

### 6.4 通用 deterministic mapping engine

首版 engine 支持可以声明式表达的常见操作：

- 读取 CSV、TSV、XLSX、JSON/JSONL 和 Parquet；
- exact-key joins 和显式 cardinality 约束；
- 按 authoritative image group；
- 按 source row 保留重复 annotation object；
- 字段复制、JSON-compatible 类型解析和显式 null policy；
- 已文档化的 bbox corner、xywh、center-xywh 转换；
- 稳定排序、临时写入和原子发布。

它不支持 fuzzy join、从图像或文件名推断标签、医学术语归一化、自动合并类别、未知 mask-to-bbox、自动
选 volume slice 或 WSI region。无法表达时停止到 Tier C。

为了复现，每个 dataset bundle 仍保留一个 plan-hash 固定的 invocation script；脚本只调用通用 engine，
不包含模型临时生成的行级逻辑。

### 6.5 图像准备

图像 job 只处理 annotation 引用的文件：

- 已支持的安全 PNG/JPEG 可以按明确 policy 原样使用或复制，保留 bytes/hash；
- 单帧 DICOM、2D NIfTI/NRRD/MHA/MHD 等由受测试 converter 生成确定性 8-bit display PNG；
- multi-frame、true volume、WSI、pyramidal/multi-page 文件拒绝，不让 Agent 选 slice；
- 不 resize/crop/rotate/flip；任何编码选择必须来自用户批准或已知 adapter policy；
- annotation 与最终图像 manifest 必须一一对应并验证尺寸、hash 和非恒定像素。

### 6.6 DatasetBundleManifest 与 catalog v2

完成的 normalized dataset 是一个不可变 bundle：

```text
<normalized_dataset>/
├── dataset_bundle.json
├── annotation.jsonl
├── image_manifest.jsonl
├── images/
├── source_inventory.json
├── conversion_plan.json
├── annotation_report.json
├── image_preparation_report.json
└── scripts/
```

`dataset_bundle.json` 至少记录：source candidate/revision/license、source manifest hash、annotation schema
ID/hash、plan hash、converter/tool versions、annotation/image manifest hashes、record/image counts、format、
validation status 和 unresolved risks。

catalog v2 不再直接把 `annotation_path` 当作全部事实，而是指向 bundle manifest。加载器从 bundle 解析
annotation 和 image manifest，并复核 hash/readiness。迁移期读取 legacy catalog v1，但所有新自动注册项
只写 v2。catalog 更新由代码原子完成；Agent 只能提交 registration candidate。

## 7. Agent 工具与权限模型

工具按阶段动态装载。任一 Agent 同时看到的工具不超过 5 个，避免功能重叠和上下文污染。

### 7.1 Dataset Discovery 工具

| 工具 | 功能 | 副作用 |
|---|---|---|
| `dataset_search` | 在允许 connectors 中搜索，返回 bounded candidates | 无 |
| `dataset_describe` | 获取一个 candidate 的 card/Croissant/文件摘要和 metadata provenance | 无 |
| `dataset_submit_acquisition_plan` | 代码验证下载提议并请求/签发 approval | 只写 plan artifact |
| `dataset_acquire` | 使用 scoped approval token 启动或恢复 downloader job | 下载到 staging |

Agent 可以根据第一次搜索结果改变关键词、provider 或候选，但不能直接调用 shell 命令、传任意 URL 或
选择任意本地目录。

### 7.2 Dataset Onboarding 工具

| 工具 | 功能 | 副作用 |
|---|---|---|
| `ingestion_inventory` | 启动/读取 deterministic source inventory job | 只写 inventory artifact |
| `ingestion_check_mapping` | 验证 `ConversionPlanV2` 对 inventory 的引用和约束 | 无 |
| `ingestion_submit_mapping` | 保存通过验证的 candidate mapping 并进入 approval | 只写 candidate artifact |
| `ingestion_build_bundle` | 对 approved plan 执行 annotation/image jobs | 写独立 normalized staging |
| `ingestion_register_bundle` | 复核所有报告和 hash 后原子注册 catalog v2 | 修改 catalog，需 approval/policy |

`ingestion_build_bundle` 是 job trigger，不是模型执行代码。工具拒绝没有 approved mapping hash 的调用。

### 7.3 Capability 工具

| 工具 | 功能 |
|---|---|
| `capability_task_context` | 返回 task requirement、response format、候选安全字段和紧凑 profile |
| `capability_field_summary` | 返回候选字段的 bounded exact values/counts/distinct/null/truncated 状态 |
| `capability_submit_task_plan` | 调用现有 `validate_task_plan`，通过时结束 loop，不直接写 Blueprint |

### 7.4 Reference Curation 工具

| 工具 | 功能 | 边界 |
|---|---|---|
| `reference_search` | 在官方机构、专业学会、PubMed/PMC 和 allowlist 搜索 | 记录 query/domain/date |
| `reference_describe` | 获取 title、publisher、date/version、license/access、landing/download URL | 不把网页指令当 policy |
| `reference_submit_collection` | 验证 source hierarchy、版本和相关性，生成 candidate list | 只写 candidate artifact |
| `reference_acquire` | 下载、hash、解析并分类到 `rag_ready`/link-only/error | 不绕过 paywall/access |

它复用 `collect-medical-reference-materials` 的 provenance、版本筛选和 `rag_ready` contract，但产品运行时代码
应位于 `src/medbenchagent`；skill 和应用调用同一个实现，避免再次产生双真相源。

### 7.5 Task Evidence / Agentic RAG 工具

| 工具 | 功能 |
|---|---|
| `rag_search` | 选择 lexical/semantic/hybrid、metadata filters 和 top-k，返回强制经过 grader 的 reranked passages |
| `capability_field_summary` | 为 letter mapping 核查精确字段值 |
| `rag_report_corpus_gap` | 提交已尝试 queries 和缺失证据类型，请求 ResourceCoordinator 补充 corpus |
| `rag_submit_task_design` | 验证 claim ledger、citations 和现有 `validate_task_design`，通过时结束 loop |

Query planning 是 Agent 决策，不包装成工具。每次 `rag_search` 后固定执行 evidence-grade 节点，Agent 不能
跳过；grader 的模型判断也不是最终验证。提交时代码确认每个引用属于当前 corpus version，chunk text、
PDF hash 和页码可回放，设计中的医学 claim 至少关联一个未冲突的 evidence item。

### 7.6 工具通用 contract

所有工具返回：

```json
{
  "status": "ok | invalid | unavailable | needs_approval | needs_human",
  "data": {},
  "warnings": [],
  "provenance": [],
  "truncated": false,
  "job_ref": null
}
```

大结果写 artifact，模型只接收摘要和引用。运行时注入允许的 dataset/candidate/task/material/corpus/path，
模型不能通过参数扩大权限。所有有副作用工具要求 idempotency key；重复相同调用返回同一 job/result。

## 8. Agentic RAG 详细设计

### 8.1 Corpus 版本

`materials/` 不再只靠目录扫描表达来源。每个可检索 corpus 有 `reference_manifest.jsonl` 和 corpus manifest：

- candidate URL、publisher、version/date、source type、license/access；
- local file、sha256、fetch status、text-extractable status；
- parser/chunker version、chunk count；
- embedding model/reranker/lexical index version；
- selected disease、modality、task scope；
- corpus ID 由内容和索引配置计算。

旧目录 PDF 可以导入生成 manifest。只有 `rag_ready` 文件进入索引；link-only 来源可以显示给用户，但不能
作为 citation。

### 8.2 检索路径

每个 task 先由代码构造可观察输入：task description、question type、modality/anatomy、dataset-support
reason、已知 field semantics 和 evidence gaps。Agent 按复杂度走：

```text
simple path:
  one query → hybrid retrieval → rerank → mandatory grade → submit

corrective path:
  query plan → retrieve → mandatory grade
      ├─ sufficient → claim ledger → submit
      ├─ partial/conflict → rewrite/filter/alternate strategy → retrieve
      └─ corpus gap → Reference Curation → new corpus version → retrieve
```

默认先使用 hybrid retrieval：lexical 覆盖 BI-RADS、TNM、缩写和精确术语，semantic 覆盖自然语言变体，
再通过 RRF 或等价稳定方法合并并 rerank。实际是否优于当前 vector+reranker 必须由 eval 决定；若没有收益，
保留 semantic+reranker 并简化工具 strategy。

### 8.3 Evidence ledger

Agent 不直接把 passages 拼进 `medical_task_description`。先提交 task-level ledger：

```json
{
  "task_id": "finding_attribute.margin",
  "claims": [
    {
      "claim_id": "claim-1",
      "claim": "bounded medical design claim",
      "evidence": [
        {
          "chunk_id": "...",
          "pdf_sha256": "...",
          "page": 12,
          "support": "direct | partial | conflicting"
        }
      ]
    }
  ],
  "unresolved_gaps": []
}
```

Design generator 只能使用 accepted ledger 和 dataset field observation。Validator 要求 citation 与 claim
可追踪、option/answer mapping 不超出 annotation 语义、`medical_task_description` 中的依据有来源。无证据
时返回 `evidence_insufficient`，不使用模型内部常识伪造依据。

### 8.4 预算与终止

| 限制 | Dataset Discovery | Dataset Onboarding | Capability | 单 task Agentic RAG |
|---|---:|---:|---:|---:|
| 最大模型决策轮数 | 10 | 10 | 12 | 10 |
| 最大 search/retrieval calls | 6 | 0 | 0 | 5 |
| 最大 deep inspection calls | 3 | 6 | 6 | 3 |
| 最大 invalid tool calls | 2 | 2 | 2 | 2 |
| 最大同参重复调用 | 1 | 1 | 1 | 1 |

预算是初始实验值，写入配置并由 eval 调整。模型不能提高预算。确定性终止原因：

- `accepted`；
- `needs_approval`；
- `needs_human`；
- `not_found`；
- `adapter_required`；
- `evidence_insufficient`；
- `budget_exhausted`；
- `model_protocol_error`；
- `cancelled`。

失败不能静默降级为未验证输出。每种终止都保留当前 plan、observations 和下一步建议。

## 9. 持久化、记忆和可观察性

### 9.1 Durable jobs

本地 CLI 从 `MemorySaver` 切换为 `SqliteSaver`。dataset download、inventory、normalization、image conversion、
reference collection、indexing 和 research 都有稳定 job ID、checkpoint、idempotency key 和 cancellation。

建议命令：

```text
medbenchagent build
medbenchagent build --resume <run_id>
medbenchagent jobs list
medbenchagent jobs show <job_id>
medbenchagent jobs cancel <job_id>
medbenchagent resources list
```

恢复时验证 checkpoint、job manifest、artifact hash 和允许根路径一致；不一致时停止，不猜测哪份正确。

### 9.2 长期记忆

只保存验证或确认后的结构化事实：

| namespace | 内容 | 失效键 |
|---|---|---|
| `dataset_candidate/<provider>/<repo>/<revision>` | metadata、许可/access、下载结果 | provider revision/metadata hash |
| `dataset_bundle/<dataset_id>/<source_hash>` | validated mapping、字段能力、adapter version | source/schema/converter hash |
| `material/<pdf_sha256>` | verified version、chunk metadata、适用 tasks | PDF/parser/chunker hash |
| `project/<project_id>/preferences` | 用户确认的语言、source allowlist、download policy | 用户修改 |

未确认 Agent conclusion、raw messages、完整 construction events 和 raw chain-of-thought 不进入长期记忆。
memory hit 只是 candidate context；当前 hash/version 必须重新验证。提供查看、导出、删除和按 namespace 清理。

### 9.3 轨迹和解释

在 construction events 上增加 resource/research job 事件，记录：step、tool、脱敏 arguments hash、artifact
ref、tokens、latency、预算、approval、termination reason。用户界面显示计划摘要、工具进度、观察摘要、
来源和风险，不展示或持久化 raw chain-of-thought。

## 10. 分阶段实施计划

每阶段单独提交、可回滚。修改图、schema、稳定 artifacts 或职责边界时同步更新
`docs/ARCHITECTURE.md`。

### Phase 0：契约统一、基线和 tool-call 阻断验证

**目标**：在增加资源自治前消除双真相源，并证明本地 DeepSeek serving 能执行真实多轮工具调用。

工作项：

1. 以 `src/medbenchagent/schemas/annotation.v2.schema.json` 为唯一 schema。
2. 升级或 deprecated 旧 v1 normalization skill；让 preparation skills 和 validator 检查同一 schema hash。
3. 定义 `DatasetCandidate`、`AcquisitionPlan`、`SourceInventory`、`ConversionPlanV2`、
   `DatasetBundleManifest`、`ReferenceCandidate`、`EvidenceLedger` contract。
4. 建立 workflow/RAG/dataset-preparation baseline eval。
5. 用无副作用探针验证单/多轮 tool call、tool result、并行 call、strict arguments、thinking message 回传。
6. 新增 tool-capable model adapter，不修改现有 `complete_json` 行为。

预计关键文件：

- `src/medbenchagent/schemas/annotation.v2.schema.json`；
- `.codex/skills/normalize-medical-image-dataset/`；
- `.codex/skills/prepare-medical-annotations/`；
- `src/medbenchagent/agent_runtime.py`、`src/medbenchagent/llm.py`；
- `src/medbenchagent/resource_contracts.py`；
- `tests/test_agent_runtime.py`、skill/schema consistency tests、`tests/fixtures/agent_eval/`。

硬门：

- annotation schema 只有一个权威 hash，所有 validator 使用 v2；
- 合法 tool arguments 比例至少 95%；
- 100% 非法参数在执行业务函数前被拒绝；
- 连续四轮 tool call/result 可正确回传和恢复；
- 本地 serving 达不到门槛时停止 Agent 实施，不用 prompt 文本 `Thought/Action` 冒充工具调用。

### Phase 1：Durable Resource Job Runtime

**目标**：先建立可恢复、有 provenance 的资源作业层，不立即引入 Agent 搜索。

工作项：

1. `SqliteSaver`、稳定 run/job ID、resume/list/show/cancel。
2. 通用 job 状态机：planned → approval_pending → running → validating → ready/failed/cancelled。
3. staging、原子 publish、idempotency、hash manifest、disk quota 和安全 archive extraction。
4. 配置 dataset source、normalized、materials source、RAG corpus 和 state roots。
5. 现有本地数据集和 PDF 可导入生成 bundle/corpus manifest。

预计关键文件：

- `src/medbenchagent/jobs.py`、`src/medbenchagent/resource_artifacts.py`；
- `src/medbenchagent/settings.py`、`config/default.toml`；
- `src/medbenchagent/cli.py`；
- `src/medbenchagent/construction.py`；
- job、path safety、resume 和 fault-injection tests。

硬门：在 download/inventory/conversion/indexing 的每个阶段终止后都能恢复；同一 job 不产生重复文件或
artifact；越界路径、archive traversal、hash mismatch 和 quota overflow 均明确失败。

### Phase 2：Dataset Discovery 与 Acquisition Agent

**目标**：从自然语言 benchmark 目标搜索候选数据集并在 policy 内完成下载。

工作项：

1. 实现 ModelScope、OpenDataLab/OpenXLab、Hugging Face connectors 和统一 candidate metadata。
2. 读取 dataset card/Croissant/README/file listing，保留 metadata provenance 和 unknown。
3. 实现 Dataset Discovery ReAct loop 和四个 discovery/acquisition tools。
4. 实现 `manual` 与 `open_under_limit` acquisition policy、scoped approval token 和 credential injection。
5. 生成 source manifest、metadata 和 acquisition report。

预计关键文件：

- `src/medbenchagent/resources/dataset_sources.py`；
- `src/medbenchagent/resources/acquisition.py`；
- `src/medbenchagent/agents/dataset_discovery.py`；
- `src/medbenchagent/tools/dataset_discovery.py`；
- connector contract tests、recorded metadata fixtures、download integration tests。

硬门：

- gold discovery cases 的 usable candidate recall@5 达到预设基线；
- license/access/size 缺失不会被模型补全；
- gated/unknown license 不会自动下载；
- downloaded files 与 source manifest 100% hash 对应；
- 搜索不到合适数据集时能正确 `not_found`，不会为完成任务选择明显不匹配资源。

### Phase 3：Dataset Onboarding 与 annotation v2 自动规范化

**目标**：对 Tier A/B 数据集自动生成可复现 bundle 并注册。

工作项：

1. 把 `prepare-medical-annotations` 的 inventory、mapping、review、deterministic conversion 和 report contract
   下沉为应用模块；skill 调用相同代码。
2. 实现 common table inventory 和 `ConversionPlanV2` validator。
3. 实现声明式 mapping engine、bbox transforms、stable grouping/order 和 atomic output。
4. 把 `prepare-medical-images` / `convert-medical-images-to-png` 的受支持转换下沉为受测试 image job；同步
   当前应用的 PNG/JPEG format contract，不再让旧 skill 隐式假定所有非 PNG 都必须重编码。
5. 实现 Dataset Onboarding Agent，遇到语义歧义时请求用户，无法表达时 `adapter_required`。
6. 实现 bundle manifest、catalog v2、legacy migration 和原子注册。

预计关键文件：

- `src/medbenchagent/ingestion/inventory.py`；
- `src/medbenchagent/ingestion/mapping.py`；
- `src/medbenchagent/ingestion/engine.py`；
- `src/medbenchagent/ingestion/images.py`；
- `src/medbenchagent/agents/dataset_onboarding.py`；
- `src/medbenchagent/dataset_catalog.py`；
- `.codex/skills/prepare-medical-annotations/`、`.codex/skills/prepare-medical-images/`；
- mapping fixtures、determinism、bbox、image、catalog migration 和 end-to-end tests。

硬门：

- 每个 ready bundle 是一图一行、v2 schema valid、ID/path 唯一；
- source rows、joins、missing/orphan/duplicate/conflict 全部 reconciled；
- 同输入重复运行 annotation、images 和 manifests byte-identical；
- 原始 source 不修改，完整记录和 PHI 不进入模型/event；
- 未批准 mapping 或 image encoding 不能执行；
- Tier C 数据集稳定输出 `adapter_required` 和可操作 inventory，而不是猜测。

### Phase 4：Capability Research Agent

**目标**：使用新 bundle/profile 按需深查字段，生成更可靠的 Task Plan。

工作项：

1. 实现 `capability_task_context`、`capability_field_summary`、`capability_submit_task_plan`。
2. ReAct 只在 profile 摘要不足时深查；简单任务允许直接提交。
3. accepted candidate 经现有 validator、HITL 后原子 apply；样本分配保持确定性。
4. bundle memory 可复用 validated field capability，但 source/profile hash 变化时失效。

硬门：额外 observation gold cases 正确率至少 80% 且比一次性基线提高至少 10 个百分点；field 越权和
原始行泄漏为零；无显著收益则保留现有一次性 Task Plan。

### Phase 5：Reference Curation 与 Agentic RAG

**目标**：系统能发现/整理缺失医学资料，并用 adaptive/corrective retrieval 生成可回放 design。

工作项：

1. 把 `collect-medical-reference-materials` 的 source selection、fetch、manifest 和 `rag_ready` contract
   下沉为应用模块；skill 和产品共用。
2. 实现 Reference Curation Agent、来源 allowlist/policy 和 corpus version manifest。
3. 将 `LocalPdfRag` 拆分为 corpus/index、retriever、grader、ledger 和 designer 边界。
4. 实现 lexical/semantic/hybrid + reranker；以 eval 决定最终 strategy。
5. 实现 Task Evidence Agent：query plan、retrieve、grade、correct、corpus gap、ledger、submit。
6. 设计写入 Blueprint 前先形成 immutable candidate design artifact，再由 deterministic node apply。

预计关键文件：

- `src/medbenchagent/materials/collection.py`；
- `src/medbenchagent/rag/corpus.py`、`retrieval.py`、`evidence.py`；
- `src/medbenchagent/agents/reference_curation.py`；
- `src/medbenchagent/agents/evidence_research.py`；
- `src/medbenchagent/rag/graph.py`；
- `.codex/skills/collect-medical-reference-materials/`；
- retrieval benchmark、citation replay、corpus-gap 和 prompt-injection tests。

硬门：

- citation PDF/hash/page/chunk/corpus scope 100% 可回放；
- accepted design 100% 通过 contract；
- retrieval recall/precision 和 claim support 分别报告且不低于当前基线；
- 首轮失败 case 能纠正或正确 `evidence_insufficient`；
- corpus gap 只通过 Reference Curation 补资料，不直接引用 Web；
- 人工 design 修订轮数至少下降 20%，否则不默认启用多轮路径。

### Phase 6：产品整合、长期记忆与默认模式

**目标**：交付从目标到 benchmark 的可恢复工作台，而不是若干 Agent demo。

工作项：

1. CLI/Studio 展示 resource plan、许可/体量、jobs、mapping review、action timeline、evidence 和风险。
2. ResourceCoordinator 接入根图；现有 ready resource path 保持快速路径。
3. 引入 `SqliteStore` validated memory 和 hash-based invalidation。
4. `research_summary.json` 汇总资源来源、mapping、evidence gaps、终止原因和成本，不复制大 artifact。
5. manifest 记录 dataset bundle、corpus、Agent/tool/schema versions、memory hits、预算和 policy。
6. 对 `workflow`、`research_local`、`research_connected` 做固定数据集 A/B 和人工盲审。

成为默认 research 产品的硬门：

- Phase 2–5 全部质量、安全和恢复门通过；
- 从无本地 dataset 的受支持 case 能完成 discovery → ready bundle → benchmark；
- 从本地原始 dataset 的受支持 case 能完成 onboarding → benchmark；
- final hard-audit rejection rate 不高于 workflow 基线；
- 没有高严重度许可、路径、PHI、source injection 或未批准副作用问题；
- token、wall time、磁盘和人工修订成本都有报告，产品收益支持额外开销。

### Phase 7：可选扩展

只在前述产品稳定后考虑：TCIA/Grand Challenge 等新 connector、更多 declarative transforms、受审查的
custom adapter drafting、并行 task evidence workers、Postgres/对象存储、多用户权限和分布式 jobs。

多 Agent 试验只有同时满足：单 task 质量稳定；Evidence research 占总耗时超过 60%；任务无共享可变
状态；固定总 token 下并发提升吞吐且不降低 citation validity；worker 失败不会丢失或重复其他 artifacts。

## 11. 测试和评测体系

### 11.1 Resource eval

- dataset discovery recall@k、candidate relevance、license/access completeness；
- duplicate repository/content recognition、revision pinning、download success/hash coverage；
- policy approval precision，特别是 gated/unknown/oversized cases；
- reference source hierarchy、current-version selection、link-only correctness。

### 11.2 Ingestion eval

- authoritative image population、join cardinality 和 source-row reconciliation；
- annotation v2 schema、entity scope、bbox conversion 和 value preservation；
- image mapping、format handling、dimension/hash/non-constant checks；
- same-input byte determinism、source immutability、catalog readiness；
- `needs_human`/`adapter_required` 对真实歧义的 precision。

### 11.3 Agentic RAG eval

- lexical/semantic/hybrid retrieval recall@k 和 precision@k；
- reranker gain、query recovery rate、corpus-gap detection；
- citation precision、claim-level support/coverage/conflict detection；
- design contract validity、answer mapping coverage 和人工修订轮数；
- retriever 与 designer 分别诊断，不能只看最终 design 是否通顺。

### 11.4 Agent runtime 与端到端

1. 工具 contract：schema、权限、bounded output、truncation、idempotency；
2. 脚本化轨迹：fake model 覆盖重复、invalid、budget、cancel、needs approval；
3. 模型行为：允许不同有效轨迹，按 end state 和约束评估；
4. 故障注入：网络中断、磁盘不足、archive 错误、模型错误、RAG 空结果、checkpoint 损坏；
5. 最终 artifact：bundle、Blueprint、raw、provenance、Audit 和 manifest 全链路回放。

核心运行指标包括 valid tool-call rate、重复调用率、budget exhaustion、resume success、duplicate artifact、
tokens、wall time、disk bytes、峰值 state、越权调用和敏感数据泄漏。

每次修改模型、tool schema/description、connector、mapping engine、image converter、retriever、reranker、
chunker 或 memory policy 都运行对应固定 eval suite。单个成功 demo 不能作为阶段完成标准。

## 12. 安全与治理边界

- Agent 不代表用户接受数据许可、隐私协议、付费或 gated access 条款。
- credentials 只由 runtime 注入，不出现在模型消息、tool arguments、events 或 manifest。
- 下载仅通过 allowlisted connectors，目标位于隔离 source root，经过 quota、hash 和安全解压。
- 不提供任意 shell、Python、SQL、URL 或文件读取工具。
- 模型只有 read/research、submit candidate 和带 approval token 的 job trigger；没有直接 artifact write。
- 原始表、完整记录、PHI 和图像像素不发送到外部模型。
- 行级转换、图像转换、schema validation、catalog 注册和 Blueprint apply 均由确定性代码执行。
- Web/PDF 文本只作为不可信数据，不能改变 system/tool/policy；Web 不可直接成为 citation。
- mapping identity、entity scope、bbox convention、mask conversion 和 medical image encoding 不明确时停止。
- 长期记忆可查看、可删除、按 source/schema/tool hash 失效，不保存 raw chain-of-thought。
- Agent 达到预算、来源冲突、许可不明或需要扩大目标时请求用户，不自行放宽 policy。

## 13. 明确不做的事情

- 不声称支持任意医学数据集全自动转换。
- 不让 Agent 根据列名、文件名、图像外观或医学常识补充源数据没有的标签。
- 不让模型逐行处理 annotation 或生成后直接执行任意 dataset-specific Python。
- 不把 provider CLI、filesystem、DuckDB 或 HTTP client 的全部底层 API 暴露为工具。
- 不为了 ReAct 把下载、hash、转换、注册、样本分配、生成和硬审计改成模型决策。
- 不把五个角色改成互相闲聊的人格 Agent，不让模型跳过 Generation/Audit/finalize。
- 不把模型自己的 evidence grade 当作最终 citation truth。
- 不在无 eval 的情况下默认开放 Web、多 Agent、并行研究或 custom code execution。
- 不把全部历史消息、原始数据或 construction events 塞入向量库当作“记忆”。

## 14. 产品完成定义

满足以下条件时，项目才真正成为端到端自主 benchmark research 产品：

1. 用户只给 benchmark 目标时，系统能搜索、比较并提出可核验的数据集和医学资料 acquisition plan。
2. policy 允许的开放资源可以自动下载；其他资源在明确许可/体量/访问确认后恢复执行。
3. Tier A/B 数据集能从 source manifest 经 inventory、mapping、annotation v2、image preparation、validation
   和 catalog v2 注册形成可复现 bundle。
4. 未知或歧义数据稳定停止为 `needs_human`/`adapter_required`，不静默猜测。
5. Capability Agent 在需要额外观察的 case 中会依据字段工具结果改变任务决策。
6. Agentic RAG 会依据 retrieval observation 改写查询、换策略、补 corpus 或声明证据不足，而不是固定调用
   序列。
7. 每个 accepted medical claim、citation 和 task design 都可回放到 corpus chunk 和 source hash。
8. 最终题目答案仍由 annotation witness 唯一确定，Agent 不参与自由生成答案。
9. 所有长任务可跨进程恢复，不产生重复下载、bundle、artifact、题目或 provenance。
10. 用户能查看 resource plan、许可、mapping、工具进度、证据、风险和预算，并能暂停、修改或取消。
11. 长期记忆只复用验证事实，source/schema/tool version 变化后自动失效。
12. 与 workflow 基线相比，资源准备时间、研究质量或人工修订成本有量化改善，且安全、token、时间和
    磁盘代价可接受。

最终产品是一个领域专用、证据驱动、可恢复、可监管的 **Autonomous Medical Benchmark Factory**。
它从资源发现开始具备真实自主能力，但每一项自治都由明确工具、policy、artifact、validator 和 eval
约束，而不是用更多 Agent 名称掩盖不确定性。
