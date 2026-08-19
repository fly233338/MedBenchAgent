# MedBenchAgent 自主研究 Agent 产品与实施路线图

状态：提案

日期：2026-08-19

适用范围：benchmark 构建流程；不改变独立的 Medical VLM 推理与评测子系统

## 1. 执行结论

MedBenchAgent 的目标产品不应是通用聊天 Agent，也不应把现有五个角色全部替换为 ReAct
循环。可落地的产品形态是一个 **Medical VLM Benchmark Research Workspace**：用户给出
benchmark 目标并确认研究范围，系统在受限数据集和医学资料内自主核查任务可行性、补齐证据、
生成可审核的方案；用户确认后，现有确定性生成和 Audit 流程负责产出最终 benchmark。

ReAct 只用于两个确实需要“观察后再决定下一步”的功能：

1. **数据集能力研究**：面对语义不明确的标注字段，Agent 决定是否继续查看字段值分布、比较候选
   字段或放弃任务，再提交完整 Task Plan。
2. **医学证据与题型设计研究**：Agent 根据检索质量决定是否改写查询、补充另一份材料、核查答案
   字段精确值或声明证据不足，再提交一个可验证的 task design。

以下环节不引入 ReAct：需求字段提取、数据集文件校验、样本数分配、Blueprint 写入、逐题生成、
答案和 evidence 固定、provenance、硬审计、manifest 导出。这些问题已有明确算法和契约，改为
Agent 决策只会增加成本和不确定性。

首个可发布版本采用 **单 Agent、两段受限研究子图、固定外层工作流**。只有在评测证明任务间
可以独立并行、上下文成为实际瓶颈且收益覆盖额外成本后，才考虑 supervisor-worker 多 Agent。

## 2. 当前基线与需要解决的问题

当前根图是固定顺序：

```text
run_initialize
  → SpecificationAgent
  → CapabilityAgent
  → ClinicalDesignAgent
  → GenerationAgent
  → AuditAgent
  → run_finalize
```

现有实现已经具备后续 Agent 化所需的重要基础：

- 严格的 Agent input/output/private state contract；
- 内容寻址的数据集缓存、DuckDB、dataset profile 和 access manifest；
- 本地 PDF 检索、reranker、引用和版本化 RAG design；
- 受限字段投影、确定性答案、annotation witness 和 provenance；
- 幂等生成进度、硬审计、语义筛查和不可变运行产物；
- LangGraph interrupt、子图、stream 和 checkpoint 接口。

当前“不是真正自主 Agent”的原因不是缺少角色名称，而是缺少以下运行时能力：

- 模型不能返回并连续消费 tool calls；`complete_json()` 只能完成一次固定 JSON 生成；
- 根图和角色内部都没有通用的 decide → act → observe → decide 循环；
- CLI checkpoint 只存在于当前进程，退出后不能恢复；
- 运行产物和 RAG 缓存不会自动成为跨运行、可失效的长期记忆；
- 没有 Agent 专用的 step/tool/token 预算、终止原因和轨迹评测；
- 现有 Task Plan 一次接收完整 profile，遇到 top values 不足以判断语义时无法主动获取更多受限信息；
- 现有 RAG design 每个任务只执行一次预设检索，无法依据低相关结果主动改写查询。

## 3. 外部项目与工程经验

本计划参考下列公开项目和生产经验，但只迁移与本项目问题相符的模式。

| 来源 | 可迁移经验 | 本项目决策 |
|---|---|---|
| [ReAct](https://arxiv.org/abs/2210.03629) | 推理与外部行动交替，观察结果可修正计划 | 只用于字段语义核查和医学证据研究两个开放决策点 |
| [Open Deep Research](https://github.com/langchain-ai/open_deep_research) | 先形成 research brief；显式完成工具；限制 research iterations 和并发研究单元 | 采用 research brief、显式终止和硬预算；首版不采用多 Agent |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | planner、information gathering、publisher 分离；来源跟踪贯穿过程 | 将研究证据与最终构建分离；复用现有 provenance 和版本化 artifact |
| [STORM / Co-STORM](https://github.com/stanford-oval/storm) | 先研究和组织证据，再写作；允许人类在研究过程中调整方向 | 研究方案确认后才进入生成；保留实时进度和 interrupt |
| [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | 优先简单、可组合方案；只有可测收益时才增加自治复杂度 | 固定外层工作流，局部引入 Agent；每阶段设置继续/停止门 |
| [Anthropic: multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system) | 多 Agent 适合高价值、可并行且超出单上下文的研究，但成本和协调复杂度明显上升 | 多 Agent 不是首版目标，必须由上下文和吞吐数据触发 |
| [Anthropic: writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | 工具应少而明确、响应高信号、严格命名空间，并通过 Agent eval 优化 | 首版只有五个只读/提交工具，不暴露底层 API、SQL 或文件系统 |
| [OpenAI Deep Research 产品流程](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt) | 用户可审阅研究计划、观察进度、随时打断，最终报告有可核验来源 | 产品提供计划确认、实时 action timeline、证据引用和中断恢复 |

这些项目不是直接模板。MedBenchAgent 与开放 Web research 的关键差异是：最终答案必须由规范标注
唯一确定，医学事实必须可引用，数据访问必须遵守字段白名单。因此，本项目的自主性必须发生在
“研究和提出方案”阶段，而不是“自由写答案”阶段。

## 4. 可行性反思与架构决策

### 4.1 是否需要把根图改成动态 supervisor

**初步设想**：让一个 supervisor 自主决定调用五个角色。

**反思**：五个阶段存在稳定依赖，Generation 必须等待确认后的 Blueprint，Audit 必须等待 raw 和
provenance。动态选择不会产生有效新路径，却会让遗漏阶段成为可能。

**决策**：根图继续固定编排；只在 Capability 和 Clinical Design 内部加入动态研究子图。

### 4.2 CapabilityAgent 是否真的需要 ReAct

**反对理由**：当前模型已经一次看到全部任务和 profile，简单再包一层 ReAct 不会增加信息。

**成立条件**：Agent 必须能获得当前提示里没有的、但代码可安全提供的额外观察，例如一个候选字段
最多 32 个精确 scalar、计数、截断状态和语义类别风险；并能依据观察改变 retain/drop、字段或 metric
决策。

**决策**：只有加入按需字段深查工具后，Capability ReAct 才实施。如果离线评测没有优于当前一次性
Task Plan，则保留现状并删除该实验分支。

### 4.3 ClinicalDesignAgent 是否真的需要 ReAct

**现有缺口**：每个任务使用一次固定检索查询；当结果不足、偏离具体 question type 或没有覆盖答案
类别定义时，只能生成失败或等待人工反馈。

**具体收益**：Agent 可以查看首轮 passage 和分数，改写疾病/模态/任务组合查询，或查询答案字段值后
重新组织 options 和 mapping。

**决策**：这是首版最明确的 ReAct 场景。没有足够证据时必须返回 `evidence_insufficient`，不能依靠模型
内部知识补齐引用。

### 4.4 是否立即采用多 Agent

**反思**：当前任务共享同一个 Blueprint、dataset profile 和材料集合，Capability 决策还需要跨任务
去重。并行角色会复制上下文、增加冲突合并，并使复现更困难。

**决策**：首版只有一个 Capability research loop 和按任务顺序执行的 Evidence research loop。未来可
把互相独立的 task evidence research 作为同一子图的并行 map，而不是先创造不同人格的 Agent。

多 Agent 试验只有同时满足以下条件才立项：

- 单任务 Evidence research 的质量门已经稳定通过；
- 一次运行的 evidence research 时间占总构建时间超过 60%；
- 任务间没有共享可变状态，合并只依赖 task ID；
- 2–4 个并发单元在固定总 token 预算下提升吞吐且不降低 citation validity；
- 故障注入证明某个 worker 失败不会使其他 task artifact 丢失或重复。

### 4.5 “记忆”是否等于保存所有消息

**反思**：保存完整对话会快速膨胀，`construction_events.jsonl` 还可能包含未脱敏用户输入。未验证的
Agent 结论跨运行复用会放大错误。

**决策**：分离 thread checkpoint、validated memory 和 immutable artifacts；不保存 raw chain-of-thought，
不把完整 construction events 建立向量索引，不让未确认结论进入长期记忆。

### 4.6 本地 DeepSeek V4 Flash 是否能可靠调用工具

官方 DeepSeek API 支持 tool calls，但当前项目连接本地 OpenAI-compatible endpoint，能否正确返回
`tool_calls` 取决于实际 serving、tool parser 和 chat template。不能从模型名称推断本地服务已经可用。

**决策**：工具调用兼容性是 Phase 0 的阻断门。必须验证多轮 tool call、tool result 回传、并行 tool
call、invalid arguments 和 thinking message 回传。失败时先修复 serving 配置；不使用只在 prompt 中打印
`Thought/Action` 文本的伪工具循环冒充产品功能。

### 4.7 是否首版开放 Web 和代码执行

**反思**：开放 Web 会引入 prompt injection、来源质量、版本漂移和内网部署问题；任意 Python/shell
对 benchmark 构建没有首版必需价值。

**决策**：MVP 只研究已选择的本地医学 PDF 和规范数据集。后续外部检索只接入 PubMed、指南发布机构
或用户允许域名，下载并验证为本地材料后才能用于 design。任何阶段都不提供任意 shell、Python、SQL
或文件读取工具。

## 5. 最终产品形态

### 5.1 产品定位

产品名称暂定 **MedBench Research Workspace**。它不是替用户自由搜索医学答案，而是将以下过程自动化：

> 给定 benchmark 目标、一个已注册数据集和一组已批准医学材料，研究哪些能力可以被该数据集可靠
> 评测，为每项能力建立标注证据和医学依据，生成可审核方案，然后用确定性代码构建并审计 benchmark。

### 5.2 用户流程

```text
1. 用户描述 benchmark 目标
2. SpecificationAgent 提取 BenchmarkInfo，用户确认
3. 系统推荐数据集，用户确认
4. 系统展示 Research Plan：候选任务、资料范围、预算、停止条件
5. 用户确认后启动 Capability research：核查任务与字段
6. 系统展示 Capability Proposal：retain/drop、supporting fields、metric 和字段统计；用户确认后执行
   确定性样本分配
7. 系统启动 Evidence research：检索医学证据并设计题型
8. 系统展示 Evidence-backed Design Proposal
   - question templates、answer mapping
   - 引用、证据不足和风险
9. 用户整体确认或用自然语言定向修改
10. 现有 GenerationAgent 确定性生成题目
11. 现有 AuditAgent 审计并交付 benchmark、manifest 和报告
```

用户可以在第 5 步查看实时 action timeline，并在 tool call 之间暂停、缩小范围或取消。timeline 展示
结构化计划摘要、工具名、经过脱敏的参数、观察摘要和终止原因，不展示或持久化模型 raw chain-of-thought。

### 5.3 目标图结构

```text
run_initialize
  → SpecificationAgent                    # 固定 JSON 提取 + HITL
  → CapabilityAgent
      → dataset_select                     # 现有固定流程 + HITL
      → research_plan_review                # 新增 HITL
      → capability_decide ↔ capability_tools # 有界 ReAct
      → task_plan_validate_and_review       # 代码验证 + HITL
      → deterministic_sample_allocation
  → ClinicalDesignAgent
      → evidence_decide ↔ evidence_tools    # 每 task 有界 ReAct
      → design_validate_and_review          # 代码验证 + HITL
  → GenerationAgent                        # 保持确定性
  → AuditAgent                             # 保持硬审计 + 抽样语义筛查
  → run_finalize
```

根图仍然可以清楚显示五个业务角色。新增自主循环保留在角色私有 state 中，不把完整消息、passage 文本、
字段值和 tool payload 提升到根状态。

### 5.4 运行模式

迁移期提供两个显式模式：

- `workflow`：保持当前行为，作为基线和回滚路径；
- `research`：启用受限 ReAct 和持久 checkpoint。

`research` 达到发布门后成为默认值；`workflow` 至少保留一个发布周期用于结果对比和故障回退。模式
必须写入 `effective_config.json` 和最终 manifest，不能根据模型响应隐式切换。

## 6. ReAct 的具体功能规格

### 6.1 工具集合

首版工具固定为五个。工具名按资源命名，功能不重叠，全部使用严格 JSON schema。

| 工具 | 调用者 | 具体功能 | 禁止内容 |
|---|---|---|---|
| `dataset_task_context` | Capability | 返回一个 task 的要求、response format、候选安全字段及紧凑 profile | 原始行、标识符、路径、长文本、bbox 坐标 |
| `dataset_field_summary` | Capability、Evidence | 返回当前 task 安全候选字段的精确 bounded values、计数、distinct count、null/truncated 状态 | 任意 SQL、候选白名单外字段、完整数据集导出 |
| `medical_evidence_search` | Evidence | 在已确认 PDF 中检索，返回 bounded passages、chunk ID、PDF、页码和 rerank score | 任意文件路径、未选择材料、网络访问 |
| `proposal_submit_task_plan` | Capability | 调用现有 `validate_task_plan`；成功时结束循环并保存 candidate 到 state，失败时返回逐项错误 | 直接修改 Blueprint 或写 artifact |
| `proposal_submit_task_design` | Evidence | 调用现有 `validate_task_design`；成功时结束当前 task 循环，失败时返回契约错误 | 直接写 Blueprint、绕过字段或引用验证 |

工具统一返回：

```json
{
  "status": "ok | invalid | unavailable",
  "data": {},
  "warnings": [],
  "provenance": [],
  "truncated": false
}
```

每个响应设置结构和字符上限。大对象写入角色私有 artifact，只把引用和摘要返回模型。工具实现接受
`run_id`、dataset access manifest 和允许资源集合等运行时注入值；模型不能把这些安全上下文作为参数
覆盖。

### 6.2 Capability ReAct 示例

`finding_presence` 候选任务要求同时存在明确正例和负例。Agent 的有效轨迹可以是：

```text
decide: 需要确认候选 finding 字段是否真的表达 presence/absence
act: dataset_task_context(finding_presence)
observe: 候选字段有多个值，但 profile 只展示高频 finding 名称
act: dataset_field_summary(image.annotation.findings[*].type)
observe: 值全部是阳性 finding 名称，没有 absent/normal 类别
decide: 该字段不能支持 presence/absence，drop 任务
act: proposal_submit_task_plan(...)
observe: validation accepted
```

如果 Agent 只是固定调用每个工具一次，或观察不会改变决策，则该实现不算有效 ReAct，应退回一次性
结构化生成。

### 6.3 Evidence ReAct 示例

```text
decide: 首轮查询需覆盖具体任务和模态
act: medical_evidence_search("breast ultrasound lesion margin assessment guideline")
observe: 只有一般性 BI-RADS 描述，未覆盖目标 question type
decide: 改写为 question type 对应术语并限定已选指南
act: medical_evidence_search("BI-RADS ultrasound circumscribed non-circumscribed margin")
observe: 获得可引用页码和定义
act: dataset_field_summary(image.annotation.lesion.margin)
observe: 精确值可以映射为三个不重叠选项
act: proposal_submit_task_design(...)
observe: validation accepted
```

若连续检索仍没有达到最小相关度或引用覆盖，Agent 返回结构化
`evidence_insufficient(task_id, attempted_queries, reason)`，进入人工审核，不得生成无引用 design。

### 6.4 循环预算与终止

默认预算先作为实验值进入配置，并由评测调整：

| 限制 | Capability research | 单个 Evidence research task |
|---|---:|---:|
| 最大模型决策轮数 | 12 | 8 |
| 最大字段深查次数 | 6 | 2 |
| 最大医学检索次数 | 0 | 4 |
| 最大 invalid tool calls | 2 | 2 |
| 最大同参重复调用 | 1 | 1 |

确定性终止原因只有：

- `accepted`：提交内容通过代码验证；
- `needs_human`：证据不足、工具不可用或存在需要用户选择的冲突；
- `budget_exhausted`：达到任一硬预算；
- `model_protocol_error`：连续无效 tool call；
- `cancelled`：用户中断。

模型不能自行提高预算。`budget_exhausted` 和 `model_protocol_error` 不得自动退化为未经说明的一次性模型
答案，必须保留当前 observation refs 并进入人工审核。

## 7. 持久化与记忆设计

### 7.1 Thread checkpoint

本地 CLI 从 `MemorySaver` 切换为 `SqliteSaver`，以稳定 `thread_id` 关联 `run_id`。新增：

```text
medbenchagent build
medbenchagent build --resume <run_id>
medbenchagent runs list
```

恢复时验证 checkpoint、run directory、working Blueprint 和已记录 artifact hash 一致。任何不一致都停止，
不猜测哪一份状态正确。

### 7.2 长期记忆

长期记忆只保存经过验证或确认的结构化事实：

| namespace | 内容 | 失效键 |
|---|---|---|
| `project/<project_id>/preferences` | 用户确认的语言、默认排除任务、常用材料类别 | 用户显式修改 |
| `dataset/<dataset_id>/<annotation_sha256>` | 已验证字段能力、失败原因、适用 question types | annotation hash 或 profile version 变化 |
| `material/<pdf_sha256>` | 已确认引用片段元数据和适用 task 关键词 | PDF hash、chunk/reranker version 变化 |

写入规则：

- 只有代码验证通过且用户确认的 Task Plan/design 可以生成长期记忆；
- Agent 自己的 reflection、失败草稿和 raw messages 不进入长期记忆；
- memory item 必须带来源 artifact、hash、schema version 和创建时间；
- memory 命中只能作为候选上下文，仍需对当前 dataset/material hash 重新验证；
- 默认不对未脱敏自然语言输入建立 embedding；
- 提供按 namespace 查看和删除记忆的管理命令。

本地模式使用 `SqliteStore`；生产部署再根据并发和运维条件迁移到 Agent Server managed Store 或
Postgres。现有内容寻址缓存和版本化 artifacts 继续作为事实源，Store 只是索引和复用层。

### 7.3 轨迹记录

在现有 construction events 上增加：

- `research_started` / `research_completed`；
- `tool_call_started` / `tool_call_completed` / `tool_call_failed`；
- `research_terminated`；
- step index、tool name、脱敏 arguments hash、observation artifact ref；
- tokens、latency、termination reason、budget counters。

不记录 raw chain-of-thought。需要解释时保存不超过固定长度的 `decision_summary`，它描述“缺少什么证据、
为什么调用该工具”，而不是模型内部逐 token 推理。

## 8. 分阶段实施计划

每个阶段都必须单独提交、可回滚，并在修改图契约时同步更新 `docs/ARCHITECTURE.md`。

### Phase 0：基线评测与工具调用技术验证

**目标**：证明本地模型能够稳定执行真实 tool calls，并建立判断 ReAct 是否有价值的基线。

工作项：

1. 建立 30–50 个脱敏、小规模 Agent eval case，覆盖：
   - 字段有多个 raw values 但只有一个语义类别；
   - finding presence 缺少显式负例；
   - 多个字段都看似可用但只有一个与任务语义一致；
   - RAG 首轮无结果、低分结果、跨 PDF 冲突；
   - answer mapping 值截断；
   - invalid tool arguments、工具异常和重复调用。
2. 固化当前一次性 Task Plan/RAG design 的质量、token、延迟和人工修订轮数基线。
3. 用两个无副作用探针工具验证本地 DeepSeek/vLLM：单 tool call、多轮 tool result、并行 call、严格参数、
   thinking 模式消息回传和超长响应。
4. 定义 canonical `ToolCall`、`ToolResult` 和 provider message adapter，但不接入业务图。

预计关键文件：

- `src/medbenchagent/llm.py`：保留 `complete_json`，新增独立 tool-capable 调用边界；
- `src/medbenchagent/agent_runtime.py`：provider message 转换、预算和终止类型；
- `tests/test_agent_runtime.py`；
- `tests/fixtures/agent_eval/`。

进入下一阶段的硬门：

- 合法 tool arguments 比例至少 95%；
- 100% 的非法参数被代码拒绝且没有执行业务函数；
- 连续四轮 tool call/result 可正确恢复上下文；
- 失败不会被解析为普通成功回答；
- 探针测试在本地 serving 配置中可重复。

如果两轮 serving/schema 调整后仍达不到门槛，停止 ReAct 实施，记录模型服务阻断，不用 prompt 文本动作
替代真实 tool calls。

### Phase 1：可恢复运行与统一 Agent 运行时

**目标**：先解决长运行可靠性，再增加自治。

工作项：

1. CLI 使用 `SqliteSaver`，建立 run ID 与 thread ID 映射。
2. 实现 `--resume` 和只读的运行列表；继续支持当前新建流程。
3. Agent runtime 增加 max steps、invalid-call budget、same-call deduplication 和 cancellation。
4. construction events 增加 tool/research 事件，但暂不改变五角色行为。
5. 在每个 interrupt、RAG task、generation row 和 finalize 前后做故障恢复测试。

预计关键文件：

- `config/default.toml`、`src/medbenchagent/settings.py`；
- `src/medbenchagent/cli.py`；
- `src/medbenchagent/workflow/graph.py`；
- `src/medbenchagent/construction.py`；
- `tests/test_cli.py`、`tests/test_run_artifacts.py`。

验收门：

- 在关键节点强制终止进程后可用同一 run ID 恢复；
- 恢复后 artifact hash 与不中断运行一致；
- 没有重复 human interaction、Task Plan 版本、题目 ID 或 provenance；
- checkpoint 与 artifact 不一致时明确失败；
- 当前 `workflow` 模式回归测试全部通过。

### Phase 2：Capability Research MVP

**目标**：让 Agent 自主决定何时深查字段并据此修正完整 Task Plan。

工作项：

1. 实现 `dataset_task_context`、`dataset_field_summary` 和
   `proposal_submit_task_plan`。
2. 新增 `CapabilityResearchState`，只保存 bounded messages、预算、已查字段、observation refs 和 candidate
   Task Plan。
3. 在 `CapabilityAgent` 内以 feature mode 替换 `task_plan_generate`，数据集选择、Task Plan review、apply
   和 sample allocation 保持不变。
4. 实现 Research Plan interrupt：展示任务范围、可用工具和预算，用户确认后才运行。
5. 工具只调用现有 profile/access manifest 和 validator；成功提交前不修改 Blueprint。

预计关键文件：

- `src/medbenchagent/agents/capability_research.py`；
- `src/medbenchagent/agents/contracts.py`；
- `src/medbenchagent/tools/contracts.py`；
- `src/medbenchagent/tools/dataset.py`；
- `src/medbenchagent/tools/proposal.py`；
- `src/medbenchagent/agents/capability.py`；
- `tests/test_capability_research_agent.py`。

验收门：

- 所有 accepted Task Plan 都通过现有 `validate_task_plan`；
- field path 越权和原始记录泄漏为零；
- 专门需要额外 observation 的 gold cases 正确率至少 80%，且比当前基线提高至少 10 个百分点；
- false retain 和 false drop 分别报告，不能只看总正确率；
- median tool calls 不超过 6，任一运行不超过硬预算；
- 若质量没有显著提升，则 `workflow` 保持默认并撤回 Capability ReAct。

### Phase 3：Evidence Research MVP

**目标**：让每个保留任务通过多轮检索和字段核查得到可引用 design，或明确声明证据不足。

工作项：

1. 将 `LocalPdfRag.retrieve` 包装为 `medical_evidence_search`，保持 selected materials、reranker threshold 和
   bounded passage 约束。
2. 实现 `proposal_submit_task_design`，复用现有 `validate_task_design`。
3. 新增每 task 的 `EvidenceResearchState`，检索 query、passage body 和精确字段值仅保留在私有 checkpoint
   或内容 artifact。
4. 当前 `design_task` 作为 `workflow` 基线；`research` 模式进入 decide/tools 循环。
5. 没有足够 evidence 时输出结构化 `evidence_insufficient` 并转入 review，不自动接受模型常识。
6. 继续逐 task 写入 Blueprint，但只在 validated candidate 产生后由确定性节点写入。

预计关键文件：

- `src/medbenchagent/agents/evidence_research.py`；
- `src/medbenchagent/tools/evidence.py`；
- `src/medbenchagent/rag/pdf.py`；
- `src/medbenchagent/rag/graph.py`；
- `tests/test_evidence_research_agent.py`；
- `tests/test_pdf_rag.py`。

验收门：

- citation 的 PDF、页码、chunk ID 和 selected-material scope 100% 可回放；
- 100% accepted design 通过现有 validator；
- citation precision、evidence coverage、answer mapping coverage 均不低于 workflow 基线；
- 首轮检索故意失败的 case 能通过改写查询恢复，或正确返回 `evidence_insufficient`；
- Agent 不调用未选择 PDF，不把内部知识伪装为 citation；
- 人工 design 修订轮数下降至少 20%，否则不默认启用。

### Phase 4：产品整合、长期记忆与默认模式评估

**目标**：形成可交付的 Research Workspace，而不是只有后端 Agent demo。

工作项：

1. CLI/Studio 展示 Research Plan、action timeline、预算、暂停、恢复和 Proposal review。
2. 生成一个 `research_summary.json`，只汇总最终任务、证据引用、风险、终止原因和资源用量；不复制完整
   Task Plan/RAG design 内容。
3. 引入 `SqliteStore` 和严格的 validated memory 写入/失效策略。
4. manifest 记录 mode、Agent schema version、toolset version、memory hits 和完整预算。
5. 在固定评测集上进行 workflow/research A/B，并做人工盲审。

预计关键文件：

- `src/medbenchagent/memory.py`；
- `src/medbenchagent/cli.py`；
- `src/medbenchagent/run_export.py`；
- `src/medbenchagent/run_artifacts.py`；
- `docs/ARCHITECTURE.md`、`README.md`；
- memory、manifest、端到端和隐私测试。

`research` 成为默认模式的硬门：

- Task Plan 和 design gold eval 都通过 Phase 2/3 门槛；
- 最终 hard-audit rejection rate 不高于 workflow 基线；
- 从任一支持 checkpoint 恢复的端到端测试通过；
- 没有高严重度数据泄露或路径越权；
- 总 token/延迟增幅有报告，并被质量或人工时间收益支持；
- 至少两名人工 reviewer 的盲审偏好 research 结果，或 research 显著降低修订轮数。

### Phase 5：可选的受控外部研究与并行化

本阶段不属于首个正式产品版本。

只有本地 Evidence research 无法覆盖目标疾病或指南版本时，才增加外部文献发现。允许来源限定为
PubMed、学会/政府指南站点和用户 allowlist；下载、hash、版权/可访问性、PDF 解析和人工材料确认完成后，
新材料才进入现有 RAG。Web 内容中的指令一律视为不可信数据。

并行化优先使用同一 Evidence research 子图按 task map 执行。只有满足 4.4 的全部条件才实现真正的
supervisor-worker，并明确限制最大并发、总 token、取消传播和部分失败合并。

## 9. 测试与评测体系

Agent 评测不能只断言某一条固定轨迹。相同目标可能有不同的有效工具路径，应同时评估工具行为和最终
状态。

### 9.1 四层测试

1. **工具 contract 测试**
   - strict schema、权限、bounded output、truncation、idempotency；
   - 注入未知 dataset/task/field/path 时必须失败；
   - tool result 不包含 identifier、raw row 或未选择材料。
2. **脚本化轨迹测试**
   - fake model 返回指定 tool calls；
   - 覆盖合法多轮、重复调用、无 tool call、invalid arguments、budget exhausted 和 cancel；
   - 不依赖真实模型，稳定验证图路由和 state reducer。
3. **模型行为 eval**
   - 用固定 case 比较 workflow 和 research；
   - 记录 task 决策、工具选择、引用、token、延迟和终止原因；
   - 允许不同轨迹，只按约束和 end state 判定。
4. **端到端与故障注入**
   - 工具超时、RAG 空结果、模型协议错误、进程终止、checkpoint 损坏；
   - 最终 artifact、manifest、provenance 和 Audit 全链路验证。

### 9.2 核心指标

| 维度 | 指标 |
|---|---|
| Capability | false retain、false drop、metric compatibility、invalid field rate |
| Evidence | citation precision、evidence coverage、query recovery、mapping coverage |
| 产品质量 | hard-audit rejection、semantic flags、人工修订轮数、最终接受率 |
| Agent 行为 | valid tool-call rate、重复调用率、budget exhaustion、needs-human precision |
| 运行可靠性 | resume success、duplicate artifact/item、checkpoint/artifact mismatch detection |
| 成本 | model calls、tool calls、input/output tokens、wall time、峰值私有 state 大小 |
| 安全 | 越权工具调用、敏感字段泄漏、未选择来源访问、未验证 memory write |

每次修改模型、prompt、工具描述、tool schema、reranker 或 memory policy 都运行同一 eval suite。不能用单个
成功 demo 作为阶段完成标准。

## 10. 安全与治理边界

- 模型只有只读研究工具和“提交候选”工具；提交成功不等于已写入 Blueprint。
- 所有业务写入由确定性节点在验证和 HITL 后执行。
- 不提供 shell、Python、任意 SQL、任意 URL、任意文件路径或通用文件读取。
- 工具依据 state 注入允许 dataset、task、field 和 material 集合，模型参数不能扩大权限。
- 来自 PDF/Web 的文本只作为数据，不得改变 system/tool policy。
- 网络能力默认关闭；开启时使用 allowlist、下载大小、类型、重定向和内容 hash 限制。
- 所有模型输出在改变稳定 artifact 前经过现有或新增代码 validator。
- 长期记忆可查看、可删除、按内容 hash 失效；不得保存 raw chain-of-thought。
- Agent 达到预算、遇到来源冲突或需要删除已确认任务时请求人工决定，不自行扩大目标。

## 11. 明确不做的事情

- 不把五个现有角色包装成互相对话的“人格 Agent”。
- 不为了展示 ToolNode 而把确定性函数逐个包装成工具。
- 不让模型自由决定是否跳过 Generation、Audit 或 finalize。
- 不让 Agent 直接写 Blueprint、raw benchmark、provenance 或 manifest。
- 不把所有历史消息和 artifact 塞进向量库当作“记忆”。
- 不在没有对照 eval 的情况下默认开启多 Agent、并行研究或开放 Web。
- 不以“模型有 reasoning 输出”替代可验证的工具观察和最终契约。

## 12. 产品完成定义

当以下条件全部满足时，可以认为 MedBenchAgent 已从固定 LLM workflow 演进为可用的自主研究 Agent
产品：

1. 用户确认目标和 Research Plan 后，系统能在无需逐 task 人工指导的情况下完成字段能力研究和医学
   证据研究。
2. 在需要补充观察的评测 case 中，系统会根据 tool observation 改变后续工具或 Task Plan/design；
   ReAct 不是固定调用序列。
3. 所有 accepted 方案由现有代码 validator 接受，最终题目仍由 annotation witness 唯一决定。
4. Agent 能明确停止为 `accepted`、`needs_human` 或失败原因，不会无限循环或静默降级。
5. 运行可跨进程恢复，恢复不产生重复 artifact、题目或 provenance。
6. 用户能看到计划、工具进度、引用和风险，并能暂停、修改或取消。
7. 长期记忆只复用已验证事实，数据或材料版本变化后自动失效。
8. 与 `workflow` 基线相比，研究质量或人工修订成本有量化改善，且 token/延迟代价可接受。
9. Audit、数据访问、隐私和可回放边界不因 Agent 化而削弱。

这一定义对应一个领域专用、证据驱动、可恢复且可监管的 benchmark research agent；它具有真实工具
选择和反馈循环，但将自主性限制在能产生实际产品价值的位置。
