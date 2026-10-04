# MedBenchAgent 架构

MedBenchAgent 用 LangGraph 协调医疗数据准备和 benchmark 构建。Supervisor 接收完整的自然语言请求，根据当前产物和可用工具选择下一步；数据处理、规划、生成和评测保持独立边界。

## 运行流程

一次请求对应一个可恢复的 Run。Supervisor 可以在同一 Run 内：

1. 检查 Catalog、已登记数据集和配置来源；
2. 通过 `DataProcessingAgent` 准备或复用数据；
3. 提取 benchmark brief，选择数据集；
4. 为每个数据集生成并审核完整 Task Plan；
5. 全局分配样本并生成问题；
6. 由 `FinalizationAgent` 验证并发布 delivery。

缺少用户事实时，当前 Agent 通过 `ask_user` 中断；回答写入同一 Run 的 checkpoint 后继续。数据集选择也使用同一机制：先冻结候选，再用英文问题让用户选择，回答后由 Supervisor 映射到候选 ID 并由代码校验绑定。与系统无关的请求使用 Supervisor 的 `end` 工具结束。

## 模块边界

- `runtime/`：会话、Run、checkpoint、事件、Agent harness、依赖和运行产物。
- `workflow/graph.py`：Supervisor 工具、产物就绪判断和子 Agent 委派。
- `agents/`：数据处理、任务规划、模板、分配、生成和最终化 Agent。
- `data/`：来源、annotation、图像、Profile、Card、Catalog 和确定性验证。
- `planning/`：需求提取、数据集选择、任务树、协议和样本计数。
- `generation/`、`audit/`、`rag/`、`evaluation/`：各自领域的执行和评测。

模型负责语义判断，代码负责结构校验、文件操作、产物依赖和确定性验证。共享状态只保存协调摘要和产物引用；Agent 的详细上下文写入私有状态。

## 新运行布局

新实验使用 `experiments/<experiment>/BIR/` 保存实验级 canonical 中间状态，Run 使用 `.runtime/` 保存 checkpoint、派生视图和结果引用。

```text
<run>/
  control/
    effective_config.json
    contract.json
    artifacts.json
    events.jsonl
    operations.jsonl
  inputs/
    catalog_snapshot.json
    candidate_cards/
  agents/<agent>/<scope>/
    state.json
    access.json
    evidence/
    planning outputs
  workers/<dataset>/
    inputs/ work/ outputs/ state/ logs/ conversations/
```

`operations.jsonl` 保存工具调用、参数、依赖和结果，用于幂等重放。语义产物仍按 Agent 和数据集组织，便于审阅和恢复。

## 数据准备

数据准备的 canonical 产物使用 `annotation`、`asset_manifest.jsonl` 和
`preparation_report.json`。Case grouping、Evidence ref 和 Asset 完整性由宿主确定性校验；
OpenHands Coding Session 只负责在 Worker workspace 中探索来源和编写可重复转换器。旧的
image-centric 产物不属于 数据处理运行时。

原始来源以只读方式访问。OpenHands Coding Session 在 Worker workspace 中探索来源、编写适配器并提交候选；宿主重新核验来源、locator、总体、annotation 和图像资产，再注册 Dataset、Profile、Card 和 Catalog。

Supervisor 先解析数据准备请求：请求中包含可读的原始目录时，规范化目录名后委派 `prepare_data`；缺少目录时只询问目录路径。目录清单、格式和 manifest 由 DataProcessingAgent 检查，目录不可读或无法解析数据集 ID 时才进入澄清。

有效的已登记数据默认复用。`prepare_data` 只有在用户明确要求时才使用 `rebuild`。数据源决议和准备候选绑定当前 Run，成功后才进入 benchmark 选择。

## Benchmark 构建

`specify` 生成带字段级来源证据的 Requirement，并保留开放式意图和用户提到的数据集名称。`discover_datasets` 扫描并校验 Catalog 与 Prepared 根目录，冻结候选 Card；它不替用户做选择。`select_datasets` 只绑定冻结候选 ID。随后代码生成并校验 `DatasetAccessReceipt`，模型不会决定路径、revision 或 digest。

每个 annotation 数据集由 TaskPlanningAgent 分阶段读取规划证据：Dataset Overview、Evidence Catalog、Relation Summary 和有界 Case Samples。模型按 Registry 中的 Axis 和 Axis Value 做判断；代码组合为 Task Scenario，模型只处理当前 Scenario 并提出 Leaf Task、配置数量的等价问题模板，随后确定 Task Paradigm。Capability Category 由 Registry 中的 Operation 映射派生。Screening `keep` 只表示允许进入 Leaf 探索；Leaf 阶段可以在证据充分且没有有效 Leaf 时显式排除 Scenario，所有排除理由和证据都写入 canonical plan。完整计划的 canonical 文件位于 BIR 的 `datasets/<id>/planning/`；Run 目录中的 `task_plan.json`、`task_scenarios.json` 和 `leaf_tasks.json` 是派生视图。Generation 使用冻结 seed 在模板中可复现地抽取问题，Delivery 记录 Scenario、Leaf Task、Capability Category、TaskParadigm、OutputSpec、完整性、审计和短缺统计。

Leaf design receives a bounded structured context containing the dataset
revision, Evidence Catalog, source-backed annotation semantics, documented
relations, representative Cases, coverage limitations and current Scenario
Leaf summaries. This context is carried in the planning request rather than
reconstructed from truncated observation summaries.
Read state and evidence caches are bound to dataset ID, dataset revision and
current Scenario ID; switching or recovering a Scenario invalidates stale
context. The runtime injects this dynamic context once per request.

Generation resolves model inputs in the order declared by `Resolver.inputs`.
Benchmark Items record `options`, ordered input evidence/assets and explicit
role bindings. Audit recomputes these bindings, the selected question
template, the Resolver answer and reference evidence before accepting an item.
Derived Closed VQA labels use the same deterministic scalar encoding during
planning, generation and audit.
When a Leaf needs a smaller categorical vocabulary, the Resolver may contain
an explicit `map_label` step. Unmapped values remain ineligible; no runtime
agent infers or edits mappings.

## 恢复与界面

Run 的配置、产物索引、事件日志和 Agent 状态都位于 Run 目录。GraphRunner 使用 LangGraph checkpoint 恢复；TUI 右侧显示 Run 列表、每个 Agent 的状态表和可过滤事件日志。

## 评测边界

benchmark 构建和模型推理是两个独立阶段。evaluation 读取已发布 benchmark 和图像 manifest，模型适配器、预测解析和指标计算不参与构建流程。

