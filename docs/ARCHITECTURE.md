# MedBenchAgent 架构

MedBenchAgent 用 LangGraph 协调医疗数据准备和 benchmark 构建。Supervisor 接收完整的自然语言请求，根据当前产物和可用工具选择下一步；数据处理、规划、生成和评测保持独立边界。

## 运行流程

一次请求对应一个可恢复的 Run。Supervisor 可以在同一 Run 内：

1. 检查 Catalog、已登记数据集和配置来源；
2. 通过 `DataProcessingAgent` 准备或复用数据；
3. 提取 benchmark brief，选择数据集；
4. 为每个数据集生成并审核完整 Task Plan；
5. 在所有计划完成后执行一次全局 Allocation；
6. 按数据集执行 Generation；
7. 由 `FinalizationAgent` 全局审计、合并、平衡选项并发布 delivery。

缺少用户事实时，当前 Agent 通过 `ask_user` 中断；回答写入同一 Run 的 checkpoint 后继续。数据集选择也使用同一机制：先冻结候选，再用英文问题让用户选择，回答后由 Supervisor 映射到候选 ID 并由代码校验绑定。与系统无关的请求使用 Supervisor 的 `end` 工具结束。

## 模块边界

- `runtime/`：会话、Run、checkpoint、事件、Agent harness、依赖、部署运行时和运行产物。
- `workflow/graph.py`：Supervisor 工具、产物就绪判断和子 Agent 委派。
- `preparation/`：来源冻结、唯一三阶段 preparation graph、context、worker 协议和宿主验收入口。
- `agents/`：任务规划、模板、分配、生成和最终化 Agent。
- `data/`：来源、annotation、图像、Profile、Card、Catalog 和确定性验证。
- `planning/`：需求提取、数据集选择、任务树、协议和样本计数。
- `generation/`、`audit/`、`rag/`、`evaluation/`：各自领域的执行和评测。

模型负责语义判断，代码负责结构校验、文件操作、产物依赖和确定性验证。共享状态只保存协调摘要和产物引用；Agent 的详细上下文写入私有状态。

## 新运行布局

新实验使用 `experiments/<experiment>/BIR/` 保存实验级 canonical 中间状态，使用
`experiments/<experiment>/outputs/` 保存 Generation 和 Finalization 的可交付产物；Run 使用
`.runtime/` 保存 checkpoint、operation ledger、Agent 私有状态和其他运行时派生视图。

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
  preparation/<dataset>/
    source snapshot / checkpoint.sqlite
    attempts/<stage>/<attempt>/
      request.json prompt.md result.json
      work/inputs/ work/outputs/ work/logs/ work/conversations/
```

`operations.jsonl` 保存工具调用、参数、依赖和结果，用于幂等重放。语义产物仍按 Agent 和数据集组织，便于审阅和恢复。

## 数据准备

唯一主线为 Understand → Prepare Text → Prepare Image → Package → Publish。
Understand 只交付 mapping 和 image_conversion_plan；mapping 明确最终图片相对路径与来源的对应关系。
Text 生成直接引用这些路径的 Cases，mask 几何直接附加到 reference Evidence。
Image 生成普通图片和唯一 image_manifest。已有 JPEG/PNG 不无谓转码，医学展示生成 PNG。
Package 完成最终 cross-check；Publisher 只按已验证 digest 原子提交，不修改数据。

Published revision 只有 cases.jsonl、image_manifest.jsonl、images/、metadata.json。
metadata 保留字段语义和需要时的 sampling/split；原始 mapping、转换计划和工作报告留在 runtime。
Card/Profile/DuckDB 在 revision 外的 .derived/ 按需重建，发布回执在 .commits/，current.json 只指向 immutable revision。
根目录不再复制数据集。详细契约见 PREPARATION_VALIDATION.md。

## Benchmark 构建

`specify` 生成带字段级来源证据的 Requirement，并保留开放式意图和用户提到的数据集名称。`discover_datasets` 扫描并校验 Catalog 与 Prepared 根目录，冻结候选 Card；它不替用户做选择。`select_datasets` 只绑定冻结候选 ID。随后代码生成并校验 `DatasetAccessReceipt`，模型不会决定路径、revision 或 digest。

每个 annotation 数据集由 TaskPlanningAgent 分阶段读取规划证据：按需查询得到的 Dataset Overview、Evidence Catalog、Relation Summary 和有界 Case Samples。模型按 Registry 中的 Axis 和 Axis Value 做判断；代码组合为 Task Scenario，模型只处理当前 Scenario 并提出 Leaf Task、配置数量的等价问题模板，随后确定 Task Paradigm。Capability Category 由 Registry 中的 Operation 映射派生。Screening `keep` 只表示允许进入 Leaf 探索；Leaf 阶段可以在证据充分且没有有效 Leaf 时显式排除 Scenario，所有排除理由和证据都写入 canonical plan。完整计划唯一 canonical 文件位于 BIR 的 `datasets/<id>/planning/task_plan.json`；不会默认写 axis/scenario/leaf/review 导出文件；消费者直接读取 Task Plan。

所有选中数据集的 Plan 完成后，Allocation 使用每个 Leaf 的 `eligible_count` 计算一个全局配额，默认先均衡数据集再均衡数据集内 Leaf。Generation 按数据集消费该配额和冻结的 Sampling manifest，并写出 JSONL。Finalization 重新审计每个生成项，隔离非法项，检查每图上限、manifest 引用和重复项，保留 shortfall，平衡 Closed VQA 选项位置，并发布统一的 `benchmark.jsonl`、`benchmark_manifest.json`、`audit_report.json` 和 `quarantine.jsonl`。

Leaf design receives a bounded structured context containing the dataset
revision, Evidence Catalog, source-backed annotation semantics, documented
relations, representative Cases, coverage limitations and current Scenario
Leaf summaries. This context is carried in the planning request rather than
reconstructed from truncated observation summaries.
Read state and evidence caches are bound to dataset ID, dataset revision and
current Scenario ID; switching or recovering a Scenario invalidates stale
context. The runtime injects this dynamic context once per request.

Generation resolves model inputs in the order declared by `Resolver.inputs` and the frozen Sampling manifest; order is never inferred from Case IDs, filenames, or dictionary order.
Benchmark Items record `options`, ordered input evidence/assets and explicit
role bindings. Audit recomputes these bindings, the selected question
template, the Resolver answer and reference evidence before accepting an item.
Derived Closed VQA labels use the same deterministic scalar encoding during
planning, generation and audit.
Reference-only Resolver steps read immutable prepared features through the frozen
Dataset Access manifest. The registered `mask_to_bbox` operation reads the
normalized bounding box embedded in mask Evidence without adding the mask to model input;
Planning, Generation and Audit share the same asset context and digest checks.
`read_mask_feature` exposes only registered geometry fields, and `argmax` requires a
manifest-frozen ordered numeric collection with ties rejected.
When a Leaf needs a smaller categorical vocabulary, the Resolver may contain
an explicit `map_label` step. Unmapped values remain ineligible; no runtime
agent infers or edits mappings.

## 恢复与界面

Run 的配置、产物索引、事件日志和 Agent 状态都位于 Run 目录。GraphRunner 使用 LangGraph checkpoint 恢复；TUI 右侧显示 Run 列表、每个 Agent 的状态表和可过滤事件日志。

## 评测边界

benchmark 构建和模型推理是两个独立阶段。evaluation 读取已发布 benchmark 和图像 manifest，模型适配器、预测解析和指标计算不参与构建流程。


## Planning Prompt Extensions

The BIR Registry freezes five complete core planning Prompts under
`registry/agent_prompts/`. The shared_planning_policy is composed once with the stage prompt. Each stage retains
clinical target semantics, Resolver constraints, examples, and output contract.
Each file has one final `{{DATASET_SPECIFIC_RULES}}` placeholder.

Optional disease or dataset extensions live under `registry/planning_extensions/`.
Supervisor freezes the available extension candidates after DatasetSelection and
asks the human which IDs to load for each dataset. Task Planning never asks an
LLM to select extensions. Selected extension Markdown is filtered by stage and
appended at the end of the complete core Prompt. An empty selection leaves the
core Prompt standalone. Extensions may tighten or clarify constraints but cannot
relax evidence grounding, Resolver validation, or output contracts.

Scenario Screening records validated clinical targets rather than speculative task directions.
Each Leaf references one `clinical_target_id` from its Scenario, and code validates the target
type, reference fields, Resolver contract, capacity, and derived plan projections.



## Ownership and task input resolution

Source discovery belongs to data; Supervisor domain arguments and dependency defaults belong to workflow.
Runtime retains execution, dependency checks, operation logs, checkpoint, interrupt and cancellation mechanics.
Planning facts live in data.task_facts; BIR owns task-plan persistence. Axis/scenario/leaf/review exports
are reproducible projections and are never read back as authority.

Planning, Generation and Audit share resolve_task_input. It binds a Case and SamplingUnit to ordered
Evidence and verified image records. eligible_count counts eligible SamplingUnits.
The benchmark item records input_images; consumers use those ordinary image files. Audit
recomputes the bindings and digests, with reference-only masks excluded from model input.
