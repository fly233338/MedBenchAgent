# MedBenchAgent 架构

本文是压缩上下文或开启新任务后的架构恢复入口，只记录稳定事实与边界。执行规范见
`AGENTS.md`；若本文与实现冲突，以代码契约为准，并在同一次改动中修正文档。

## 项目定位

MedBenchAgent 从自然语言需求、目录注册的图像级 `annotation.jsonl` 和医学 PDF 出发，生成
Medical VLM benchmark 题目。当前系统实际是 annotation-driven：不读取图像内容，也不调用
VLM；`image_path` 仅作为不透明元数据随题目输出，题目事实必须来自获准的 annotation 字段。

## 主流程

LangGraph 入口是 `medbenchagent.workflow.graph:benchmark_graph`。根图采用固定的
role-specialized agents + deterministic orchestration：

`run_initialize → SpecificationAgent → CapabilityAgent → ClinicalDesignAgent → rag_design_export → GenerationAgent → generation_summary → AuditAgent → run_finalize`

各角色内部保留自己的确认、修订和逐项处理循环，根图不做动态路由：

1. `SpecificationAgent` 在首次 DeepSeek 调用前取得已预分配的 run 路径，将自然语言需求提取为 BenchmarkInfo，经人工确认后创建 working Blueprint。
2. `CapabilityAgent` 从 `datasets/dataset_catalog.json` 推荐并确认数据集。
   代码解析路径、校验 `annotation.jsonl`，再生成
   内容寻址的 DuckDB、dataset profile 和 access manifest。
3. `CapabilityAgent` 根据 profile 为全部预定义任务生成 Task Plan；人工可逐项修订。确认后，代码删除
   不支持的任务并写入 `supporting_fields`，同时保留完整 `task_plan_report.json`。
4. `CapabilityAgent` 按任务 category 稳定加权分配样本数：仅在保留类别间归一化默认权重，以最大余数法分配类别整数总数，再在类别内均分任务；自然语言调整由模型转为 JSON，经校验和确认后写入 Blueprint。
5. `ClinicalDesignAgent` 包装现有 `rag_designs` 子图，对每个非零任务使用医学 PDF 做 RAG，逐项把设计和引用写入 Blueprint，再由
   人工整体确认或定向重做。检索使用本地 Qwen embedding/reranker 与持久化 Chroma 缓存。
6. `GenerationAgent` 在私有状态中从确认后的 Blueprint 构造 `GenerationRequest`。现有生成子图按 letter、bbox、free_text 三个阶段
   独立读取 DuckDB；每阶段仅投影任务获准字段、稳定随机排序候选并逐任务生成/审核，写入各自可恢复进度文件。
7. 完成或数据耗尽后生成 raw JSONL；根图先记录 generation summary，再由 `AuditAgent` 硬审计图像完整性、题目契约、annotation witness、provenance 和重复项，平衡通过的选择题位置，最后按 task 稳定抽样进行 fail-open 的宽松 LLM 语义筛查。

## 状态、契约与产物

根图以 `BenchmarkInputState` 接收可选的 `messages`、`user_requirement`、`seed`、
`max_tasks_per_image` 和 `question_temperature`。`BenchmarkState` 只保存跨 Agent 输入、
运行上下文、紧凑摘要和稳定产物路径，不保存任何角色的 draft、feedback、confirmed、queue、
cursor、`GenerationRequest` 或数据集行；`BenchmarkOutputState` 则只返回 BenchmarkInfo、选择的数据集
ID、Blueprint/Task Plan 报告/raw 路径，以及最终题目计数、任务计数和 shortfall。

五个角色的 Input、Output 和私有状态 contract 位于 `agents/contracts.py`，领域 contract 仍只定义在
顶层 `contracts.py`。`ClinicalDesignAgent` 内的 `rag_designs` 子图以 Blueprint 路径、BenchmarkInfo、已选资料和 dataset profile 为输入；模型可按需请求
当前 supporting field 的安全 `top_values` 摘要，DICOM metadata、bbox 和原始 annotation 行不会进入提示。
其任务队列、定向重做反馈、当前任务和完成计数只保存在该 Agent checkpoint。GenerationAgent 先在私有状态中构造 request，现有生成子图仍以只包含 `generation_request` 的输入契约开始，
并只向根图返回 raw 路径、题目计数、任务计数和 shortfall。阶段、游标、目标/完成计数、progress 路径、
拒绝计数和耗尽标记属于 GenerationAgent 私有 checkpoint，不进入根图状态或公共输出。稳定类型位于 `contracts.py`：`BlueprintTask`（包含稳定 `category`）是规划到生成
的任务快照，`GenerationRequest` 是生成边界，`BenchmarkItem` 是输出题目契约。输入记录契约由
`schemas/annotation.v1.schema.json` 定义。

每次运行的核心产物是 working `blueprint.json`、`task_plan_report.json`、
阶段临时 `benchmark_items.*.progress.jsonl` 和 `benchmark.<dataset_id>.raw.jsonl`；raw 写入成功后删除临时 progress 文件。运行目录还输出 `item_provenance.jsonl`、`audit_records.jsonl`、`semantic_audit_records.jsonl`、只含元数据的 append-only `construction_events.jsonl`、事件聚合的 `construction_report.json`、确认后的 `rag_designs.json`、非敏感的 `effective_config.json` 和 `benchmark_manifest.json`。配置按代码回退值、`config/default.toml`、显式 `MEDBENCH_CONFIG` 和非空环境变量依次覆盖；Graph 的三个 generation 输入只覆盖本次运行。快照记录生效值、配置源哈希和覆盖键，不记录密钥。Blueprint 的 `dataset` 仅含 `dataset_id`；catalog 名称/描述、annotation hash、profile/cache 版本、图像数、image manifest hash 和固定 `8bit_grayscale_png` 格式只写入最终 manifest。manifest 中路径相对运行目录，除自身外记录 SHA-256；`benchmark_id` 是最终 BenchmarkInfo、dataset ID 及任务定义的规范化内容哈希，和 `run_id` 无关。数据集缓存按 annotation 内容哈希与缓存版本隔离，包含
`dataset.duckdb`、`dataset_profile.json` 和 `dataset_access_manifest.json`。

## 不变量与失败语义

- 语义型用户输入和审核先由 DeepSeek 转为受限 JSON，再由代码验证；RAG 前固定扫描 `materials/` 下
  分类目录中的 PDF，用户自然语言选择的精确相对路径经验证后才参与检索。
- 数据集目录路径不进入推荐提示或人工审核；生成提示只接收当前任务的获准 annotation 投影。
- Task Plan 与样本分配只在确认后改变 Blueprint；RAG 设计先逐项持久化，再通过确认/修订闭环收敛。
- 同一 task/image 最多生成一题；默认每阶段每图最多接受三题。确定性 ID 和幂等合并保证 checkpoint 与文件状态可安全重放。
- letter VQA 除 question/answer 外临时返回一个由 supporting-field path、每层原数组索引和原样 scalar value 组成的 evidence witness；代码在 Reviewer 前精确验证，raw item 不保留 witness，provenance 只保存 path、indices 和覆盖完整 witness 的 hash。
- bbox 仅接受恰有一个合法标注框的候选；`visual_grounding` 还必须有与该框配对的 finding 文本，并确定性优先于 `detection`。`detection` 仅定位泛化 lesion/abnormal region，不能泄露 finding 名称；bbox 与自由文本保持各自的直接 annotation 答案契约。
- 数据耗尽是带 per-task shortfall 的成功结果；答案平衡不覆盖 raw 文件，也不改 bbox/自由文本题。
- 根图默认只展示五个角色及确定性适配节点；xray 图、LangSmith 与使用
  `stream_mode=["updates", "custom"]`、`subgraphs=True` 仍可展开角色子图并接收每个 RAG 任务与每条生成记录的进度。启用 LangSmith 后，DeepSeek 调用显示为所属 Agent 下可展开的 LLM span。
- `BenchmarkItem` 包含稳定的 `dataset` 字段。provenance 只覆盖写入 raw 的 reviewer-approved item；失败和拒绝记录在 construction events。LLM 事件使用五个固定 Agent 名称并记录 operation、模型逻辑名称、token 与 latency；`AuditAgent` 只有抽样语义筛查产生 LLM 调用，硬审计仍为确定性。人类事件仅在自然语言被成功解析并通过代码校验后记录，不保存输入内容。
- `run_started` 和 `run_completed` 使用 `orchestrator` component，不归属任何业务 Agent，也不进入五角色的 LLM 统计。
- Audit 不重新判断医学事实或调用视觉模型；raw 保持不可变。硬审计使用 source row 回溯同一 annotation 文件并重算 letter witness hash，随后平衡答案位置。语义层每 task 最多稳定抽取十题，只宽松检查明显任务偏移、答案泄露、歧义/选项重叠及语义近重复/模板塌缩；失败时 fail-open，flag 不会 reject，`flagged.jsonl` 始终是 `audited.jsonl` 的观察性子集。`audit_report.json` 采用 `audit-report.v1`，分别在 `hard_audit` 与 `semantic_audit` 中表达两层计数。PNG hash 分块读取且以 Pillow extrema 检查非恒定灰度像素；重复 item ID 和完整 canonical item 会拒绝，重复问题只写入统计。

## 维护边界

修改图流程、稳定类型、输入输出契约、产物名称、模型职责或数据可见边界时，必须同步更新本文。
`README.md` 面向使用方式；`AGENTS.md` 面向工作约束；本地旧 SVG/Draw.io 和历史运行产物均不是
当前架构真相源。本文不描述部署拓扑、威胁模型或未来规划。
