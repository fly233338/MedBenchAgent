# MedBenchAgent 架构

本文是压缩上下文或开启新任务后的架构恢复入口，只记录稳定事实与边界。执行规范见
`AGENTS.md`；若本文与实现冲突，以代码契约为准，并在同一次改动中修正文档。

## 项目定位

MedBenchAgent 从自然语言需求、目录注册的图像级 `annotation.jsonl` 和医学 PDF 出发，生成
Medical VLM benchmark 题目。当前系统实际是 annotation-driven：不读取图像内容，也不调用
VLM；`image_path` 仅作为不透明元数据随题目输出，题目事实必须来自获准的 annotation 字段。

## 主流程

LangGraph 入口是 `medbenchagent.workflow.graph:benchmark_graph`，主链路为：

1. DeepSeek 将自然语言需求提取为 BenchmarkInfo，经人工确认后创建 working Blueprint。
2. 从 `datasets/dataset_catalog.json` 推荐并确认数据集；根目录 `dataset_catalog.json` 仅作为模板。
   代码解析路径、校验 `annotation.jsonl`，再生成
   内容寻址的 DuckDB、dataset profile 和 access manifest。
3. DeepSeek 根据 profile 为全部预定义任务生成 Task Plan；人工可逐项修订。确认后，代码删除
   不支持的任务并写入 `supporting_fields`，同时保留完整 `task_plan_report.json`。
4. 代码按任务 category 稳定加权分配样本数：仅在保留类别间归一化默认权重，以最大余数法分配类别整数总数，再在类别内均分任务；自然语言调整由模型转为 JSON，经校验和确认后写入 Blueprint。
5. `rag_designs` 子图对每个非零任务使用医学 PDF 做 RAG，逐项把设计和引用写入 Blueprint，再由
   人工整体确认或定向重做。检索使用本地 Qwen embedding/reranker 与持久化 Chroma 缓存。
6. 从确认后的 Blueprint 构造 `GenerationRequest`。生成子图按 letter、bbox、free_text 三个阶段
   独立读取 DuckDB；每阶段仅投影任务获准字段、稳定随机排序候选并逐任务生成/审核，写入各自可恢复进度文件。
7. 完成或数据耗尽后生成 raw JSONL；选择题位置平衡由独立 CLI 手工执行。

## 状态、契约与产物

根图以 `BenchmarkInputState` 接收可选的 `messages`、`user_requirement`、`seed`、
`max_tasks_per_image` 和 `question_temperature`。`BenchmarkState` 是渐进填充的内部 checkpoint 状态，只保存
紧凑摘要和产物路径，不保存数据集行；`BenchmarkOutputState` 则只返回 BenchmarkInfo、选择的数据集
ID、Blueprint/Task Plan 报告/raw 路径，以及最终题目计数、任务计数和 shortfall。

`rag_designs` 子图以 Blueprint 路径、BenchmarkInfo、已选资料和 dataset profile 为输入；模型可按需请求
当前 supporting field 的安全 `top_values` 摘要，DICOM metadata、bbox 和原始 annotation 行不会进入提示。
其任务队列、定向重做反馈、当前任务和完成计数只保存在子图 checkpoint。生成子图以只包含 `generation_request` 的输入契约开始，
并只向根图返回 raw 路径、题目计数、任务计数和 shortfall。阶段、游标、目标/完成计数、progress 路径、
拒绝计数和耗尽标记属于生成子图私有 checkpoint，不进入根图状态或公共输出。稳定类型位于 `contracts.py`：`BlueprintTask`（包含稳定 `category`）是规划到生成
的任务快照，`GenerationRequest` 是生成边界，`BenchmarkItem` 是输出题目契约。输入记录契约由
`schemas/annotation.v1.schema.json` 定义。

每次运行的核心产物是 working `blueprint.json`、`task_plan_report.json`、
阶段临时 `benchmark_items.*.progress.jsonl` 和 `benchmark_items.raw.jsonl`；raw 写入成功后删除临时 progress 文件。数据集缓存按 annotation 内容哈希与缓存版本隔离，包含
`dataset.duckdb`、`dataset_profile.json` 和 `dataset_access_manifest.json`。

## 不变量与失败语义

- 语义型用户输入和审核先由 DeepSeek 转为受限 JSON，再由代码验证；RAG 前固定扫描 `materials/` 下
  分类目录中的 PDF，用户自然语言选择的精确相对路径经验证后才参与检索。
- 数据集目录路径不进入推荐提示或人工审核；生成提示只接收当前任务的获准 annotation 投影。
- Task Plan 与样本分配只在确认后改变 Blueprint；RAG 设计先逐项持久化，再通过确认/修订闭环收敛。
- 同一 task/image 最多生成一题；默认每阶段每图最多接受三题。确定性 ID 和幂等合并保证 checkpoint 与文件状态可安全重放。
- bbox 仅接受恰有一个合法标注框的候选；`visual_grounding` 还必须有与该框配对的 finding 文本，并确定性优先于 `detection`。`detection` 仅定位泛化 lesion/abnormal region，不能泄露 finding 名称；选择题、bbox 与自由文本均使用独立的 question/answer 生成和审核契约。
- 数据耗尽是带 per-task shortfall 的成功结果；答案平衡不覆盖 raw 文件，也不改 bbox/自由文本题。
- RAG 与生成是直接挂载在根图上的编译子图，因此 xray 图、LangSmith 与使用 `stream_mode=["updates", "custom"]` 和
  `subgraphs=True` 可接收每个 RAG 任务与每条生成记录的进度；启用 LangSmith 后，DeepSeek 调用显示为
  对应子图内可展开的 LLM span。

## 维护边界

修改图流程、稳定类型、输入输出契约、产物名称、模型职责或数据可见边界时，必须同步更新本文。
`README.md` 面向使用方式；`AGENTS.md` 面向工作约束；本地旧 SVG/Draw.io 和历史运行产物均不是
当前架构真相源。本文不描述部署拓扑、威胁模型或未来规划。
