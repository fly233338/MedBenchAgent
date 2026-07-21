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
2. 从 `dataset_catalog.json` 推荐并确认数据集；代码解析路径、校验 `annotation.jsonl`，再生成
   内容寻址的 DuckDB、dataset profile 和 access manifest。
3. DeepSeek 根据 profile 为全部预定义任务生成 Task Plan；人工可逐项修订。确认后，代码删除
   不支持的任务并写入 `supporting_fields`，同时保留完整 `task_plan_report.json`。
4. 代码稳定分配各任务样本数；自然语言调整由模型转为 JSON，经校验和确认后写入 Blueprint。
5. 对每个非零任务使用医学 PDF 做 RAG，逐项把设计和引用写入 Blueprint，再由人工整体确认或
   定向重做。检索使用本地 Qwen embedding/reranker 与持久化 Chroma 缓存。
6. 从确认后的 Blueprint 构造 `GenerationRequest`。生成子图逐条读取 DuckDB、仅投影任务获准
   字段，调用 DeepSeek 路由并生成题目，审核通过后写入可恢复进度文件。
7. 完成或数据耗尽后生成 raw JSONL；默认另写 final JSONL，仅平衡选择题正确答案位置。

## 状态、契约与产物

`BenchmarkState` 和 `GenerationGraphState` 只保存紧凑摘要、计数、游标及产物路径，不保存数据集
行。稳定类型位于 `contracts.py`：`BlueprintTask` 是规划到生成的任务快照，
`GenerationRequest` 是生成边界，`BenchmarkItem` 是输出题目契约。输入记录契约由
`schemas/annotation.v1.schema.json` 定义。

每次运行的核心产物是 working `blueprint.json`、`task_plan_report.json`、
`benchmark_items.progress.jsonl`、`benchmark_items.raw.jsonl` 和默认的
`benchmark_items.jsonl`。数据集缓存按 annotation 内容哈希与缓存版本隔离，包含
`dataset.duckdb`、`dataset_profile.json` 和 `dataset_access_manifest.json`。

## 不变量与失败语义

- 语义型用户输入和审核先由 DeepSeek 转为受限 JSON，再由代码验证；本地材料目录是直接路径输入。
- 数据集目录路径不进入推荐提示或人工审核；生成提示只接收当前任务的获准 annotation 投影。
- Task Plan 与样本分配只在确认后改变 Blueprint；RAG 设计先逐项持久化，再通过确认/修订闭环收敛。
- 同一 task/image 最多生成一题，确定性 ID 和幂等合并保证 checkpoint 与文件状态可安全重放。
- bbox 答案必须精确匹配唯一引用框；选择题与自由文本由独立模型审核一次。
- 数据耗尽是带 per-task shortfall 的成功结果；答案平衡不覆盖 raw 文件，也不改 bbox/自由文本题。

## 维护边界

修改图流程、稳定类型、输入输出契约、产物名称、模型职责或数据可见边界时，必须同步更新本文。
`README.md` 面向使用方式；`AGENTS.md` 面向工作约束；本地旧 SVG/Draw.io 和历史运行产物均不是
当前架构真相源。本文不描述部署拓扑、威胁模型或未来规划。
