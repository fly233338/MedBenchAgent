# MedBenchAgent 架构

本文是压缩上下文或开启新任务后的架构恢复入口，只记录稳定事实与边界。执行规范见
`AGENTS.md`；若本文与实现冲突，以代码契约为准，并在同一次改动中修正文档。

## 项目定位

MedBenchAgent 从自然语言需求、目录注册的图像级 `annotation.jsonl` 和医学 PDF 出发，生成
Medical VLM benchmark 题目。当前系统实际是 annotation-driven：不读取图像内容，也不调用
VLM；`image.image_path` 仅作为不透明元数据随题目输出，题目事实必须来自获准的实体字段。
独立的 `medbenchagent.evaluation` 子系统在构建完成后才读取审计 PNG 和 benchmark 题目，
用于本地 Medical VLM 推理与指标统计，不进入 LangGraph 构建流程或 Agent 状态。

## 主流程

LangGraph 入口是 `medbenchagent.workflow.graph:benchmark_graph`。根图采用固定的
role-specialized agents + deterministic orchestration：

`run_initialize → SpecificationAgent → CapabilityAgent → ClinicalDesignAgent → rag_design_export → GenerationAgent → generation_summary → AuditAgent → run_finalize`

各角色内部保留自己的确认、修订和逐项处理循环，根图不做动态路由：

1. `SpecificationAgent` 在首次 DeepSeek 调用前取得已预分配的 run 路径，将自然语言需求提取为 BenchmarkInfo，经人工确认后创建 working Blueprint。
2. `CapabilityAgent` 从 `datasets/dataset_catalog.json` 推荐并确认数据集。
   代码解析路径、校验 `annotation.jsonl`，再生成
   内容寻址的 DuckDB、dataset profile 和 access manifest。
3. `CapabilityAgent` 根据 profile 为每个预定义任务顺序调用一次模型生成 Task Plan entry，再聚合为完整
   Task Plan；随后执行一次轻量的全局一致性复核，删除明显重复或明显不受支持的保留任务并重写必要的 reason，
   再交给人工逐项修订。全局复核不能新增 task、字段、question type 或 metric。每个保留的最终任务同时选择一个由共享 metric registry 提供、且与
   `response_format` 兼容的主指标。所有保留的 letter 任务必须有至少两个语义答案类别，代码以获准字段的
   `distinct_value_count >= 2` 作为必要条件兜底；`finding_presence` 表示在没有直接 finding 类别标签时对指定 finding
   做二元 presence/absence 判断，且必须同时具有显式正例和负例。
   `anatomy_attribute` 与 `finding_attribute` 按单一语义答案轴识别 question type，每个类型只使用一个字段，并展开为
   `<parent_task_id>.<question_type_id>` 独立任务；每个展开任务独立保存 metric。确认后，代码删除不支持的任务并写入
   `supporting_fields` 与 metric；首轮逐任务输出经过全局复核后写入 `task_plans/v1.json`，每轮人工修订写入递增的 `vN.json`，不覆盖历史版本。
4. `CapabilityAgent` 按任务 category 稳定加权分配样本数：仅在保留类别间归一化默认权重，以最大余数法分配类别整数总数；类别内先在原始父任务之间分配，再在父任务的有效 question types 之间均分，因此拆分不会提高父任务权重。自然语言调整由模型转为 JSON，经校验和确认后写入 Blueprint。
5. `ClinicalDesignAgent` 包装现有 `rag_designs` 子图，对每个非零任务使用医学 PDF 做 RAG，逐项把设计和引用写入 Blueprint，再由
   人工整体确认或定向重做。新设计为每个具体 question type 生成配置数量的唯一、语义等价模板，默认三个；letter 任务先请求一个获准字段的安全精确分类值摘要，再生成结构化 `answer_field` 与 `answer_value_mapping`。检索使用本地 Qwen embedding/reranker 与持久化 Chroma 缓存。
6. `GenerationAgent` 在私有状态中从确认后的 Blueprint 构造 `GenerationRequest`。现有生成子图按 letter、bbox、free_text 三个阶段
   独立读取 DuckDB；每阶段仅投影任务获准字段、稳定随机排序候选，并按 seed、task、image 和 source row
   确定性选择一个问题模板后逐任务生成/审核。三种格式都把 `medical_task_description`、`generation_logic`
   和选定模板传入生成与 Reviewer prompt；答案和 evidence 均由代码根据获准 annotation/reference 固定，LLM 只生成题干。
7. 完成或数据耗尽后生成 raw JSONL；根图先记录 generation summary，再由 `AuditAgent` 硬审计图像完整性、题目契约、annotation witness、provenance 和重复项，平衡通过的选择题位置，最后按 task 稳定抽样进行 fail-open 的宽松 LLM 语义筛查。

## 状态、契约与产物

根图以 `BenchmarkInputState` 接收可选的 `messages`、`user_requirement`、`seed`、
`max_tasks_per_image` 和 `question_temperature`。`BenchmarkState` 只保存跨 Agent 输入、
运行上下文、紧凑摘要和稳定产物路径，不保存任何角色的 draft、feedback、confirmed、queue、
cursor、`GenerationRequest` 或数据集行；`BenchmarkOutputState` 则只返回 BenchmarkInfo、选择的数据集
ID、Blueprint/Task Plan 报告/raw 路径，以及最终题目计数、任务计数和 shortfall。

五个角色的 Input、Output 和私有状态 contract 位于 `agents/contracts.py`，领域 contract 仍只定义在
顶层 `contracts.py`。`ClinicalDesignAgent` 内的 `rag_designs` 子图以 Blueprint 路径、BenchmarkInfo、已选资料、dataset profile 和 access manifest 为输入。初始 profile 只包含安全的稳定样例；设计 letter mapping 时，模型可按需选择一个获准字段，代码再从 DuckDB 返回最多 32 个高频精确 scalar、计数、总 distinct count 和截断状态。字符串最长 200 字符，identifier、文件引用、DICOM metadata、bbox 坐标、长自由文本和原始数据行不会进入提示。
其任务队列、定向重做反馈、当前任务和完成计数只保存在该 Agent checkpoint。GenerationAgent 先在私有状态中构造 request，现有生成子图仍以只包含 `generation_request` 的输入契约开始，
并只向根图返回 raw 路径、题目计数、任务计数和 shortfall。阶段、游标、目标/完成计数、progress 路径、
拒绝计数和耗尽标记属于 GenerationAgent 私有 checkpoint，不进入根图状态或公共输出。稳定类型位于 `contracts.py`：`BlueprintTask`（包含稳定 `category`）是规划到生成
的任务快照，`GenerationRequest` 是生成边界，`BenchmarkItem` 是输出题目契约。输入记录契约由
`schemas/annotation.v2.schema.json` 定义。每行仍代表一张图，顶层为 `patient`、`case`、`study`
和 `image`；各实体拥有独立 annotation，任务字段路径与该层级保持一致。

每次运行的核心产物是 working `blueprint.json`、版本化的 `task_plans/vN.json`、
阶段临时 `benchmark_items.*.progress.jsonl` 和 `benchmark.<dataset_id>.raw.jsonl`；raw 写入成功后删除临时 progress 文件。运行目录还输出 `item_provenance.jsonl`、`audit_records.jsonl`、`semantic_audit_records.jsonl`、append-only `construction_events.jsonl`、事件聚合的 `construction_report.json`、首轮完整设计及每轮人工修订均追加的 `rag_designs/vN.json`、非敏感的 `effective_config.json` 和 `benchmark_manifest.json`。construction events 除 LLM 元数据外还保存每轮完整、未脱敏的用户自然语言输入；初轮 artifact snapshot 记录版本路径与 SHA-256，修订事件以同一 interaction ID 关联用户输入、实际变更对象及新 artifact 路径与 SHA-256，因此运行目录可能含敏感信息。配置按代码回退值、`config/default.toml`、显式 `MEDBENCH_CONFIG` 和非空环境变量依次覆盖；Graph 的三个 generation 输入只覆盖本次运行。快照记录生效值、配置源哈希和覆盖键，不记录密钥。Blueprint 的 `dataset` 仅含 `dataset_id`；catalog 名称/描述、annotation hash、profile/cache 版本、图像数、image manifest hash 和固定 `8bit_grayscale_png` 格式只写入最终 manifest。manifest 中路径相对运行目录，除自身外记录 SHA-256；`benchmark_id` 是最终 BenchmarkInfo、dataset ID 及任务定义的规范化内容哈希，和 `run_id` 无关。数据集缓存按 annotation 内容哈希与缓存版本隔离，包含
`dataset.duckdb`、`dataset_profile.json` 和 `dataset_access_manifest.json`。确认后的设计采用
`rag-designs.v2`；manifest 的展开任务记录可选 `parent_task_id` 与 `question_type_id`。模板数量由配置项
`generation.question_template_count` 控制并写入生效配置快照，不属于 Graph 输入覆盖项。
RAG 设计中的 `medical_task_description` 保持为“先描述评估能力，再说明医学依据”的一段话；
`generation_logic` 是一条语义完整的指令句，描述如何利用获准标注语义和选定题目模板生成题目。

## 不变量与失败语义

- 语义型用户输入和审核先由 DeepSeek 转为受限 JSON，再由代码验证；RAG 前固定扫描 `materials/` 下
  分类目录中的 PDF，用户自然语言选择的精确相对路径经验证后才参与检索。
- 数据集目录路径不进入推荐提示或人工审核；生成提示只接收当前任务的获准字段投影。
- 初始 Blueprint 模板不预填 metric；Task Plan 与样本分配只在确认后改变 Blueprint，确认时才写入每个保留最终任务的 metric。Task Plan 和 RAG 设计的每个审核版本均作为不可变 JSON 保存，RAG 设计仍逐项写入 Blueprint 后通过确认/修订闭环收敛。
- 同一 task/image 最多生成一题；默认每阶段每图最多接受二十题。确定性 ID 和幂等合并保证 checkpoint 与文件状态可安全重放。
- 三种答案格式都保留可回放的 annotation witness。letter 任务由代码从 `answer_field` 唯一定位原始 scalar，并通过 `answer_value_mapping` 固定正确 option；free_text 由代码唯一定位直接答案字符串；bbox 由代码按 xyxy 顺序定位同一字段及共同外层索引下的四个原始坐标。三种格式的 LLM 都只生成题干，代码在 Reviewer 前精确固定并验证答案与 evidence；raw item 不保留 witness，provenance 只保存 path、indices 和覆盖原始值的 hash。
- bbox 任务只接受恰有一个合法标注框且能唯一定位四坐标 witness 的候选，不接受 mask 或运行期
  mask-to-bbox 转换。`visual_grounding` 还必须有与该框配对的 finding 名称、结构化标签或描述；
  `detection` 仅定位泛化 lesion/abnormal region，不能泄露 finding 名称。两者是独立能力，可同时保留。
  bbox 与自由文本保持各自的直接 annotation 答案契约。
- `diagnosis` 允许影像或病理确认的具体诊断作为 reference；letter VQA 仅接受能由
  `answer_field` 和 `answer_value_mapping` 唯一确定正确 option 的候选。
- 数据耗尽是带 per-task shortfall 的成功结果；答案平衡不覆盖 raw 文件，也不改 bbox/自由文本题。
- 根图默认只展示五个角色及确定性适配节点；xray 图、LangSmith 与使用
  `stream_mode=["updates", "custom"]`、`subgraphs=True` 仍可展开角色子图并接收每个 RAG 任务与每条生成记录的进度。启用 LangSmith 后，DeepSeek 调用显示为所属 Agent 下可展开的 LLM span。
- 内网交互入口 `medbenchagent build` 在当前进程中用 `MemorySaver` 编译同一根图，从终端收集
  requirement 和每次自然语言确认，并消费 `tasks`、`updates`、`custom` 子图事件展示进度。
  该入口不改变 Graph 契约或产物，强制关闭可选外部 tracing、模型下载与遥测；checkpoint
  仅在进程存活期间有效。
- `BenchmarkItem` 包含稳定的 `dataset` 字段。provenance 只覆盖写入 raw 的 reviewer-approved item；失败和拒绝记录在 construction events。LLM 事件使用五个固定 Agent 名称并记录 operation、模型逻辑名称、token 与 latency；`AuditAgent` 只有抽样语义筛查产生 LLM 调用，硬审计仍为确定性。每轮用户输入写为独立 `human_interaction`，包括解析失败的输入；修订完成后以同一 `interaction_id` 写入 `revision_applied`、实际变化对象及对应版本 artifact 的相对路径和 SHA-256。`construction-report.v2` 按 stage 汇总最终确认状态、成功修订轮数和实际修改对象数，materials 另计选择数量。
- `run_started` 和 `run_completed` 使用 `orchestrator` component，不归属任何业务 Agent，也不进入五角色的 LLM 统计。
- Audit 不重新判断医学事实或调用视觉模型；raw 保持不可变。硬审计使用 source row 回溯同一 annotation 文件并重算所有答案格式的 witness hash；free_text 必须与回放字符串完全一致，bbox 必须与四个回放坐标数值一致。letter 任务还必须满足回放 raw value、mapping、语义 option 与最终答案字母一致。随后仅平衡选择题答案位置。语义层每 task 最多稳定抽取十题，只宽松检查明显任务偏移、答案泄露、歧义/选项重叠及语义近重复/模板塌缩；失败时 fail-open，flag 不会 reject，`flagged.jsonl` 始终是 `audited.jsonl` 的观察性子集。`audit_report.json` 采用 `audit-report.v1`，分别在 `hard_audit` 与 `semantic_audit` 中表达两层计数。PNG hash 分块读取且以 Pillow extrema 检查非恒定灰度像素；重复 item ID 和完整 canonical item 会拒绝，重复问题只写入统计。

## 推理与评测边界

评测入口显式接收 benchmark JSONL、`benchmark_manifest.json` 和数据集旁的
`image_manifest.jsonl`。加载器依据共享 metric registry 验证三者哈希、task/response-format/metric 契约以及被引用 PNG
的路径、尺寸和哈希，只通过 `image_id` 找图，不解释 `image_path`。当前稳定指标映射为
letter 的 accuracy/macro-F1、free_text 的 SacreBLEU、bbox 的 mIoU。

模型通过显式注册的 `ModelAdapter` 加载；每个具体模型拥有独立模块，共用族级实现。目前支持
Qwen2.5-VL、InternVL3、Qwen3-VL Instruct/Thinking、Lingshu、Hulu-Med、QoQ-Med-VL 和
Chiron-o1。Qwen2.5/Qwen3 以处理后图像尺寸解释绝对 bbox，InternVL3/Chiron 以固定
`[0,1000]` 坐标空间解释 grounding，Hulu 以原 PNG 尺寸解释。letter VQA 的 system prompt 只要求
返回正确选项的单个字母，user content 只含图像及题干与选项，不提供答案或输出格式示例；解析器也
只接受裸单字母响应。free-text VQA 不设置 system message，直接把完整非空终答作为预测；所有 bbox
任务统一由 system 要求仅输出带唯一 `bbox` 字段的绝对 xyxy JSON 对象。三类 user content 都只含
图像和题目（letter 另含选项），所有 adapter 复用同一 prompt 函数。

`ModelOutput` 同时保留完整 `raw_text`、可选模型坐标空间和可选 `answer_text`。解析器仅在 adapter
按模型官方协议提取终答时优先使用 `answer_text`：Qwen3 Thinking 使用最后一个 `</think>`，Chiron
使用最后一个 `### The final answer is:`，Hulu 对同一 output IDs 分别保留 thinking 与无 thinking
解码。分隔符缺失时回退完整响应，空终答保持为空并产生严格 parse error；推理产物只把完整响应写入
`raw_prediction`，wire schema 仍是 `medbench-inference.v1`。

`torchrun` 单节点每 GPU 启动一个 rank，每个 rank 独立加载一份完整模型，以轮转切片和普通 batch
循环推理，不使用 DDP 或 tensor parallel。InternVL 族在 adapter batch 内按公共 system prompt 分组
调用 `batch_chat` 后恢复输入顺序；Hulu 因 processor 限制在 batch 内逐样本执行。rank 临时 JSONL
全部成功后，rank 0 按输入顺序写出自包含推理 JSON；中断不自动续跑。evaluation 子系统的 JSON
写入工具不依赖 benchmark 构建运行时，因此可在只安装 `requirements/evaluation.txt` 的独立 Conda
环境中通过源码 `PYTHONPATH` 运行。

`medbench-evaluation.v1` 从推理 JSON 重新计算逐题结果、逐 task 主指标、同指标 task-macro
以及 category/全局阈值正确率。选择题 macro-F1 使用选项文本而非平衡后的字母位置；
不同指标不合成为单一总分。评测产物不修改 benchmark run 或其 manifest。

## 维护边界

修改图流程、稳定类型、输入输出契约、产物名称、模型职责或数据可见边界时，必须同步更新本文。
`README.md` 面向使用方式；`AGENTS.md` 面向工作约束；本地旧 SVG/Draw.io 和历史运行产物均不是
当前架构真相源。本文不描述部署拓扑、威胁模型或未来规划。
