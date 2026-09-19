# MedBenchAgent 架构

本文记录当前实现与模块边界，执行约定见 AGENTS.md。

## 项目定位与流程

项目使用 LangGraph 构建 annotation-driven Medical VLM benchmark。构建不调用视觉模型推断标签，
图像处理、benchmark 构建和 evaluation 保持独立职责。

流程：独立 Dataset 准备 → Profile/Card 注册 → Requirement → Dataset Selection →
逐 Dataset TaskPlanning → 全局 Allocation → 逐 Dataset Generation/Audit → 合并 benchmark.jsonl。
同一数据集的图像、表格等来源可以关联整理；不同数据集的 annotation 不再合并。

- `data/`：准备执行后端、算子、图像与标注验证、Profile/query、Card 和 Catalog。
- `planning/`：需求、数据集选择、Concept/Leaf 契约、Protocol 与确定性分配。
- `agents/`：数据准备、局部规划、全局 Allocation、生成和审计。
- `runtime/`：共享行动循环、工具契约、依赖、事件、运行目录与最终导出。
- `generation/`、`audit/`、`rag/`、`evaluation/` 保持领域边界。

## 准备、Card 与 Catalog

`prepare --dataset-id <id>` 通过自然语言来源描述进入 Supervisor，后者只准备这一逻辑数据集；
支持 `--resume`。`register --dataset-id <id> --bundle <path>` 验证并登记已有完整 bundle。
构建时也可以显式准备新数据集。准备版本位于 `paths.prepared_datasets_root/<dataset_id>/<version>`，
默认根目录 datasets/Prepared。准备失败不替换已登记版本。

每个 Source 有独立 Inventory、PreparationPlan、Annotation、ImageInventory/ImagePlan 和 NormalizedSource。
PreparationPlan 保存语义依据及顺序算子；不引入第二套 DAG 或表达式语言。CSV 保留字符串 ID，join 显式
定义基数、空键与未匹配策略；group_findings 保留同实体标签与 bbox 配对。标准算子无法表达时，Agent
可以记录原因并生成 Python，输出相同的 annotation 包装和 provenance 索引。

Harmonization 只用于一个 Dataset 内有依据的来源字段整理，不建立全局医学本体。未对齐字段保留
`<entity>.annotation.sources.<source_id>`；单来源空计划为 identity，缺失不补为阴性。bundle 执行
整图隔离、精确像素去重、来源与数量对账。同像素且标注/身份冲突的组整体隔离；不推断跨数据集身份。

PNG/JPEG 保留原字节；其他二维图像使用可表示原像素的 PNG 编码或有依据的显示窗。不裁框、不重采样、
不改变空间方向。多帧、真三维、金字塔及 WSI 不支持；缺 decoder 是运行能力错误，不批量当作坏图。
仅物化 annotation 引用图像。来源目录不全量复制，已用文件摘要用于发现外部修改。

发布包含 annotation.jsonl、四字段 image_manifest.jsonl、cache/Profile、来源与质量产物及 ready.json。
Profile 使用 task-capability-profile.v4，DuckDB cache 使用 annotation-cache.v8；完整值域、字段共现、
数组结构与 Protocol eligibility 通过 DatasetQuery 按需读取，统计明确分母。

Dataset Card v2 复用 Profile 的计数、字段、结构和分布；语义部分只描述数据集自身的领域、描述和字段角色。
Card 不包含 Blueprint digest 或 Concept capability hints；Blueprint 变化只影响规划，不影响已准备数据。
Card 绑定 annotation、Profile 和 image manifest 的内容，字段角色必须引用 Profile 中真实字段。

准备 Agent 可通过 save_dataset_metadata 提交来源文档的逐字引用（source_id、path、quote）。代码验证引用
来自已检查的 Source 并匹配原文；独立模型调用仅根据这些引文提取 dataset_name、dataset_description、
source_description，连同来源引用保存到 dataset_metadata.json。没有可靠来源说明时这些字段为空。
Card 生成仅接收 Profile 和该准备元数据，绝不接收 benchmark user_requirement。

`refresh-card --dataset-id <id>` 使用现有数据和 Profile，在 bundle/cards/<uuid>.json 发布新 Card 修订，
随后原子切换 Catalog 引用。旧 Card 和底层数据保持不变；刷新失败不切换注册项。v1 迁移只检查旧 Card
的确定性数据绑定，不复用其名称、描述、字段角色或能力提示；没有来源元数据时基于 Profile 重新总结。

Catalog 条目保存 dataset_id、name、description、available 和 annotation_path、image_manifest_path、
profile_path、dataset_card_path。普通构建不自动准备整个候选目录；缺少标准准备数据使用 prepare，已准备
但缺少/仍使用 v1 Card 的数据使用 refresh-card。显式指定优先、单候选直选、多候选高召回选择。

每次 Selection 在操作目录创建 dataset_candidates.json：记录实际需求与版本、名称解析所见目录、解析
结果、完整候选顺序、每个候选的 Card 摘要与内容摘要、以及不可用/未准备/未请求原因。模型看到的候选
Card 原文复制到该目录的 candidate_cards/；dataset_selection.json 同时保存冻结输入、选择方式和结果。
DatasetCandidates 是独立 artifact，Selection 依赖 Requirement、该快照和选中数据的 Card 版本。
同一操作恢复复用快照及已完成决策，不读取未选中候选的实时 Card；新 Selection 操作创建新快照。
选中数据继续核验 annotation、Profile、manifest 与冻结 Card 的一致性，外部 Card 刷新不改变运行中版本。

## 执行后端

`data_processing.backend` 为 auto（默认）、docker 或 local。auto 在准备开始前检查 Docker CLI、daemon、
配置镜像及最小启动能力；不可用时记录原因并使用本地 Python。强制 docker 时预检失败直接报错。
不自动拉取镜像或安装依赖。实际后端保存到 execution_runtime.json 和操作回执，后续操作及恢复不切换；
业务失败、超时或中途 Docker 故障不触发降级。

两个后端共用 MEDBENCH_SOURCE_ROOT、MEDBENCH_WORK_DIR、MEDBENCH_ANNOTATION_PATH、
MEDBENCH_DATA_SCRATCH 路径契约。固定操作和生成脚本都运行在子进程，保留超时、产物验证、来源检查
及两次输出一致性比较。Docker 使用只读源/代码挂载、禁网、内存/CPU/进程限制，记录实际镜像 ID。
Local 使用当前解释器、独立工作目录和 UTF-8，超时结束对应进程树；记录解释器和相关库版本。
本地生成代码具有当前用户权限，不提供挂载/网络隔离和容器内存限制。DuckDB 内存/线程配置两端通用。
本地依赖安装命令为 `uv sync --extra data-processing`，缺依赖时报告具体错误。

## 局部规划与全局分配

每次 TaskPlanning 只接收一个已选择 Dataset 的 Card/Profile/query，以及需求和 Blueprint。
Core Concept 来自静态模板，Leaf 动态定义 opaque ID，并显式保存 concept_id。不同 Dataset 的同名
Leaf 不代表同一任务。Extension 继续经过字段证据、临床意义、目标相关性及无法归入 Core 的审查，
只属于当前 Dataset，不增加跨 Dataset 语义对齐阶段。

Planner 在理解 Concept、展开 Leaf、设计 Protocol 和发现 Extension 时可查询 RAG；PDF 不补造标签。
所有 Core 有 complete/unsupported 决议后检查 coverage gap，完成 discovery，再审查局部完整树。
修正次数存入 PlanningDraft；缓存 Review、重新委派和恢复不重置额度。上游失效保留局部身份与已耗额度，
重新验证事实和审查。数据集选择、导入和题量分配不再属于 Planner。

Plan 发布要求结构、归属、Extension 决议、当前 Review、Protocol 和 eligible_count 有效。
eligible_count 由代码使用与 Generation 相同的答案解析器计算。允许没有可执行任务的完成 Plan；
全部选中 Dataset 都没有可用任务时整体 blocked。失败不能被当作 unsupported 静默跳过。

AllocationAgent 读取所有当前 Plan，模型给出有理由的 Concept 权重，可显式给出下层权重；缺省下层
使用等权。代码按 Concept → Dataset → Leaf 分层执行 largest-remainder、容量截断和重分配。
Core 能力共享，Extension 以 Dataset 作用域独立计量；预算不随 Leaf 数量自动增加。
零权重明确排除，父层容量仅累计正权重子项；取整平局使用稳定 ID 排序。Allocation 独立发布，不回写 Plan。
逐 Leaf 容量不是每图限额下的联合可行解；本阶段不进行图像匹配或自动补量循环。

## 状态、依赖与恢复

Supervisor 共享状态只保留选中 ID、各 Dataset 产物引用和进度摘要。专业 Agent 的完整 context 保存到
局部状态文件，重新委派时恢复；不整体回灌父状态。第一版串行调度，所有 Plan 完成后才可 Allocation。
零配额 Dataset 明确跳过；正配额 Dataset 的当前 Generation/Audit 是导出前置条件。

共享 Harness 使用严格 Pydantic JSON 参数和一致的工具可见/执行门控。专业 Agent 局部 ask_user 使用
LangGraph interrupt，跨职责问题返回 Supervisor。明确区分 complete、needs_input、blocked、failed。
私有观察有界，完整资料通过工具读取；人工等待不计入运行时间。超出预算 blocked。

| Artifact | 主要依赖 |
| --- | --- |
| dataset/<id>/Dataset | DataConfig；准备发布还依赖局部 Harmonization |
| dataset/<id>/DatasetProfile / DatasetCard | 对应 Dataset / Profile |
| DatasetCandidates | Requirement、冻结的候选输入文件 |
| DatasetSelection | Requirement、DatasetCandidates、选中 Card |
| dataset/<id>/Plan | Requirement、Dataset、Card、Materials、PlanningConfig、PlanningDraft |
| Allocation | Requirement、Selection、AllocationConfig、全部选中 Plan |
| dataset/<id>/Templates | Plan、TemplateConfig |
| dataset/<id>/Generation | Dataset、Plan、Templates、GenerationConfig、Allocation |
| dataset/<id>/Audit | Generation、Dataset、Plan、AuditConfig |
| Export | Selection、Allocation、正配额 Dataset 的 Audit |

Source 依赖嵌套为 dataset/<id>/source/<source_id>/<kind>，没有跨数据集共享 current 指针。
依赖递归失效但保留旧版本。文件 artifact identity 包含内容摘要，不能仅靠路径复用。
操作回执绑定参数和输入版本，完成结果可重用；生成子图保留游标与逐阶段进度。
SQLite checkpoint 支持恢复，读取保存的配置并校验已登记数据。运行契约为 agent-runtime.v7；
代码或契约变化要求新运行，不迁移旧 checkpoint。

## 生成、审计与导出

Protocol 显式定义 execution_type、response_format、metric、data_binding、answer_contract、
question_intent、medical_constraints 和 template_slots。letter 使用唯一 scalar 的映射答案；free_text
使用唯一直接参考字符串；bbox 使用唯一绝对像素 xyxy，并要求 target 与框属于同一标注实体。

生成请求在私有适配层装配局部 Plan、配额和模板。模板模型只接收 TemplateSpecification，不接收逐图答案
或完整映射；逐题执行不调用模型。模板独立保存，Plan 不包含模板或 sample_count。
同一 Dataset/Leaf/image 最多一题，每种响应阶段保持现有每图限额。样本 ID 包含数据集身份。

各 Dataset 独立硬审计和抽样语义审查；语义 flags 不自动拒绝。最终流式合并通过硬审计的样本为
benchmark.jsonl，并导出 benchmark.raw.jsonl、audit_report.json、benchmark_manifest.json。
局部生成、审计、模板和 Plan 继续保留。最终 task_id 为 dataset_id::local_leaf_id，manifest 显式保存
局部映射；image_id 保留局部值，image_path 改为相对最终文件的正确路径，不再复制图像。
不合并 annotation 或 image manifest，不执行跨数据集像素去重。

benchmark-manifest.v3 包含 datasets、concepts、tasks、allocation 和产物引用。导出检查样本唯一性、
数据集/任务/图像引用和数量，重复导出不追加。planning_shortfall 为请求量减分配量；generation_shortfalls
为分配量减 raw 生成量；total_shortfall 只计算请求量减最终审计通过量，不能重复相加。
本次没有修改 evaluation，旧加载器不接受 v3 构建 manifest；多数据集评测接入留待后续。

## 推理与评测边界

评测入口显式接收 benchmark JSONL、`benchmark_manifest.json`、数据集旁的 `image_manifest.jsonl`，以及非空、
无重复的 `letter`/`bbox`/`free_text` 类型列表。加载器依据共享 metric registry 验证
task/response-format/metric 契约以及被引用图像
的路径和尺寸，只通过 `image_id` 找图，不解释 `image_path`。当前稳定指标映射为
letter 的 accuracy/macro-F1、free_text 的 SacreBLEU、bbox 的 mIoU。完整输入校验后只保留指定类型并
重新生成连续推理索引；任一指定类型无样本时在模型加载前失败。

模型通过显式注册的 `ModelAdapter` 加载；每个具体模型拥有独立模块，共用族级实现。目前支持
BreastGPT 8B、Qwen2.5-VL 7B/32B、InternVL3/3.5 8B、Qwen3-VL Instruct 8B/32B 与 Thinking 8B、
Lingshu 7B/32B、Hulu-Med 7B/32B、QoQ-Med-VL 7B/32B、Chiron-o1 和 MedGemma 1.5 4B IT。
Qwen2.5/Lingshu/QoQ-Med-VL 以处理后图像尺寸解释绝对 bbox，Qwen3、InternVL3/3.5、Chiron 与 BreastGPT
以固定 `[0,1000]` 坐标空间解释 grounding，Hulu 以原始图像尺寸解释；MedGemma 接收 896×896 视觉输入，
但定位输出采用归一化 `[y0,x0,y1,x1]` `[0,1]` 协议，并在 adapter 内换序为规范 xyxy。BreastGPT-8B 从本地
ModelScope checkpoint 的 `plugin/` 目录通过 MS-SWIFT 加载其专用模型与模板注册。InternVL3-8B 的动态预处理
最多使用六个宽高比分块，并在多分块时保留一张缩略图，因此每张图最多产生七个 `448x448` 输入；
InternVL3.5 与 Chiron 保持共享实现的 12 分块上限。letter VQA 的公共 system prompt 保持
原始单字母回答要求，user content 只含图像及题干与选项，不提供答案或输出格式示例；解析器也
接受裸单字母、ASCII 或全角括号包裹的单字母（括号内也可带选项文本）、Lingshu 风格的单字母加 ASCII 句点，或合法字母和句点后继续输出选项文本或解释；
例如 `A. E.` 与 `A. because ...` 均解析为 `A`。free-text VQA 不设置 system message，直接把完整非空终答作为预测；所有 bbox
任务都要求唯一框，并由各 adapter 明确其协议：固定 `[0,1000]`、处理后图像绝对坐标、原始图像像素坐标，
或 MedGemma 原生的 `[0,1]` yxyx JSON-list 对象。三类 user content 都只含
图像和题目（letter 另含选项），adapter 复用公共 prompt 构造并注入族级 bbox 坐标说明。InternVL3.5 adapter
使用模型专用 reasoning system prompt，要求推理完整置于 `<think>...</think>`，并按题型把终答置于
`<answer>...</answer>`；InternVL3 adapter 则保持公共直接回答 prompt 和普通生成行为。MedGemma adapter
对非 bbox 任务在公共 user prompt 后要求以 `FINAL_ANSWER:` 行结束，优先提取该边界；对既有输出仅兼容明确的
`answer is`、`category is X.`、Markdown option 和 `\boxed{X}` letter 终答。bbox 使用无 system message 的
原生 JSON-list 定位提示并严格提取唯一 `box_2d`；多个、越界或格式错误的候选保持为空并触发 parse error。

QoQ-Med-VL 7B/32B 使用其原生 medical reasoning prompt，要求 `<think>...</think>` 推理并把终答放入
`\boxed{...}`；adapter 从最后一个 balanced boxed payload 提取终答，支持内部嵌套 bbox JSON。Chiron
保留 checkpoint 默认 system message，在 user prompt 后追加官方 `Let's reason step-by-step` trigger，
并明确要求使用 `### The final answer is:` 作为终答边界。Chiron 与 InternVL3.5 的 generation config
均显式令 `pad_token_id` 等于 tokenizer EOS ID，只用于消除 open-end generation 警告。

`ModelOutput` 同时保留完整 `raw_text`、可选模型坐标空间和可选 `answer_text`。解析器仅在 adapter
按模型协议提取终答时优先使用 `answer_text`：QoQ 使用最后一个 balanced `\boxed{...}`，Qwen3 Thinking
使用最后一个 `</think>`，InternVL3.5 使用最后一个完整 `<answer>...</answer>` 并兼容旧 `</think>` 后缀，Chiron
使用最后一个 `The final answer is:`（兼容前方存在或缺少 `###`），Hulu 对同一 output IDs 分别保留
thinking 与无 thinking 解码，MedGemma 对文字任务使用 `FINAL_ANSWER:` 或上述受限 observed protocol fallback
letter 协议，对 bbox 将唯一 `box_2d` 从归一化 yxyx 换序为 xyxy。
分隔符缺失时回退完整响应，空终答保持为空并产生严格 parse error；bbox 最终答案以确定性候选提取兼容
`bbox`/`bbox_2d` 标签、Markdown 或外围文本，以及作为完整答案的裸四数数组/元组。只有唯一不同框才接受，
随后仍按 adapter 坐标空间严格检查 `0 <= x1 < x2 <= width`、`0 <= y1 < y2 <= height` 并缩放；越界框不裁剪、
不猜测协议，错误同时记录实际 bbox 与声明空间。推理产物把完整响应写入 `raw_prediction`，并在每题写入解析时
实际使用的 `coordinate_size: [width, height]`；该字段在向后兼容的 `medbench-inference.v1` 中可选，旧文件可以缺失。
所选有序类型写入 metadata。

`torchrun` 单节点每 GPU 启动一个 rank，每个 rank 独立加载一份完整模型，以轮转切片和普通 batch
循环推理，不使用 DDP 或 tensor parallel；32B checkpoint 也必须完整装入每个 rank 所属 GPU，增加 rank
只增加数据并行副本，不会切分模型。InternVL 族在 adapter batch 内按公共 system prompt 分组
调用 `batch_chat` 后恢复输入顺序；Hulu 因 processor 限制在 batch 内逐样本执行。rank 临时 JSONL
写完后创建完成标记，rank 0 无超时地等待全部标记，再按输入顺序写出自包含推理 JSON；耗时推理后
不再使用受 NCCL 默认超时约束的 collective barrier，中断仍不自动续跑。evaluation 子系统的 JSON
写入工具不依赖 benchmark 构建运行时，因此可在只安装 `requirements/evaluation.txt` 的独立 Conda
环境中通过源码 `PYTHONPATH` 运行。

评测输出文件以 response format 为顶层键，每个所选键对应一份完整的 `medbench-evaluation.v2`，独立从推理
JSON 重新计算逐题结果、逐 question type 主指标以及 category/类型内阈值正确率，并在逐题 `items` 中同时保留规范化 `prediction` 与模型完整
`raw_prediction`。每个 letter task 不论主指标均额外记录基于选项文本的 `macro_f1`，free-text
记录 `null`，bbox 则不输出 macro-F1；category 和 summary 对可用的 letter task F1 做 task-macro 平均。
选择题 macro-F1 使用选项文本而非平衡后的字母位置。letter/free-text 的 `metric_groups` 保持现有两列
层级宏平均；bbox 则显式输出 `acc_at_iou` 与 `miou`：先在每个父 task 内等权平均各 question type 的
阈值正确率与 mIoU，再在父 tasks 间等权平均。因此 question type 数量不改变父 task 权重；
父 task 聚合只使用 manifest 中显式的 `parent_task_id`，缺失该字段时任务独立计分，不从带点号的任务名推断父级。
其中 bbox 的 summary、category、task 与 metric group 均使用 `miou` 和 `acc_at_iou` 字段，逐题使用 `iou`；
`acc_at_iou` 采用可配置 IoU 阈值（默认 0.5）；
单框输出不含 confidence，因此不报告标准 AP/mAP。评测产物不修改 benchmark run 或其 manifest。
report CLI 的 `--reparse` 仍只从 `raw_prediction` 重解析 letter；独立的 `--reparse-bbox` 使用逐题
`coordinate_size` 重解析 bbox。旧 Qwen3 推理可根据 adapter metadata 恢复固定 1000×1000 空间，Thinking 型号
只解析最后一个 `</think>` 后的终答；旧的动态坐标 adapter 若未记录坐标空间则明确失败，不做范围启发式猜测。
MedGemma 重解析按已确认的原生 `[0,1]` yxyx 协议处理，因此可纠正旧 inference 中错误记录的 896×896 声明；
两种重解析都只生成 report，不原地修改 inference JSON。

## 维护边界

修改图流程、稳定类型、输入输出契约、产物名称、模型职责或数据可见边界时，必须同步更新本文。
`README.md` 面向使用方式；`AGENTS.md` 面向工作约束；本地旧 SVG/Draw.io 和历史运行产物均不是
当前架构真相源。本文不描述部署拓扑、威胁模型或未来规划。
