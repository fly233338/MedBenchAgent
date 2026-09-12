# MedBenchAgent 架构

本文记录当前实现与模块边界。执行约定见 AGENTS.md；接口、职责或产物改变时同步更新本文。

## 项目定位

项目基于 LangGraph，从自然语言需求、标准图像级标注和医学 PDF 构建 Medical VLM benchmark。
构建仍以 annotation 为事实来源，不调用视觉模型推断标签。图像处理与独立 evaluation 子系统各自保持边界。

## 模块边界

- `runtime/`：JSON 行动 Harness、Pydantic 工具契约、依赖版本、运行产物、事件、模型客户端。
- `planning/`：需求提取、任务族筛选、全局协调、题量分配。
- `data/`：数据目录与缓存、profile、公共标注/图像校验、来源检查和二维图像转换。
- `audit/`：硬审计、语义审查和答案位置平衡。
- `agents/`、`generation/`、`rag/`、`evaluation/`、`workflow/` 保持角色与业务边界。

原根目录模块路径已删除，不提供转发层。公共标注 Schema 仍在 `schemas/annotation.v2.schema.json`。

## 自主编排与工具契约

入口为 `medbenchagent.workflow.graph:benchmark_graph`。根图初始化运行目录后执行 SupervisorAgent；
它通过受限 JSON 行动选择需求提取、数据准备、规划、生成或审计。医学任务、答案规则和指标不变。

`runtime/harness.py` 提供共享行动循环，行动包含 tool、arguments、reason。
`runtime/tool_contracts.py` 定义严格 Pydantic 模型；模型看到 JSON Schema，调用前验证必填字段、
类型、枚举、范围与额外字段。普通工具经 LangChain StructuredTool 接收具名参数；finish 和 ask_user
采用相同参数校验。参数错误返回定位到字段的 errors；内部 KeyError/TypeError 向外传播。
领域工具负责数据存在性、字段可用性和阶段前置条件。

专业 Agent 使用共享 ask_user 直接 interrupt，记录角色、问题、方案引用与回答，CLI 统一展示并恢复原
checkpoint。相同角色/方案/问题复用已记录决定。scope=global 的跨职责问题返回 Supervisor；局部问题
不需要 Supervisor 转述，不作为失败计数。Supervisor 澄清需求后产生新的需求版本，必须重新提取完整 brief。

各 Agent 的私有观察最多保留六条，每条摘要最多 800 字符并附回执引用；最近工具结果最多向模型展示
6000 字符。完整产物与回执在运行目录中。需求摘要、当前版本、进度和待处理问题进入 describe，完整 profile、
Blueprint、脚本与 PDF 内容通过查询工具渐进读取。公开图输出只含结果、状态和路径。

## 运行快照、依赖与恢复

新运行使用独立目录，保存有效配置、任务模板、数据目录和可选医学资料库的本地快照。
选中标准数据时复制 annotation、manifest 及其精确引用图像；原始数据准备首次接入时复制来源目录。
同一运行后续处理使用保存的输入。DuckDB/profile 与 PDF 分块/索引位于运行内部，不跨运行按路径复用。
profile refresh 创建新 Dataset 版本；PDF refresh 创建新缓存目录并保留旧缓存。没有输入内容指纹或散列检测。
需要采用外部新输入时启动新运行；系统不检测用户绕过系统修改运行目录的行为。

`.runtime/artifacts.json` 记录每个版本的 ID、类型、产物引用与直接依赖，以及当前版本引用。
当前版本同时由 LangGraph context/checkpoint 保存，旧版本仍可保留，但不能通过当前交付检查。

| 版本 | 主要直接依赖 |
| --- | --- |
| Dataset | DataConfig、DatasetSelection（准备输入另依赖 PreparedDataset） |
| Plan | Requirement、Dataset、Materials、PlanningConfig、BlueprintDraft |
| Templates | Plan、TemplateConfig |
| Generation | Dataset、Plan、Templates、GenerationConfig |
| Audit | Generation、Dataset、Plan、AuditConfig |

PlanningDraft、BlueprintDraft 和数据准备阶段也登记直接依赖。统一 require 检查递归失效，替代
planned/generated/audited 与 plan_binding；生成、审计、交付使用这套检查。改动上游时只切换当前引用。
内容相关配置按职责登记；日志、超时等运行参数不加入内容依赖。

终端使用 SQLite checkpoint，支持 `build --resume <identifier>`。回执绑定验证后的参数与输入依赖版本，
匹配时复用完成结果；数据容器操作还有同样绑定的内部完成回执。恢复读取运行保存的配置和输入。
`.runtime/contract.json` 保存运行时契约版本、Git commit 与源码工作区差异；代码或契约变化拒绝恢复，
明确要求新运行，不迁移旧 checkpoint。生成子图保留逐行 checkpoint、阶段游标及进度。

harness 限制行动次数、执行时间、连续失败和相同行动重复次数。人工等待不计入执行时间，预算耗尽返回
blocked；未审计批次不能交付。语义筛查模型失败继续使用观察性 fail-open。

## TaskPlanning

一个 TaskPlanningAgent 管理完整规划：

1. 查询预定义任务目录、具体定义及类别；分页查询字段路径和指定字段的统计、分布与结构。
2. 分页查询资料、读取 PDF 元数据、选择资料并检索相关段落。
3. 按现有 category 输入相关字段子集，保存每个任务 retain/drop、支持字段、question types、metric 和理由的草案。
4. 全局协调仅接收任务定义与紧凑决策，不重新输入完整 profile；检查重复语义、残余属性答案轴和覆盖。
5. 代码验证完整任务覆盖及证据契约，再复用原展开、题量分配和医学设计逻辑，最后发布 Plan。

关键词查询只是发现工具，不决定任务删除。任务族允许全部 drop；完整计划仍需至少一个保留任务。
全局协调不能引入草案未检查的新支持字段或新答案轴，必须先修订对应任务族。
修订保存新的族草案并重新协调；未变化任务的设计可复用，资料变化时不复用旧医学设计。
草案版本不能直接进入生成。

## 数据准备与公共校验

DataProcessingAgent 的阶段为 Source/Inventory → Mapping → AnnotationScript/Annotation →
ImageInventory/ImagePlan → PreparedDataset → Dataset。每个阶段检查直接依赖，完成后登记结构化结果；
来源或映射变化使后续版本自动失效。固定工具检查文件、表格、sheet、列类型/空值/唯一值与连接匹配计数；
Agent 决定实体语义、字段映射、冲突与未匹配项处理，无法据来源确定时直接询问。

数据集特定标注脚本在 Docker 中执行，源与项目代码只读，输出可写，网络关闭且限制资源和时间。
脚本两次运行比较字节，验证标准结构、唯一 image_id/image_path、来源引用、bbox 和计划图像数。
图像阶段只处理 annotation 引用文件，annotation 目录另以只读方式挂载；固定生成的 per-dataset 图像脚本
使用显式图像方案，两次执行后验证精确集合、编码、原始尺寸与可重复性。
最终保留来源盘点、结构化准备/图像方案、脚本和一份 validation.json；内部运行目录保留重放所需工作产物。

`data/validation.py` 提供公开标注、bbox 和图像集合校验；`data/image_assets.py` 提供 manifest 和
validate_image_asset。DataProcessing、Audit、Evaluation 不互相调用私有函数，也不从 skill 目录导入实现。
独立 skill 保留自包含资源，并与项目标准接口、图像策略同步。

PNG/JPEG 按内容识别并复制原始字节；其他支持的二维输入默认保留可表示的像素值、颜色和位深输出 PNG。
需要显示窗时使用单一可靠来源窗或绑定方案的用户决定，不自动百分位拉伸、选窗、丢弃颜色或位深。
不自动选体数据切片或全切片区域，拒绝多帧与金字塔。DataProcessing 与 Audit 保留纯色图检查；
Evaluation 不拒绝纯色图但限定 PNG/JPEG 编码。

## 答案与生成契约

每条标准标注代表一张图，含 patient、case、study、image，各实体独立 annotation。
图像 manifest 恰含 image_id、png_path、width、height；路径相对 manifest，历史键 png_path 也允许 JPEG。

- letter：唯一获准 scalar 经 answer_value_mapping 固定正确选项。
- free_text：唯一直接参考字符串。
- bbox：唯一合法绝对像素 xyxy；visual_grounding 要求与框配对的目标名称。
- 答案由代码固定，PDF 仅作为医学设计依据，不补造标签或实体关系。
- 同一 task/image 最多一题，保留每阶段每图限制和任务题量限制。

生成前按 generation.question_template_count 统一构建、审核模板，只允许 modality 槽位，visual_grounding
额外要求 target。逐题阶段没有模型生成或 Reviewer 调用。最终硬审计检查图像、答案与重复，抽样语义
flag 不自动拒绝题目。题量与 shortfall 以硬审计通过样本计算，数据耗尽允许带明确缺额完成。

## 产物与交付

版本工作产物在 `.runtime/` 的独立操作目录中，完整 Task Plan 嵌入 Blueprint；系统不覆盖已完成版本。
最终选择的有效版本复制到运行根目录：blueprint.json、templates.json、raw/audited JSONL、audit_report.json、
construction_events.jsonl、effective_config.json 和 benchmark_manifest.json。
不生成逐题 provenance、哈希链、多份事件汇总报告；现有 benchmark-manifest.v2 与图像 manifest 格式不变。
旧内部接口、checkpoint 和废弃工具不提供兼容层。

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
