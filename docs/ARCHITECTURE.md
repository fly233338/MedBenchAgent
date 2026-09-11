# MedBenchAgent 架构

本文记录当前实现与模块边界。执行约定见 AGENTS.md；接口、职责或产物改变时同步更新本文。

## 项目定位

项目基于 LangGraph，从自然语言需求、标准图像级标注和医学 PDF 构建 Medical VLM benchmark。
构建仍以 annotation 为事实来源，不调用视觉模型推断标签。图像处理与独立 evaluation 子系统各自保持边界。

## 自主编排

入口为 `medbenchagent.workflow.graph:benchmark_graph`。
根图初始化运行目录，随后执行 SupervisorAgent；它通过受限 JSON 行动调用工具，根据反馈继续、求助或结束。

- Supervisor 的 specify 工具复用需求提取；完整性由代码检查，不推断缺失目标。
- TaskPlanningAgent 使用独立行动循环，工具覆盖数据集选择、profile、完整 Task Plan、单任务修订、
  题量分配、资料选择和医学设计。规划与指标选择复用原有领域函数。
- DataProcessingAgent 是可选的数据接入入口，检查原始来源，编写和执行转换脚本，输出标准 annotation、
  图像及 manifest。标准数据可以绕过它。
- GenerationAgent 先构建、审核模板，再以原有生成子图遍历候选、固定答案并实例化题干。
- Audit 使用确定性检查与抽样语义筛查，向 Supervisor 返回拒绝原因、问题任务和计数。

Specification 与 MedicalAlign 不再是固定角色阶段。医学设计属于规划，模板属于生成。
例行人工确认已移除；专业 Agent 返回未解决的问题，由 Supervisor 发起自然语言澄清。
Supervisor 可以选择返工，但不能绕过规划完整性、生成契约和最终审计。

## Harness、状态与恢复

`harness.py` 提供共享的模型行动循环，行动包含 tool、arguments、reason。
只允许注册工具和 finish/ask_user；参数由工具验证，工具返回 status、summary、updates。
专业 Agent 的 needs_input 返回 Supervisor；只有 Supervisor 的 ask_user 使用 interrupt。

根图公开输出只包含结果、状态与路径；context 属于内部运行状态。
模型决策只接收角色需要的摘要，完整记录不进入 Supervisor 提示。
规划的完整 profile 位于私有上下文；返回根图时使用 access manifest 中的 profile 路径。

终端入口使用 SQLite checkpoint，支持 `build --resume <identifier>`。
内部 `.runtime/` 保存完成操作的回执和工作产物；重放相同操作复用已有结果，不重复执行已完成工具。
生成子图保留逐行 checkpoint、阶段游标及进度合并。raw 已存在时复用完成结果。
运行恢复要求输入与配置不变；不存在内容指纹或自动输入变更检测。

配置中的 harness 限制每个 Agent 的行动次数、执行时间、连续失败和相同行动的重复次数。
预算耗尽返回 blocked。工具异常携带可操作原因；语义筛查的模型失败仍采用观察性 fail-open。
重新规划或重新处理数据使生成和审计状态失效；重新生成使审计失效。
各次执行的工作目录独立，避免原地修改已完成批次。

## 数据与答案契约

标准输入仍由 `schemas/annotation.v2.schema.json` 定义：
每行一张图，包含 patient、case、study、image，各实体拥有独立 annotation。
数据处理输出同一接口；模型不能从文件名、图像外观或医学知识补造标签和实体关系。

image manifest 每行仅含 image_id、png_path、width、height，路径相对 manifest。
数据处理脚本由本地 Docker 镜像执行：源目录只读、输出目录可写、网络关闭，并限制资源和时间。
二维格式检查复用已有医学图像转换参考实现。准备完成后独立检查源图像格式和空间尺寸，
PNG/JPEG 保持原始字节；不支持体数据、多帧、金字塔和全切片。

DuckDB/profile 缓存按源路径、显式 revision 和缓存格式版本组织。
PDF 缓存按资料路径、分块配置、模型路径与显式 revision 描述组织。
输入更新后通过 refresh 参数重建；原始数据与旧运行不自动迁移或删除。

保留现有答案规则：
- letter：唯一获准 scalar 经 answer_value_mapping 确定正确选项。
- free_text：答案必须唯一对应直接参考字符串。
- bbox：唯一合法标注框，绝对像素 xyxy；visual_grounding 还要求与框配对的目标名称。
- 答案由代码固定；PDF 只能提供任务设计依据。
- 同一 task/image 最多一题，保留每阶段每图限制与任务题量限制。

模板在生成前统一构建和审核，数量由 generation.question_template_count 配置。
模板只能使用 modality 槽位；visual_grounding 额外要求 target。
逐题阶段无模型生成或 Reviewer 调用；代码校验继续执行，最终按任务抽样做语义检查。
固定 seed 和稳定数据身份用于模板选择与答案位置平衡。

## 产物与交付

静态模板在 `config/blueprint.json`；运行不修改共享配置。
规划快照统一保存为 `blueprints/vN.json`，完整 Task Plan 嵌入 Blueprint。
内部工作产物位于 `.runtime/`，最终交付的选定版本复制到运行根目录：

- blueprint.json、templates.json。
- benchmark.<dataset_id>.raw.jsonl 与 audited.jsonl。
- audit_report.json：硬审计记录、统计及抽样语义记录。
- construction_events.jsonl：工具结果、模型调用与成本、异常和生命周期。
- effective_config.json：非敏感配置。
- benchmark_manifest.json：运行标识、数据集、任务指标及相对产物路径。

不生成逐题 provenance 文件、独立 witness 链或事件聚合报告。
审计通过 image_id 和任务规则直接检查 annotation，保留图片解码、尺寸、重复项与答案检查。
语义 flag 不自动拒绝题目；最终题量和 shortfall 按硬审计通过的题目计算。
数据耗尽允许带明确缺额完成；未审计的批次不能交付。

manifest 使用 benchmark-manifest.v2，image manifest 使用上述精简字段。
旧契约需要重新准备后再进入新评测入口，不增加兼容兜底。



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
