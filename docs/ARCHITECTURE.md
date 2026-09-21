# MedBenchAgent 架构

本文记录当前实现与模块边界，执行约定见 AGENTS.md。

## 项目定位与流程

项目使用 LangGraph 构建 annotation-driven Medical VLM benchmark。构建不调用视觉模型推断标签，
图像处理、benchmark 构建和 evaluation 保持独立职责。

流程：独立 Dataset 准备 → Profile/Card 注册 → Requirement → Dataset Selection →
逐 Dataset TaskPlanning ↔ 模板生成与审核 → 全部 Ready → 全局 Allocation → 全部 Generation →
全局 FinalizationAgent → delivery/benchmark.jsonl。模板阶段在 Plan 后、Allocation 前，任务缺陷返回 Planning。
这些是产物依赖，不是固定的 Supervisor 调度脚本。统一 Supervisor 持久化 run_intent：
prepare_dataset、build_benchmark 或 prepare_then_build，以及明确的准备目标 ID。
prepare/build 命令约束入口目的，Experiment 通用入口由模型解析意图、代码校验。
dataset_readiness 和 outstanding_work 从当前有效 artifacts 推导；仅准备在全部目标登记后结束。
同一数据集的图像、表格等来源可以关联整理；不同数据集的 annotation 不再合并。

- `data/`：准备执行后端、算子、图像与标注验证、Profile/query、Card 和 Catalog。
- `planning/`：需求、数据集选择、Concept/Leaf 契约、Protocol 与确定性分配。
- `agents/`：数据准备、局部规划、模板阶段、全局 Allocation、确定性生成和全局 FinalizationAgent。
- `runtime/`：共享行动循环、工具契约、依赖、事件、运行目录与最终导出。
- `generation/`、`audit/`、`rag/`、`evaluation/` 保持领域边界。

## 准备、Card 与 Catalog

`prepare --dataset-id <id>` 通过自然语言准备要求进入 Supervisor，后者只准备这一逻辑数据集；
支持 `--resume`。`register --dataset-id <id> --bundle <path>` 验证并登记已有完整 bundle。
构建时也可以显式准备新数据集。准备版本位于 `paths.prepared_datasets_root/<dataset_id>/<version>`，
默认根目录 datasets/Prepared。准备失败不替换已登记版本。

原始来源在 config 的 dataset_sources 中按 dataset_id、name、description、sources 登记，sources 复用
Source 契约，root 按项目根目录解析并写入运行配置快照。描述仅用于候选发现，不作为字段语义证据。
Supervisor 的 list_datasets/inspect_dataset_status 只读观察 Catalog、来源和有效性，不发布数据。
prepare_data 只接收 dataset_id、mode 和 feedback；代码解析配置来源。默认 reuse_or_prepare 冻结并核验
已有 Card，绑定有效 prepared 版本；原始路径不可用不阻止复用。显式 rebuild 创建新版本；禁用或损坏
的注册项不得被普通准备静默覆盖。缺失 Catalog 文件视为空，已有但格式损坏的 Catalog 明确报错。

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
准备 finish 前必须保存元数据；检查后无可靠描述则显式 save_dataset_metadata([])，不再静默补空文件。

`refresh-card --dataset-id <id>` 使用现有数据和 Profile，在 bundle/cards/<uuid>.json 发布新 Card 修订，
随后原子切换 Catalog 引用。旧 Card 和底层数据保持不变；刷新失败不切换注册项。v1 迁移只检查旧 Card
的确定性数据绑定，不复用其名称、描述、字段角色或能力提示；没有来源元数据时基于 Profile 重新总结。

Catalog 条目保存 dataset_id、name、description、available 和 annotation_path、image_manifest_path、
profile_path、dataset_card_path。普通构建不自动准备整个候选目录；缺少标准准备数据使用 prepare，已准备
但缺少/仍使用 v1 Card 的数据使用 refresh-card。显式指定优先；其余候选即使只有一个也经过相关性选择。
CLI 可重复 --dataset 直接给出 ID，绕过模型名称解析及推荐。Selection 只负责高召回 shortlist，
不提前证明任务可执行。新准备数据发布 PreparedCandidates 版本，使旧 Selection 及其下游失效；
新选择操作读取当前 Catalog 并冻结新快照，同一操作恢复保留原快照。
运行初始化不复制 Catalog；effective_config 保留真实目录路径。初始化只冻结 Blueprint 模板和 PDF
materials，不返回尚未创建的 working_blueprint_path；Requirement/Planning 写出文件后才发布该路径。

每次 Selection 在操作目录创建 dataset_candidates.json：记录实际需求与版本、名称解析所见目录、解析
结果、完整候选顺序、每个候选的 Card 摘要与内容摘要、以及不可用/未准备/未请求原因。模型看到的候选
Card 原文复制到该目录的 candidate_cards/；dataset_selection.json 同时保存冻结输入、选择方式和结果。
DatasetCandidates 是独立 artifact，Selection 依赖 Requirement、该快照和选中数据的 Card 版本。
同一操作恢复复用快照及已完成决策，不读取未选中候选的实时 Card；新 Selection 操作创建新快照。
选中数据继续核验 annotation、Profile、manifest 与冻结 Card 的一致性，外部 Card 刷新不改变运行中版本。

无匹配 prepared 数据时，从同一快照保存的配置来源中选择；显式目标仅存在于配置来源时直接准备。
source_selection.json 保存来源选择输入与决策；路径不可用、无相关来源或显式目标损坏时 blocked，
报告具体配置问题，不询问路径。来源选择只授权准备；注册后重新进行 prepared 数据选择。

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

build、prepare、Experiment 以及直接图调用共用按意图的本地预检：校验 Catalog 格式与 LLM 配置，
仅构建需要有效 Blueprint；准备后端在实际准备前检查。恢复使用保存配置。PDF、embedding、reranker
缺失只降低 RAG 能力，不阻止无检索规划；能力与原因在 CLI 和 Agent 观察中可见。

## 局部规划与全局分配

Benchmark Info 使用 clinical_scope、diseases[]、modalities[]、evaluation_focus、language、sample_count。
scope、语言和正整数题量必填；缺少语言时询问，缺省题量取 benchmark.default_sample_count（默认 1000）。
用户显式无效题量保持待澄清，不用默认值覆盖；题量来源单独记录在 Requirement，不增加 Info 字段。
brief_missing_fields 和 brief_requirement 阻止对未变化需求重复 specify，回答后重新提取完整需求。
其余只保留用户明确表达的内容。Selection 使用范围和限制，
Planning 使用范围与评测重点，Allocation 使用总体要求与局部委派偏好。

每次 TaskPlanning 只接收一个已选择 Dataset 的 Card/Profile/query，以及需求和 Blueprint。
Core Concept 来自静态模板，Leaf 动态定义 opaque ID，并显式保存 concept_id。不同 Dataset 的同名
Leaf 不代表同一任务。Extension 继续经过字段证据、临床意义、目标相关性及无法归入 Core 的审查，
只属于当前 Dataset，不增加跨 Dataset 语义对齐阶段。

Planner 在理解 Concept、展开 Leaf、设计 Protocol 和发现 Extension 时可查询 RAG；PDF 不补造标签。
Planner 初始化时检查 RAG 能力；不可用且没有 Materials 时自动发布空 Materials，不再要求空选择动作。
可用时保留显式材料选择，只有已选择材料且 RAG 可用时开放 search_materials，缺省条数取 rag.result_limit。
已有 Materials 不被初始化覆盖，变更材料仍按既有依赖使规划失效。
所有 Core 有 complete/unsupported 决议后检查 coverage gap，完成 discovery，再审查局部完整树。
Card 只提供检索线索；Profile、真实字段值和结构提供规划证据，Protocol 校验确认执行条件。
最终审查接收完整 Protocol/raw_value→option、Leaf 证据和材料段落、相关字段统计及有界真实值页、
精确 eligible_count。semantic_basis 逐字段保存 interpretation、source、reference 和 quote；
来源可以是已检索材料、已读取的数据集说明或已回答的用户决议，代码校验引用及逐字引文。
Leaf 的 mapping_grounding 必须逐项覆盖 answer_contract.value_mapping，保存 field、raw_value、option、
mode、interpretation 和 evidence_references。direct 仅允许相同的自描述 annotation 字符串（空白/大小写
显示规范化），实际值存在性由 Protocol 校验确认；编码解释、缩写展开、合并和语义改变使用 interpreted。
interpreted 必须关联同字段的文档/用户依据，或明确 dataset_specific 的材料。代码检查覆盖和引用，
Review 检查原值是否真正自描述、材料是否实际解释本数据集编码，不把模型声明当成语义证明。
不透明编码和含义不明确的缩写必须获得本数据集的语义依据，否则询问用户；不靠医学常识补造。
模型负责语义判断，结构校验不保证编码含义正确。unsupported 是合法结论，不为了概念覆盖制造 Leaf。
修正次数存入 PlanningDraft；缓存 Review、重新委派和恢复不重置额度。上游失效保留局部身份与已耗额度，
重新验证事实和审查。数据集选择、导入和题量分配不再属于 Planner。

Plan 发布要求结构、归属、Extension 决议、当前 Review、Protocol 和 eligible_count 有效。
保存 Leaf、完成 Concept 和发布 Plan 时拒绝零容量任务，必须显式修订或删除。
eligible_count 由代码使用与 Generation 相同的答案解析器计算。允许没有可执行任务的完成 Plan；
全部选中 Dataset 都没有可用任务时整体 blocked。失败不能被当作 unsupported 静默跳过。

独立 prepare_templates 阶段为 Plan 的所有可执行 Leaf 生成整套模板并审核，不依赖配额。
审核结果为 approved、revise_templates 或 revise_plan；前者在模板阶段修措辞/slot，后者将 Leaf 与原因
反馈到 TaskPlanning。template_feedback_id 绑定反馈 Plan 版本，已有 Planning 修订预算只扣一次。
模板生成/审核尝试持久化到 Dataset 局部 ledger，初次尝试加 harness.max_reworks 次返修，恢复和重新
委派不重置。specification 和 TemplateConfig 未变的批准结果复用；新 specification 必须重新审核。
合法空 Plan 发布空模板集，不调用模板模型。所有当前 Plan 与模板集批准后才允许 Allocation。

AllocationAgent 读取所有当前 Plan 和已批准 Templates，模型给出有理由的 Concept 权重，可显式给出下层权重；缺省下层
使用等权。代码按 Concept → Dataset → Leaf 分层执行 largest-remainder、容量截断和重分配。
Core 能力共享，Extension 以 Dataset 作用域独立计量；预算不随 Leaf 数量自动增加。
零权重明确排除，父层容量仅累计正权重子项；取整平局使用稳定 ID 排序。Allocation 独立发布，不回写 Plan。
逐 Leaf 容量不是每图限额下的联合可行解；本阶段不进行图像匹配或自动补量循环。

## 状态、依赖与恢复

Supervisor 从有效产物和模板返修请求计算 pending_plan/templates/generation_dataset_ids，展示与执行
入口共用门控。全部正配额 Generation 完成前，finalize 不可见也不可执行。
有效已完成阶段不可通过重复委派重做；上游失效使下游重新进入 pending。委派不覆盖原始要求和 Info，
也不是新增研究偏好的来源；结构校验不声称能判定所有语义偏移。

Supervisor 以结构化 delegation（objective、constraints、preferences、revision_notes）委派 Planning
和 Allocation；原始用户要求继续可见，委派不修改专业 Agent 的永久 system prompt。
PlanningDelegation 为 Dataset 局部 artifact，AllocationDelegation 为全局 artifact；变更通过依赖
使对应规划/分配及下游失效，不重置规划修订额度。DataProcessing 工具目录复用已有 Source、
Inventory、Mapping、AnnotationScript、Annotation、ImageInventory、ImagePlan、NormalizedSource、
Harmonization 依赖，展示与执行使用同一门控。

Experiment 为 thread_id 的稳定命名索引，位于 runs_root 同级 experiments.json，记录模式、数据集摘要、
run_directory、状态、Git commit 和时间。CLI 根据 checkpoint/interrupt/执行结果更新 registry；
registry 不参与 Agent 决策，不取代 checkpoint。完成实验只展示产物，不自动重跑或追加需求。
默认命令提供文本选择菜单，保留 --resume UUID 高级恢复接口；不实现多 Run 或批量实验管理。
CLI 按 tasks/updates/custom + subgraphs=True 的三元组事件处理流，只从 updates 顶层读取
LangGraph Interrupt，未知格式和缺少有效 question 明确报错。失败报告使用执行期间记录的运行目录，
Experiment 收尾的 checkpoint/registry 诊断失败明确显示，不覆盖原始运行错误及退出状态。
构建事件可省略 agent；模型/provider 名称不能填充缺失的 Agent 角色。FinalizationAgent 是全局工具使用 Agent；模板阶段事件归属 TaskPlanningAgent。

Supervisor 共享状态只保留选中 ID、各 Dataset 产物引用和进度摘要。专业 Agent 的完整 context 保存到
局部状态文件，重新委派时恢复；不整体回灌父状态。第一版串行调度，所有 Plan/模板 Ready 后才可 Allocation。
零配额 Dataset 明确跳过；全局 FinalizationAgent 有独立私有状态，父状态只保留状态引用与交付摘要。

共享 Harness 使用严格 Pydantic JSON 参数和一致的工具可见/执行门控。专业 Agent 局部 ask_user 使用
LangGraph interrupt，跨职责问题返回 Supervisor。明确区分 complete、needs_input、blocked、failed。
私有观察有界，完整资料通过工具读取；人工等待不计入运行时间。超出预算 blocked。

| Artifact | 主要依赖 |
| --- | --- |
| dataset/<id>/Dataset | DataConfig；准备发布还依赖局部 Harmonization |
| dataset/<id>/DatasetProfile / DatasetCard | 对应 Dataset / Profile |
| DatasetCandidates | Requirement、PreparedCandidates、冻结的候选输入文件 |
| DatasetSelection | Requirement、DatasetCandidates、选中 Card |
| dataset/<id>/PlanningDraft | Requirement、Dataset、Card、Materials、PlanningConfig、PlanningDelegation |
| dataset/<id>/Plan | Requirement、Dataset、Card、Materials、PlanningConfig、PlanningDraft、PlanningDelegation |
| Allocation | Requirement、Selection、AllocationConfig、AllocationDelegation、全部选中 Plan/ Templates |
| dataset/<id>/Templates | Plan、TemplateConfig |
| dataset/<id>/Generation | Dataset、Plan、Templates、GenerationConfig、Allocation |
| dataset/<id>/GenerationCheck | Generation、Dataset、Plan、Templates、Allocation、FinalizationConfig |
| CandidateAssembly | 全部输入版本与正配额 Dataset 的 GenerationCheck |
| BalancedCandidates / FinalValidation | CandidateAssembly / BalancedCandidates |
| Delivery | 全部输入版本；完成标记引用不可变 FinalValidation 版本 |

Source 依赖嵌套为 dataset/<id>/source/<source_id>/<kind>，没有跨数据集共享 current 指针。
依赖递归失效但保留旧版本。文件 artifact identity 包含内容摘要，不能仅靠路径复用。
操作回执绑定参数和输入版本，完成结果可重用；生成子图保留游标与逐阶段进度。
SQLite checkpoint 支持恢复，读取保存的配置并校验已登记数据。运行契约为 agent-runtime.v11；
代码或契约变化要求新运行，不迁移旧 checkpoint。

Requirement 独立保存 sample_scope（用户原文与范围 ID）。Planner 必须逐项 resolve_sample_scope，
使用真实标量字段的 in/not_in 条件或明确 unsupported；病例范围与评测主题分开解释。代码将 Dataset
条件注入全部 Leaf Protocol，发布时检查覆盖与一致性，容量、生成和审计共享同一 eligibility。
缺失字段不通过筛选，数组条件暂不支持。编码解释复用 semantic_basis，范围含义纳入完整树审核。

跨 Agent 澄清使用稳定 issue_id 关联角色、Dataset、引用和原问题。Supervisor 收到回答后调用
resolve_clarification 区分 semantic 与 requirement_change；仅后者修改 Requirement。已解决回答仅投递
给发起 Agent，并更新其私有状态引用、清除父子待解决状态。未解决问题阻止继续执行依赖工作。

## 生成、审计与导出

Protocol 显式定义 execution_type、response_format、metric、data_binding、answer_contract、
question_intent、medical_constraints 和 template_slots。letter 使用唯一 scalar 的映射答案；free_text
使用唯一直接参考字符串；bbox 使用唯一绝对像素 xyxy，并要求 target 与框属于同一标注实体。

生成请求在私有适配层装配局部 Plan、配额和模板。模板模型只接收 TemplateSpecification，不接收逐图答案
或完整映射；逐题执行不调用模型。模板独立保存，Plan 不包含模板或 sample_count。
同一 Dataset/Leaf/image 最多一题，每种响应阶段保持现有每图限额。样本 ID 包含数据集身份。

FinalizationAgent 在全部正配额 Generation 完成后启动；正常耗尽允许缺题，执行失败不等于完成。
工具支持查看输入、逐 Dataset 检查、分页调查异常、组装、均衡、最终验证及发布。模型决定检查顺序和
解释问题，不能绕过硬检查或调整答案/配额。不再运行 post-generation semantic audit，旧 audit 抽样配置移除。

GenerationCheck 按原始文件逐行检查 item、Plan、模板渲染、annotation 答案和图像引用；非法 JSON 或
item 单独隔离并记录原文件、行号、原始内容与原因，运行级必要文件损坏不能当作普通样本剔除。
raw 文件不修改，也不再合并成 benchmark.raw.jsonl；全局输出 raw_source_paths 逐 Dataset 指向原始文件。

组装按 Dataset、local Leaf、image 建立最终 identity，task_id 使用 dataset::leaf。等价重复稳定保留
一份，内容冲突的同 identity 全部隔离；不同图像相同题干不直接判重复。超 Leaf 配额的合法样本按稳定
顺序截取，其余隔离并报告。最终过滤后按最终 task 平衡 letter 正确选项位置，数量差最多 1；额外报告
按选项数量分组的总体分布。均衡保持答案文本和选项集合不变，bbox/free_text 原样透传。
最终检查再次验证这些不变量、唯一性、namespace、实际图像路径、配额和统计；失败不得发布。

报告 audit-report.v3 在全局、Dataset 和 Leaf 层次给出 allocated/generated/accepted/final，requested
仅在全局。单列无法归属、生成超额、规划缺口、逐 Leaf 生成不足、审计剔除和组装移除；每条记录只计
一次损失，多原因计数只是说明。生成超额不能抵消其他 Leaf 的不足，total_shortfall = requested - final。
题量不足仍正常交付并说明，不新增 partial、阈值、确认或自动补题循环；最终零题 blocked。

benchmark-manifest.v4 保存最终任务实际 sample_count、allocated_count、分层统计、原始产物引用和
交付文件清单。image_id 保持局部值，image_path 相对 delivery 计算，跨盘使用绝对路径；不复制图像，
不合并 annotation/image manifest，不执行跨数据集像素去重。交付包不承诺脱离 prepared 资产独立搬运。

已校验候选包的 benchmark.jsonl、benchmark_manifest.json、audit_report.json、quarantine.jsonl 在暂存
目录完成写入。FinalValidation 绑定现有 artifact 内容与输入版本；写出层只写 complete.json，再以同
文件系统目录重命名发布到 delivery/。只有带完成标记且通过绑定校验的包才是交付，Supervisor finish
只读取已发布结果。发布后、checkpoint 写回前的中断通过完成标记恢复，不重复均衡或追加；已发布目录
不原地覆盖，输入版本变化要求新运行。evaluation 没有改动，其旧加载器仍不接受多数据集构建 manifest。

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
