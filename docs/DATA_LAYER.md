# Case 中心的数据契约

原始 SourceManifest 可记录 key_files：source_id、来源内相对 path 和用户给出的 purpose。
关键文件只标记已授权目录内的输入重点，不扩大来源范围，也不建立新的标签或病例关联。
运行前检查目录与关键文件可读性；TUI 草稿通过 runtime.intake 自动生成技术契约。

## 状态与数据单元

本文固定 canonical 语义。当前实现纯类型、小型 JSON/JSONL DatasetReader、内容成员校验、
可移植快照和 BrEaST 样例入口。旧 DatasetIndex/Prepared 执行链已移除。

Dataset 是来源、Cases、Assets 与语义 Mapping 的集合。CaseRecord 是顶层临床单元，
可以代表病灶、检查或纵向病例，粒度由 Mapping 声明。patient_id 可选，一个 Patient
可对应多个 Case；一个 Case 最多关联一个已知 Patient。Study 是检查/采集分组，
不与 Case 强制一一对应，Evidence 可给出 study_id。
Image/Volume 是 Asset，不等于 Case。Benchmark Item 是任务在明确 Case 取样单元上的实例，
可使用多项有序输入；复合病例需由数据处理明确构造粒度与来源关联。

Dataset revision 是输入快照标识。Case ID 在 Dataset 内唯一，Evidence ID 在 Case 内唯一；
Patient、Study、Entity 按 Dataset 命名空间解释，不自动跨来源合并。

## Evidence 与关系

保留 value、text、asset、region 四类和 id、name、value、ref、time、unit、view_id。
name 是字段语义，不承担身份。value 为标量/简单列表，text 为文本，
asset 的 value 引用 AssetEntry.asset_id；region 保留 normalized xyxy，
通过 ref 锚定唯一目标图像 Evidence。

补充轻量字段：

- source_refs：SourceRef(source_id, path, record?, field?)，是来源描述，不是执行 DSL。
- input_visibility：model_visible、reference_only、unknown，默认 unknown。
- entity：可选 EntityRef(kind, id)，表示证据描述的临床实体。
- study_id：可选检查/采集标识。

model_visible 仅表示可被任务选择，不意味着自动成为输入。
Mask、答案和校准信息不能因为格式是图像就成为模型输入。
现有 MaskFeatureSet 保留来源绑定的几何含义，首期不要求计算。

Evidence.ref 表达 Case 内锚定/关联。CaseRelation 用 subject、predicate、object、
source_refs 表达 EvidenceRef(case_id, evidence_id) 或 EntityRef 之间的事实关系。
EntityRef 只是标识，不要求实体数据库；关系含义由 Mapping 声明，并需来源依据。

多图需要共同临床语境；多视图需要实体/检查关联和 view_id 的含义；
多时间点需要时间单位/基准或显式先后关系。不同病灶、器官、标本用 EntityRef 区分。
图像数量、文件名顺序和相同 Patient ID 均不足以证明关系。未知保持未知。

## Asset 与 Locator

沿用 AssetEntry 的 asset_id、source_id、source_path、file_digest、kind、format、
width、height、metadata。asset_id 是稳定身份；source_id + source_path 是 Locator；
file_digest 描述内容而非身份。source_path 相对 source root，不能逃逸该目录。
metadata 保存物理属性，Evidence 保存使用语境的可见性。

新 Reader 只按 asset_id 解析，不猜测引用是路径还是 ID。旧路径兼作身份的行为在能力迁入时
移除，不新增双格式读取。资产登记与呈现工具统一使用 AssetEntry；派生输出通过实际文件与摘要交接。
metadata.semantic_role 是唯一的资产语义角色键，不将 metadata.role 解释为别名。
契约 Schema 由当前 Pydantic 类型生成，校验以对应模型和 Reader 为准。

## Dataset Manifest 与 Reader

DatasetManifest 描述 dataset_id、revision、cases_path、assets_path、sources 和可选
mapping_path、sampling_path、split_path。描述文件路径相对 manifest 所在目录；
source_id 唯一，source root 相对 manifest 或为调用方明确给出的绝对路径。
profile/DuckDB 是实现缓存，不是 Manifest 前提。

members 为被引用文件的完整清单，每项为 source_id（元数据文件为 null）、path 和 SHA-256
digest。必须精确覆盖 Cases、Assets、可选 Mapping/Sampling/Split、所有资产和 Evidence/Relation
source_refs 引用的来源文件。生产者显式调用 seal_manifest；Reader 只验证，不能重算摘要
来接受变化内容。未封存成员的 Manifest 不能作为正式输入。

snapshot_dataset 复制已验证成员并将 sources root 改为包内相对目录，拒绝不同内容覆盖。
每个正式引用的 manifest digest 进一步绑定其 members 身份。输入被冻结后，原始数据目录
可独立变化，不影响既有运行；冻结副本变化会被读取检查拒绝。

| 操作 | 结果 |
| --- | --- |
| get_manifest() | DatasetManifest |
| list_cases(offset, limit) | CasePage(items, next_offset) |
| get_case(case_id) | CaseRecord |
| get_evidence(case_id, evidence_id) | Evidence |
| get_asset(asset_id) | AssetEntry |
| resolve_asset(asset_id) | ResolvedLocator(source_id, path, file_digest) |
| read_mapping() | 当前 Mapping JSON，未提供时为 None |
| verify_integrity() / content_digest | 校验 Manifest 和全部成员；返回被接受的 Manifest 摘要 |

Reader 绑定一个输入快照，检查重复 ID、基础关联和实际文件存在性。
缺失文件/ID 明确报错，不以空记录掩盖。读取执行内容检查，适合小型本地开发数据。
证据目录与字段统计由 planning 的确定性查询完成；Reader 不判断 Scenario 语义，不执行医学转换。

## Mapping 与派生数据

Mapping 描述 Case 分组、身份、字段含义、单位和实体/时间/视图关系，不是执行 DSL。
派生值和资产经 Data Processing 整理到 TaskData，保留 SourceRef、准备脚本、参数和必要元信息。
不再使用旧 derived_view 交接或并行读取格式。

## SourceManifest 与 DatasetProfile

SourceManifest 包含 DatasetRef 和具名来源目录，CLI 将目录解析为绝对路径后登记为正式输入。
DatasetProfile 包含来源、字段/资产/关系概况、inspected_scope、coverage、unknowns、observations
及可选 samples。observations 必须有唯一 observation_id 和真实 SourceRef；samples 可以为空。
ProfileReader 核验被引用来源文件的成员摘要；Profile 不是完整规范化数据的同义词。
Planner 可直接引用 observation_id 开展场景规划，抽样与未知信息保持可见。

## TaskData

data.task_data 定义后端准备结果，区别于原始来源和可复用 Canonical Dataset：
- TaskDataManifest：DatasetRef、backend_id、scenario_ids、requirements（需求 ID → ProductBinding）、可选 Profile 绑定、
  cases_path、assets_path、来源与成员摘要，以及由真实文件计算的 case_count/scenario_counts。
- TaskCase：case_id、scenario_ids、fields 和 source_refs。同一后端多个场景可共享一行 Case。
- TaskField：name、type（asset/text/value）、value、input_visibility、source_refs、关联 ref 和 metadata。
  asset value 可以是单个或有序多个 Asset ID；value 可保存准备好的结构化 reference。
- PreparedAsset：稳定 ID、来源位置、实际文件摘要、kind、format 和可选属性；不限定 PNG/JPEG。

TaskDataReader 验证 Case/Asset 身份、字段关联、覆盖统计及所有成员。正式提交时复制为不可变
自包含快照，图像等资源通过引用共享，不按 Leaf 重复复制。
TaskData.profile 记录其 Profile 来源；Planner 在当前请求中检查并接受实际 TaskData，
Generation 只消费此准备契约，不读取 derived_view 或原始目录重新推断任务数据。

TaskData 还声明 case_granularity、input_relationships、preparation_scripts 和 preparation_parameters。
兼容表示可以共享数据；相同 backend 不要求同一物理 JSONL。minimum_cases 和字段规则来自绑定需求。
准备脚本与参数可追溯，空间资源在 metadata 声明 width/height、单位等必要信息。
大型索引、原始文件转换与医学质量判断不由 Reader 自动完成。


## Run 内的数据保留

一个用户 Run 只绑定一个 Dataset；多目录通过同一 SourceManifest 的 sources 表达。
inputs 保存冻结输入，artifacts 保存正式发布内容，运行目录保存未验收候选。
Profile 发布会复制 Coding 派生证据成员并改写对应 source root，原始来源仍外部引用；
TaskData 发布沿用完整成员快照。正式产物校验不依赖 Coding 临时文件继续存在。
Dataset 登记只保存 manifest/revision 引用；取消登记不删除来源或既有 Run。
