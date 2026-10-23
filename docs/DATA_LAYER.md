# Case 中心的数据契约

## 状态与数据单元

本文固定 canonical 语义。当前实现纯类型、小型 JSON/JSONL DatasetReader、内容成员校验、
可移植快照和 BrEaST 样例入口；旧 DatasetIndex/Prepared 引擎仍等待迁移。

Dataset 是来源、Cases、Assets 与语义 Mapping 的集合。CaseRecord 是顶层临床单元，
可以代表病灶、检查或纵向病例，粒度由 Mapping 声明。patient_id 可选，一个 Patient
可对应多个 Case；一个 Case 最多关联一个已知 Patient。Study 是检查/采集分组，
不与 Case 强制一一对应，Evidence 可给出 study_id。
Image/Volume 是 Asset，不等于 Case。Benchmark Item 是任务在明确 Case 取样单元上的实例，
可使用多图；首期不跨 Case 组合。SamplingUnit 保留取样职责，不等于 Case 或 Item。

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
移除，不新增双格式读取。ImageEntry 仅负责呈现/导出，不是第二套资产登记。

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

## Mapping 与 Derived View

Mapping 描述 Case 分组、身份映射、字段含义、单位、缺失、时间/视图和实体关系；
不保存 Agent 执行状态，不建立新的可执行转换语言。

Task-specific Derived View 绑定 dataset_id/revision，通过 case_id 及必要的 Evidence/Entity
引用关联原 Case，只保存新增值，不复制整条 Case、临床表或原图。
派生资产仍使用 AssetEntry 并保留来源关联。首期只需少量合成派生行，
不实现通用 View 引擎。当前 Planner 需求交接使用 dataset、requirement_id 和 rows，
rows 每行有唯一 case_id；新增字段由需求规定。需求方记录接受的 Artifact ID/digest 和理由，
不会仅凭加工 Agent 的 completed 状态自动接受。
