# Task Planning 契约

## 输入与证据

Planner 接受一个 dataset_profile 或 canonical dataset 作为当前规划依据，可附带明确选定的
TaskPlan 和 TaskData。完整 Canonical Dataset 不再是场景发现的前置条件。
TaskPlan.dataset 标识源 Dataset；dataset_digest 绑定当前规划依据文件（Profile 或 Dataset Manifest）。

PlanningEvidenceRef 可为 Profile 的 observation_id，也可为 case_id/evidence_id。
两者是不同证据来源，均要求实际存在；不是历史 Schema 的兼容读取。
Profile 的采样范围和未知项必须保持可见，不能将抽样证据当作全量扫描结论。

## 场景与数据要求

保留 input_structure、operation、clinical_target 三个 Axis，模型提交判断，代码构造组合。
SemanticScenario 可记录 backend_ids；Backend 不作为新的 Axis。
DataRequirement 增加 affected_scenario_ids、backend_id，仍保留 affected_task_ids 和接受条件。
场景级需求可以在 Leaf 不存在时提交。

publish_task_plan 校验 Axis/Scenario 覆盖、引用和已有任务的可行性，允许保留的 Scenario 暂无 Leaf。
这项能力支持中间交接，但不规定何时委派数据处理或何时再次调用 Planner。
DataRequirement 的新数据通过 accept_data_requirement 接受；已有 TaskData 也可直接通过
accept_task_data 接受。接受记录绑定内容身份、理由和证据。处理 Agent 的 completed 不等于需求已满足。

## Leaf 与组装

SemanticTaskSpec 保留语义要求、证据、可行性、修订和排除；新增：
- task_data：接受的 TaskData Artifact ID/digest。
- capability_category：可执行 Leaf 必须明确的能力类别。
- assembly：backend_id、input_fields、question_templates、question_fields、reference_field、
  output_spec、options 和 label_map。

input_fields 是有序字段名列表；asset 字段可包含有序 Asset ID 列表。
question_fields 将模板变量绑定到实际字段；模板另可使用 options、task_objective。
只支持声明式命名占位符，不执行模板表达式或任意代码。
复杂数据加工在 Data Processing 完成；输入和问题插值不能读取 reference-only 或 unknown 字段。

可执行 Leaf 要求 TaskData 已被接受、Dataset 一致、后端被 Scenario 选择且字段可解析。
语义草稿可以尚未绑定执行数据；Generation 不会将这种草稿当成可执行任务。
GenerationSpec 仅选择 Leaf 修订、TaskData、Case 范围、数量和种子，不重复定义模板或答案规则。

## 修订与交接

草稿写入使用 base_revision，已有任务修订递增 task revision，正式产品不可变。
同一后端多个 Scenario 可以共享 TaskData，同一 Case 可以支持多个 Scenario。
历史成果是否复用取决于明确的输入绑定，而非摘要中的旧结论。

当前操作与基本约束已实现；完整的发现、整理、Leaf 设计和返工策略下一轮统一写入 Prompt。
