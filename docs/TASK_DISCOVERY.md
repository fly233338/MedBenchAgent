# Task Planning 与执行契约

Planner 绑定一个 Profile 或 canonical Dataset；Profile samples 可以为空。TaskPlan.dataset_digest
固定规划证据来源，修订必须显式提供所依据的计划，不隐式沿用私有草稿。

## 两种操作，一个 TaskPlan

- discover_scenarios：判断 input_structure/operation/clinical_target，构造组合、筛查价值、选择
  已实现后端并提出准备要求。完成当前范围即可 completed；待准备 DataRequirement 作为 WorkNeed 返回。
- design_leaves：读实际 TaskData、记录需求和数据接受、设计并预检 Leaf。必要依赖未接受时保留
  有效中间产物并报告待办。需要改变 Axis/Scenario 时交回 Supervisor 发起发现修订。

发布计划与操作完成分别检查。部分计划可以发布；范围外待办不自动阻塞本次委派。
语义无意义和表示/后端尚未就绪是不同判断。已有后端为 vqa、report_generation、visual_grounding、detection。

## 准备需求与证据

DataRequirement 表达 affected_scenario_ids、backend_id、required_fields（名字、type、visibility、目的）、
case_granularity、input_relationships、minimum_cases，保留语义目的、representation 和接受说明。
无需先创造 Leaf ID。TaskData.requirements 将 requirement_id 映射到正式需求 Artifact ID/digest。
接受时比较需求内容，不能靠相同 ID 接受过期准备结果。

ProfileEvidenceRef 用 observation_id；Case EvidenceRef 用 case_id/evidence_id。
TaskData acceptance 的 Case 证据由其精确绑定的 TaskDataReader 解析；Profile 观察仍在原 Profile 查验。
inspect_task_data 固定 artifact_id/case_id，使用 text_offset/next_text_offset 读取长文本。

## Leaf 与生成

SemanticTaskSpec 持有任务语义、能力类别、修订、task_data 绑定及 LeafAssembly。
Assembly 保存有序 input_fields、模板及 question_fields、reference_field、output_spec、options、label_map。
模板仅使用声明的命名占位符与 options/task_objective；计算与原始格式转换提前交给 Data Processing。

预检遍历绑定 TaskData 中本 Scenario 的 Cases，复用实际后端组装规则；零容量不能成为可执行 Leaf。
输入/reference 分离、模板引用、答案类型、映射与必要几何元信息由代码检查。它不等价于医学语义验收。
GenerationSpec 只绑定计划、Leaf 修订、TaskData、采样量与 seed，case_ids 负责抽样，不用于掩盖坏数据。

修订仅改受影响任务并保留理由。整份 TaskPlan 摘要变化后重新绑定必要执行规格；有效 TaskData 可复用。
新候选必须取得匹配的 Audit/Review。Review 完成不等于 approved，生成候选也不等于已交付。
