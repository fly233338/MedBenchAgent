# Task Planning 契约

## 语义任务与执行准备

原生 Planner 使用 create_agent，自主查询证据、提交判断并发布 TaskPlan。
保留 Axis → Cartesian Product → Scenario Screening → SemanticTaskSpec 的业务关系；
不使用全局 phase 锁。Task Family 在 Scenario 后选择，语义可行不等于生成后端执行就绪。

三个 Axis 为 input_structure、operation、clinical_target。模型提出值及支持/不支持判断，
提供 EvidenceRef 和理由；代码构造确定性 Scenario ID、检查引用和组合完整性。
变更 Axis 判断前先排除受影响的任务；重新判断 Scenario 后可以修订任务重新纳入。
未受影响的组合和任务保留。

## 当前唯一的 TaskPlan

TaskPlan 包含 plan_id、revision、DatasetRef、dataset_digest、axis_assessments、scenarios、
tasks、requirements、acceptances 和 exclusions。它不包含 messages、执行模板、Resolver
或 Agent 调度阶段。旧业务链的 TaskPlanState 仍是待迁移资产，不被新路径读取或转换。

每次草稿写入要求 base_revision 等于当前草稿 revision，并产生 revision + 1 的不可变草稿。
保存已有 SemanticTaskSpec 时，其 revision 同样必须递增一，change_reason 说明修改原因。
新请求可以显式绑定一个已发布计划；没有绑定时可续接同一 Dataset 下本角色的私有草稿。
新 Dataset 输入不隐式沿用旧判断；旧计划与不同 Dataset 内容绑定不匹配时明确反馈。

SemanticTaskSpec 保留 task_id、Scenario、Task Family、objective、input_requirements、
reference_requirements、output_paradigm、feasibility 和 requirement_ids，并记录：
- evidence_refs：任务成立的证据；
- input_evidence_refs：代表性的模型输入证据，必须明确 model_visible；
- reference_evidence_refs：参考答案依据，可以是 reference_only。

输入/reference 的实际跨 Case 选择与执行参数归属 GenerationSpec。Planner 不把模板、
选项生成或程序执行混入语义任务。

## 数据需求与排除

DataRequirement 绑定 Dataset、Case 范围、目的、表示、证据、接受条件、affected_task_ids
和 source_role（task_planner 或 generation）。新需求或对未接受需求的修改由 Planner 提交；
已接受需求的含义改变需要新 ID。

返回的 derived_view 使用 dataset、requirement_id、rows；每行以 case_id 关联原 Case，
新增字段由需求定义。接受工具检查版本、唯一 Case 和范围，Planner 判断自然语言接受条件；
接受记录绑定实际 Artifact ID/digest、理由和 EvidenceRef。后续发布或完成需要显式绑定
仍被活动任务使用的已接受 derived_view，Host 重新检查文件与身份。

新数据被接受后，仍需显式修订任务为 feasible。历史 completed 不自动满足需求。
任务排除保留定义、理由和证据；仅服务于已排除任务的需求不再作为活动阻塞，历史仍可查询。

## 发布与交接

publish_task_plan 验证三个 Axis、支持组合的覆盖、Scenario 判断、任务可行性和全部引用，
然后登记不可变 TaskPlan 及 DataRequirement。proposed 任务或未判断的 Scenario 不能发布。
没有支持组合时可发布带不支持依据的空计划。

需求尚未满足的计划可以发布并返回 needs_input；Host 根据正式需求形成 WorkNeed。
模型最终响应只选择本请求已提交的 ID，不能凭文本把草稿当成正式产物。
completed 要求选择与当前草稿一致的正式计划，并且不存在活动未接受需求。
是否已经生成 Benchmark 由后续产物决定。

## 验证

离线测试通过实际 create_agent 工具循环覆盖规划、提出需求、接收派生数据、修订、发布、
私有状态续接、并行写拒绝及无有效最终响应的失败路径。真实模型和医学语义质量由运行者验收。
