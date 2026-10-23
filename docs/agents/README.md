# Agent 操作与交接

| 角色 | operation | 交付 |
| --- | --- | --- |
| [Supervisor](supervisor.md) | Delegate / AskUser / Finish | 下一步工作或有证据的目标结果 |
| [Data Processing](data-processing.md) | profile / prepare_task_data | Profile / TaskData |
| [Planner](task-planning.md) | discover_scenarios / design_leaves | 同一 TaskPlan 的场景或 Leaf 成果 |
| [Generation](generation.md) | generate | Spec、Allocation、批次、候选和真实缺口 |
| [Review](review.md) | review_plan / review_candidate | 只读目标上的正式 Review，候选另附 Audit |
| [Delivery](delivery.md) | publish | 可独立读取的正式包 |

[Coding](coding.md) 仅由 DP 内部使用；[Allocation](allocation.md)、[Audit](audit.md) 与 Delivery 是确定性能力。
WorkRequest 包含 operation、明确 scope 和 ArtifactRef 输入；WorkResult 表达本次工作结果。
scope 的每项为 artifact_id、kind（artifact/scenario/task/requirement）和对象 ID 列表；整份输入用 artifact。
自由描述写在 objective/constraints，空 scope 不隐式扩大为全计划。

操作 completed、数据就绪、目标 succeeded 三者分开。发现可完成并留有准备需求；审查可完成但拒绝批准；
生成可完成但存在缺口。run.json 的 goal_spec 固定目标类型、计划层次与受限交付政策，benchmark_request 固定数量。

静态 Prompt 按 common+role+operation 组合并冻结。动态请求和材料是工作数据，工具 schema 来自唯一类型。
恢复使用冻结快照并要求实现指纹匹配；历史摘要不是正式证据。父图不保存专业角色对话。
