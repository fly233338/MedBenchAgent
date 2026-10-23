# Data Processing

原生 Agent 拥有 profile 和 prepare_task_data 两种操作，分别加载操作 Prompt。
工具覆盖来源检查、canonical schema 查询、Coding 执行、候选阅读/检查/修复和正式提交。
两种操作都有显式工具集合；inspect_candidate_cases 只在 prepare_task_data 中可用。
来源分页调用 data.sources.inspect_source，表格统计调用 data.tabular.inspect_table；工具结果说明扫描范围。

准备任务明确绑定 TaskPlan、DataRequirement 与来源/Profile。兼容的多个场景可以共享 TaskData；
同一 backend 不保证相同 Case 粒度或输入关系，不能据此强制合并。

Coding 成功只产生候选。DP 验证来源、内容、字段和计数后提交；Planner 另行验收任务适用性。
不生成临床答案、不改变任务目标，缺失或歧义作为具体工作需求返回 Supervisor。
准备委派按 Requirement scope 验收，不因一个需求关联某场景而要求处理该场景的所有需求。
细节见 [候选检查](../PREPARATION_VALIDATION.md)。
