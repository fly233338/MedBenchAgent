# Data Processing

原生 Agent 拥有 profile 和 prepare_task_data 两种操作，分别加载操作 Prompt。
工具覆盖来源检查、canonical schema 查询、Coding 执行、候选阅读/检查/修复和正式提交。

准备任务明确绑定 TaskPlan、DataRequirement 与来源/Profile。兼容的多个场景可以共享 TaskData；
同一 backend 不保证相同 Case 粒度或输入关系，不能据此强制合并。

Coding 成功只产生候选。DP 验证来源、内容、字段和计数后提交；Planner 另行验收任务适用性。
不生成临床答案、不改变任务目标，缺失或歧义作为具体工作需求返回 Supervisor。
细节见 [候选检查](../PREPARATION_VALIDATION.md)。
