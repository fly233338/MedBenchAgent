# Task Planning Agent

agents.semantic_planner 使用原生 create_agent；领域操作归属 planning，数据读取归属 data。
公共数据及任务语义见 [数据契约](../DATA_LAYER.md) 和 [任务契约](../TASK_DISCOVERY.md)。

现有 Axis、组合构建、Scenario 判断、任务保存/修订/排除和正式发布能力保留，补齐：
- inspect_profile：读取来源观察、覆盖范围与未知项。
- list_generation_backends / set_scenario_backends：查询和选择实际后端能力。
- inspect_task_data / accept_task_data：读取并接受后端任务数据。
- DataRequirement 的 Scenario/backend 范围和 TaskData 接受。
- Leaf 的明确能力类别、TaskData 绑定及 assembly 定义。

当前规划依据可以是 Profile，不强制存在规范化 Case。
场景级成果允许先发布而没有 Leaf；同一个持久 Planner 可继续处理后续任何合适的委派。
可执行 Leaf 的接受、引用与字段检查属于操作前置条件，不构成全局 phase 锁。

私有 messages 和草稿引用由原生 checkpoint 保存，草稿正文保存在内容寻址文件；
实际启用指令和配置冻结在 run.json。新请求清除上次提交引用和最终响应，保留私人工作上下文。
已接受任务数据需随当前请求明确绑定；过期引用不能替代当前事实。

本轮 Prompt 只描述职责、证据、工具及提交规则。完整协作工作流将在后续统一整理。
