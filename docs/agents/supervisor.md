# Supervisor

实现位于 medbenchagent.agents.supervisor.agent，由 Compilation Assembly 装配。

通过原生 create_agent 和 ToolStrategy 选择 Delegate、AskUser、Finish。每次委派指定角色、operation、
精确输入与 scope。Host 注入使用同一请求检查生成的就绪建议和阻塞原因，并在实际执行前再次检查。

默认协作路径为 Profile → 场景/需求 → TaskData → 验收/Leaf → Generation → Review/Audit → Delivery。
它写在 Prompt 中，不是 Graph 阶段锁。已有有效成果可复用；局部问题路由给相应责任角色。
只有无法通过允许的工作获得的重要用户事实才 AskUser。

配置默认同一问题 3 次修复、连续 3 次无进展、累计 12 次修复。计数由 Checkpoint 持有；换 request ID
不重置。预算耗尽保留有效成果，诚实结束。Finish 的目标结果和证据由 Host 核验，benchmark succeeded 必须有 Delivery。

就绪建议按 Requirement（准备）、Scenario（Leaf 设计）、Task（生成/修订）给出准确输入和阻塞原因。
使用 inspect(target=readiness) 的 query 参数及 offset/limit 分页；operation 可过滤建议，单条建议不截断依赖引用。
Requirement scope 不展开同场景的其他任务/需求；Task scope 只展开直接依赖。关联场景是上下文，不自动扩大修改权限。
