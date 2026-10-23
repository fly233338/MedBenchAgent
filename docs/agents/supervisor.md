# Supervisor

实现位于 medbenchagent.agents.supervisor，由 Compilation Assembly 装配。

通过原生 create_agent 和 ToolStrategy 选择 Delegate、AskUser、Finish。每次委派指定角色、operation、
精确输入与 scope。Host 注入使用同一请求检查生成的就绪建议和阻塞原因，并在实际执行前再次检查。

默认协作路径为 Profile → 场景/需求 → TaskData → 验收/Leaf → Generation → Review/Audit → Delivery。
它写在 Prompt 中，不是 Graph 阶段锁。已有有效成果可复用；局部问题路由给相应责任角色。
只有无法通过允许的工作获得的重要用户事实才 AskUser。

配置默认同一问题 3 次修复、连续 3 次无进展、累计 12 次修复。计数由 Checkpoint 持有；换 request ID
不重置。预算耗尽保留有效成果，诚实结束。Finish 的目标结果和证据由 Host 核验，benchmark succeeded 必须有 Delivery。
