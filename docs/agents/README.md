# Agent 与业务能力

协作机制支持 Scripted 或模型 Supervisor；专业 Agent 仍为极简合成实现。
实现与在线验收分别记录于 [PLAN](../PLAN.md)，不能将离线测试当作模型自主性证明。

| 角色/能力 | 文档 | 当前接入状态 |
| --- | --- | --- |
| Supervisor | [Supervisor](supervisor.md) | Scripted / ModelSupervisor，可显式选择 |
| Coding | [Coding](coding.md) | 普通节点，合成数据处理 |
| Task Planning | [Task Planning](task-planning.md) | 有私有状态的合成子图 |
| Generation | [Generation](generation.md) | 后续接口与职责 |
| Review | [Review](review.md) | 后续接口与职责 |
| Audit | [Audit](audit.md) | 后续接口与职责 |
| Allocation | [Allocation](allocation.md) | 后续确定性业务能力 |
| Delivery | [Delivery](delivery.md) | 后续交付能力 |

Agent 接收 WorkRequest，返回 WorkResult，只通过 Supervisor 提出跨 Agent 需求。
ArtifactRef 必须指向实际文件，completed 不等于 Benchmark 生成成功。
work 需求不触发 interrupt；用户需求由 Supervisor 协调。

私有消息、探索游标、SDK 会话不进入共享业务产物。Planner 使用 per-thread，
独立请求通常 per-invocation，Coding 长期状态由 OpenHands 管理。
不得统一强制 checkpointer=True。字段以纯契约为准，边界见 [架构](../ARCHITECTURE.md)。

原 Data Preparation 职责归入 Coding，不再是独立必经 Agent。
