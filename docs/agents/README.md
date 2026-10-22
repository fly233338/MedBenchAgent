# Agent 与业务能力

协作机制支持 Scripted 或模型 Supervisor；Coding 可独立选择 Scripted 或 OpenHands，
Planner 仍为极简合成实现。
实现与在线验收分别记录于 [PLAN](../PLAN.md)，不能将离线测试当作模型自主性证明。

| 角色/能力 | 文档 | 当前接入状态 |
| --- | --- | --- |
| Supervisor | [Supervisor](supervisor.md) | Scripted / ModelSupervisor，可显式选择 |
| Coding | [Coding](coding.md) | Scripted / Local OpenHands，按 Workstream 复用会话 |
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
独立请求通常 per-invocation，Coding 每个 Workstream 的长期状态由 OpenHands 管理。
Supervisor 只看到 Workstream 标识、初始范围和结果摘要；跨 Workstream 通过正式产物共享。
不得统一强制 checkpointer=True。字段以纯契约为准，边界见 [架构](../ARCHITECTURE.md)。

原 Data Preparation 职责归入 Coding，不再是独立必经 Agent。

## 公共接口与交接

`compilation.interfaces` 提供 AgentInput/Output、AgentSpec、CompilationContext 和 SupervisorView。
Graph 只接收注册的节点；节点读取 request，返回 result。带私有状态的子图使用窄输入/输出
schema；普通节点和外部 SDK Adapter 不必包装为 Subgraph。Assembly 根据冻结模式构造节点，
不创建数据，也不将专业 Agent 的内部 Tool 展开到顶层 Graph。

| 契约 | 主要字段与含义 |
| --- | --- |
| ArtifactRef | artifact_id、kind、相对 Compilation 根目录的 path、被引用文件的 SHA-256 digest |
| WorkRequest | request_id、compilation_id、agent_id、workstream_id、objective、scope、inputs、constraints |
| WorkResult | request_id、outcome、summary、artifacts、issues、needs |
| Workstream | Host 分配的 workstream_id、所属 Agent、初始目标与 scope |
| 协调记录 | 完整 request/result 元数据及正式引用；不含文件内容或 SDK 消息 |

outcome 为 completed、progress、needs_input、blocked 或 failed；completed 仅表示本请求完成。
need 包含 need_id、work/user 类型、summary 及可选 detail_ref。旧 needs/issues 是有来源的报告，
不会因为后续一次 completed 被自动标为已解决。历史记录及目录可通过 Supervisor inspect 查询。

Host 在执行前分配并 checkpoint 请求；恢复沿用身份，新业务要求形成新请求。使用稳定
compilation_id/thread_id、SqliteSaver 和同步 durability；等待用户通过 Command(resume=...)，
普通未完成节点以 None 续接。子图自己的持久化方式由实现决定，不统一强制 checkpointer=True。

正式文件必须真实存在并归属于本请求的输出目录，由 Host 登记摘要。读取或交接发现内容
变化时明确反馈，不重算引用来掩盖变更。摘要只覆盖该文件，不递归覆盖 Manifest 的依赖。
候选文件与正式产物分离，已交接文件不原地修改，相同字节重放可复用。

静态 Prompt 通过包资源加载，实际启用文本和摘要随 run.json 冻结；恢复不重新读源码指令。
模型凭据只从环境读取。专业失败通过 WorkResult 返回；未捕获的宿主程序错误保留失败 checkpoint。
Skill 仅是 OpenHands Coding 的可选领域扩展，其他 Agent 不需要 Skill 层。
