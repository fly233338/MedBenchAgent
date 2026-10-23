# Supervisor

Supervisor 围绕目标和真实结果选择专业 Agent、询问用户或结束协调，不管理专业 Agent 的内部阶段。
具名节点内调用 create_agent，使用 ToolStrategy 返回 Delegate / AskUser / Finish。
Host 校验 Agent、Workstream 和 Artifact ID，补齐正式引用并生成 WorkRequest。

## 当前视图与只读工具

每次模型调用注入目标、用户回答、注册能力、最新结果、近期交接、产物目录投影和遗漏数量。
目标与用户回答不静默截断；可查询详情有界。历史反馈不会因为后续一次 completed 自动关闭。

inspect 是普通只读工具，不是最终响应分支，也不构成顶层路由：

| target | 内容 |
| --- | --- |
| artifacts | 正式目录；按 ID、kind、来源请求/Agent/Workstream 筛选 |
| workstreams | 工作范围与最新结果摘要 |
| work | 请求和结果，feedback_only 可筛选曾报告 needs/issues 的记录 |
| preview | 已登记文件的限长 UTF-8 预览，检查内容摘要 |

目录返回 total、offset、next_offset；预览返回截断或不可读原因，不扫描其他 Agent 的私有目录。
max_lookups 和 loop.max_tool_calls 共同限制本次决策的工具调用，以较小值为准。

## 私有状态与执行约束

原生子图在 Compilation 内保留私有 messages 和摘要，父图不接收这些状态。
dynamic context 每次调用重新锚定当前事实，历史结论不自动成为当前结论。
使用原生摘要、模型/工具调用限额，完整请求守卫覆盖指令、工具 Schema、当前视图、
消息和输出预留；token 计数是保守预算估计，不是提供商实际用量统计。

决策错误和非法引用作为结构化反馈进入原生循环；额度耗尽、缺失最终响应或模型错误不会产生
虚假的 Finish，也不会回退 Scripted。失败保留 checkpoint。
配置、启用模型和静态 Prompt 在 run.json 冻结；恢复不读取新配置或源码 Prompt。

## Workstream 与正式交接

uses_workstreams 的 Agent 可用 null 创建新 Workstream，或选择该 Agent 已有的 ID 续接。
scope 描述工作范围，不是会话身份。Supervisor 不选择 SDK Conversation 或读取私有 Coding 历史。
跨 Workstream 只共享正式 ArtifactRef；普通 Planner 不要求 Workstream。

Host 在委派前保存请求，结果通过 collect 核验。用户需求由 Supervisor 的 AskUser 触发
LangGraph interrupt，使用 Command(resume=...) 继续。Finish 只结束协调，不证明 Benchmark 已交付。
真实模型运行由用户负责，当前离线验证范围见 [PLAN](../PLAN.md)。
