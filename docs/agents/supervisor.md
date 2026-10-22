# Supervisor

## 目标与职责

围绕用户目标和实际结果选择专业 Agent、询问用户或结束协作。
不管理准备、Axis、Scenario、Generation 等业务步骤。

## 输入、输出与能力

输入为目标、注册 Agent 描述、共享产物引用、WorkResult 摘要和可用 Workstream。
输出为一次 delegate、ask_user 或 finish。Host 分配 request_id 并在执行前 checkpoint；
决策不能伪造已完成工作或存在的文件。

对 uses_workstreams 的 Agent，workstream_id=null 表示新建，已有 ID 表示续接。
Host 分配并 checkpoint 标识和初始目标/scope，拒绝未知 ID、跨 Agent 续接或不支持该能力的
Agent 携带 ID。scope 描述本次工作，不是会话标识。Supervisor 根据工作需要选择范围，
不绑定数据处理阶段；它看不到 Conversation ID、SDK 消息或 Workspace 内部状态。

ModelSupervisor 复用既有模型客户端、请求预算计算和结构化输出修复。
模型选择 Agent 名称和已登记产物 ID，Host 补齐引用。默认只提供近期的协调投影和少量
经过摘要校验的 UTF-8 预览；目录总数、遗漏数量和截断标记显式呈现，不返回无限增长的遗漏 ID 列表。
提示词不给定 Coding/Planner 的业务顺序，专业 Agent 的职责与输入要求来自注册描述。

## 有界上下文与 inspect

目标、能力描述和用户回答完整保留；这些必需信息超过 context_chars 时明确报错。
默认近期记录窗口 history_items=8、目录页 catalog_page_size=20、每次决策 max_lookups=4。
artifact_chars 和 max_artifacts 分别限制单个预览和预览数量；实际完整请求（含 Schema 与修复消息）
始终经过同一预算守卫。预算不足时先减少低优先级投影，再对可查询详情作带标记的缩短。

inspect 是 Supervisor 内部的只读响应分支，不成为顶层 Graph 路由或新 Agent：

| target | 查询内容 |
| --- | --- |
| artifacts | 正式目录；可按 ids、kind、来源 request_id、workstream_id、agent_id 筛选 |
| workstreams | 初始目标/scope 与最近结果；可按 ids、workstream_id、agent_id 筛选 |
| work | 请求/结果元数据、原始 needs/issues；可按 ids、request_id、workstream_id、agent_id 筛选 |
| preview | 按 Artifact ids 读取限长文本，摘要不一致时返回 artifact_integrity，不使用变化后的内容 |

目录查询使用 offset/limit，返回 total 和 next_offset；work 的 feedback_only=true 只筛选曾报告
needs/issues 的记录。它不代表这些问题仍未解决，也不依据后续 completed 自动关闭需求。
旧记录仍在 checkpoint 中，可通过查询找回。查询响应也受预算约束，超限保留失败 checkpoint，
不把预算耗尽当成任务完成，不扫描 Dataset 或其他 Workstream 的私有工作区。

## 私有状态与持久化

父图保存目标、委派序号、当前请求、完整 request/result 交接记录、产物引用、用户回答及最小 Workstream 目录。
协调历史记录请求所属 Workstream，其最近结果摘要从历史生成，不另存会话状态。
SQLite 使用 compilation_id 作为 thread_id。服务和决策函数通过 Runtime context 注入。
专业 Agent 完整消息与 SDK 状态不进入 Supervisor 上下文。

## 反馈、失败与恢复

所有专业结果先返回 Supervisor，包括 failed/blocked。work 需求可转成新委派；
user 需求由 Supervisor 询问并 interrupt。恢复沿用请求身份，已 checkpoint 工作不重跑。
finish 只结束协作运行，不能声称尚未实现的 Benchmark 已交付。

结构错误、未注册 Agent、未知产物 ID 或非法 Workstream 选择反馈给模型进行有限修正。
修正失败或模型不可用时保留 checkpoint 并报错，不把失败转换成 Scripted 决策。
运行开始时保存 supervisor 模式、不含密钥的模型配置及实际静态指令文本/摘要，恢复沿用原快照。
源码 Prompt 修改不会改变未完成运行。模型调用日志使用 compilation_id、agent_id 和 call_id 关联；
尚未分配请求的决策不生成 request_id，校验错误关联实际失败的模型调用。

## 首期与后续

首期 Scripted 决策继续用于确定性测试。本阶段真实模型适配的实现/在线验证状态见
[PLAN](../PLAN.md)。Coding 已接入可选 OpenHands 实现，Planner 仍为测试实现；
不能将离线验证描述为完整真实多 Agent 或医学 Benchmark 验收。
