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

本阶段增加 ModelSupervisor，复用既有模型客户端和结构化输出修复。
模型选择 Agent 名称和已登记产物 ID，Host 补齐引用。只预览最近结果/需求涉及的少量
文本或 JSON；遗漏和截断明确标注。总请求超限时报告配置限额，不悄悄丢弃必要上下文。
提示词不给定 Coding/Planner 的业务顺序，专业 Agent 的职责与输入要求来自注册描述。

## 私有状态与持久化

父图保存目标、委派序号、当前请求、结果摘要、产物引用、用户回答及最小 Workstream 目录。
协调历史记录请求所属 Workstream，其最近结果摘要从历史生成，不另存会话状态。
SQLite 使用 compilation_id 作为 thread_id。服务和决策函数通过 Runtime context 注入。
专业 Agent 完整消息与 SDK 状态不进入 Supervisor 上下文。

## 反馈、失败与恢复

所有专业结果先返回 Supervisor，包括 failed/blocked。work 需求可转成新委派；
user 需求由 Supervisor 询问并 interrupt。恢复沿用请求身份，已 checkpoint 工作不重跑。
finish 只结束协作运行，不能声称尚未实现的 Benchmark 已交付。

结构错误、未注册 Agent、未知产物 ID 或非法 Workstream 选择反馈给模型进行有限修正。
修正失败或模型不可用时保留 checkpoint 并报错，不把失败转换成 Scripted 决策。
运行开始时保存 supervisor 模式和不含密钥的模型配置，恢复沿用相同选择。

## 首期与后续

首期 Scripted 决策继续用于确定性测试。本阶段真实模型适配的实现/在线验证状态见
[PLAN](../PLAN.md)。Coding 已接入可选 OpenHands 实现，Planner 仍为测试实现；
不能将离线验证描述为完整真实多 Agent 或医学 Benchmark 验收。
