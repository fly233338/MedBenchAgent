# Coding Agent

## 目标与职责

在 Compilation 任意阶段探索来源、理解数据、编写/运行转换、修复或报告不可行。
数据准备属于 Coding 的能力，不以 Understand/Text/Image 阶段协议作为必经入口。
医学影像函数作为 Tool/Skill 使用，不按格式拆 Agent。

## 输入、输出与能力

输入为 WorkRequest 的目标、范围、Dataset/需求/证据引用及约束。
输出为 WorkResult 和实际数据、代码或报告引用。Coding 不直接调用 Planner，
缺少信息时返回 need。

## 私有状态与持久化

一个 Compilation 可以包含多个 Coding Workstream。Supervisor 委派时使用
workstream_id=null 新建，或指定目录中已有 ID 续接；实际标识由 Host 分配并 checkpoint。
scope 描述本次工作的范围，不是会话标识。同一 scope 可以新建不同 Workstream。

每个 Workstream 拥有稳定 Conversation ID、独立 Workspace 和 SDK 持久化目录。
SDK 拥有消息、事件及执行进度；父图只保存最小 Workstream 目录、请求及交接摘要。
Supervisor 看不到 SDK 内部状态。请求完成不关闭 Workstream，不另建 Session Manager。

Terminal、File Editor 和 SDK 原生上下文压缩是本轮基础能力，工具配置不决定 Workstream
身份。所有 Workstream 使用本次 Compilation 冻结的 Coding 配置；完整 Skill 系统留待后续。

## 正式产物交接

Host 提供正式输入文件的实际读取位置、当前 Workspace 和请求输出目录。
不同 Workstream 通过 ArtifactRef 共享数据、代码和 Mapping，不读取其他 Workspace 的临时文件。
修改正式输入时先在自己的 Workspace 中产生工作副本，再提交新的文件。

SDK FinishTool 使用嵌套结构化候选结果：request_id、outcome、summary、files(path/kind)、
issues、needs。needs 本轮使用 need_id/kind/summary，详细材料作为文件提交。
模型不分配 ArtifactRef。Host 检查当前请求输出目录内的实际文件，复制到
artifacts/<request_id>/ 后登记；只允许相同内容重放，不覆盖已交接文件。
具体输出类型和字段要求来自 WorkRequest 或 DataRequirement，不写入通用 Adapter。

## 反馈、失败与恢复

在当前 Workstream 的 SDK 用户消息中按 request_id 去重。已发送但未完成的请求继续运行；
有效 FinishTool 结果已持久化时只重新交接，不重新运行 Agent。
缺少输入返回 needs_input；环境不可用或不能继续返回 blocked；执行异常、超时或无效
产物返回 failed，由 Supervisor 决定下一步，不自动回退 Scripted。

SDK 状态不复制到 LangGraph 或额外状态文件。Local Runtime 不承诺操作系统沙箱、
跨进程终端存活或任意外部命令 exactly-once。

## 首期与后续

本轮接入通用 Local OpenHands Coding，Planner 保持合成实现，实施进度见 [PLAN](../PLAN.md)。
旧 Runtime 保留给尚未迁移的调用方；新实现不依赖旧 Workflow、Preparation Graph、
StageRequest 或 StageResult。真实医学数据及完整准备校验不在本轮范围。

官方接口：[Persistence](https://docs.openhands.dev/sdk/guides/convo-persistence)、
[Structured Output](https://docs.openhands.dev/sdk/guides/structured-output)。
