# Task Planning Agent

## 目标与职责

依据证据发现任务空间，保留 Axis → Cartesian Product → Scenario Screening → Leaf。
关注任务语义、输入与参考要求、输出范式及可行性。Task Family 在 Scenario 后选择。
不同时承担程序、Resolver、最终模板和生成后端。

## 输入、输出与能力

输入为 WorkRequest、Dataset/证据引用、已有草稿和要求。
输出为任务草稿/SemanticTaskSpec、DataRequirement 或不可行原因。
使用 DatasetReader 和后续证据工具，不直接调用 Coding。

## 私有状态与持久化

task_planner 具名子图使用 checkpointer=True，保存探索上下文、草稿游标和待满足需求。
新请求不清空私有进度；namespace 由 LangGraph 管理。
Task Plan 保存业务任务，checkpoint 保存继续探索的位置，不共享完整消息。

## 反馈、失败与续接

缺少表示时保存草稿和 DataRequirement，返回 work need 并结束本次委派，
不能 interrupt 父图。新证据由 Supervisor 再次委派带入，Planner 判断是否满足需求。
Coding completed 不能自动使任务 feasible；任务排除不必使整个 Compilation 失败。

## 首期与后续

首期仅 Scripted 子图和纯任务契约。完整 Axis/Scenario、真实语义模型、模板及 Backend
后续迁入。详见 [任务契约](../TASK_DISCOVERY.md)。
