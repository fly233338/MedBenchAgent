# Supervisor

## 目标与职责

围绕用户目标和实际结果选择专业 Agent、询问用户或结束协作。
不管理准备、Axis、Scenario、Generation 等业务步骤。

## 输入、输出与能力

输入为目标、注册 Agent 描述、共享产物引用和 WorkResult 摘要。
输出为一次 delegate、ask_user 或 finish。Host 分配 request_id 并在执行前 checkpoint；
决策不能伪造已完成工作或存在的文件。

## 私有状态与持久化

父图保存目标、委派序号、当前请求、结果摘要、产物引用及用户回答。
SQLite 使用 compilation_id 作为 thread_id。服务和决策函数通过 Runtime context 注入。
专业 Agent 完整消息与 SDK 状态不进入 Supervisor 上下文。

## 反馈、失败与恢复

所有专业结果先返回 Supervisor，包括 failed/blocked。work 需求可转成新委派；
user 需求由 Supervisor 询问并 interrupt。恢复沿用请求身份，已 checkpoint 工作不重跑。
finish 只结束协作运行，不能声称尚未实现的 Benchmark 已交付。

## 首期与后续

首期为 Scripted 决策；后续通过同一决策接口接入真实模型。
合成测试验证路由、状态和恢复，不验证自主决策质量。
