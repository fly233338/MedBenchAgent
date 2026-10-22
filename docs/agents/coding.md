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

首期是处理合成字段的普通节点。后续 OpenHands 在同一 Compilation 的重复委派中
复用 Conversation 和 Workspace。SDK 拥有消息、事件与执行进度，工作区保存脚本和文件；
LangGraph 只保留交接和必要会话引用，不复制完整 SDK 状态。

## 反馈、失败与恢复

Host 检查请求身份、文件存在和产物归属。不可实现时返回 blocked 或进展说明，
由 Supervisor 决策。未来 SDK 适配抽取现有消息去重和恢复能力；
首期不承诺任意外部命令 exactly-once。

## 首期与后续

只有合成处理示例。OpenHands、真实医学数据及完整准备校验尚未迁入。
Core 不依赖 Preparation Graph 或 StageRequest。迁移见 [PLAN](../PLAN.md)。
