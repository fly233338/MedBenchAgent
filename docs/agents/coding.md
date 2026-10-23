# OpenHands Coding

Coding 是 Data Processing 的内部执行能力，不向 Supervisor 注册独立角色。内部 WorkRequest.operation
固定 execute；DP 拥有 Workstream 和由持久化 tool_call_id 派生的调用身份。

Adapter 提供只读输入位置、Workspace 和 output_directory；SDK 使用 Terminal、File Editor、原生压缩
及持久化。FinishTool 提交 CodingSubmission.result 候选，不分配正式 Artifact ID。
execute_candidate 返回候选和实际文件收据。DP 检查/修复后提交，SDK 对话不进入父图。

同一调用恢复复用已保存的完成结果；新修复创建新调用身份，可沿用自己的 Workstream。
角色配置和静态指令来自冻结快照，输出只能位于当前请求目录。数据来源/任务约束来自 DP 结构化交接。
Local 执行不承诺操作系统沙箱或任意外部命令 exactly-once。

运行环境由 config/coding-worker.requirements.txt 单独安装；不在启动时自动安装或退回脚本实现。
适配器位于 medbenchagent.agents.data_processing.coding.adapter，子进程入口为 python -m medbenchagent.agents.data_processing.coding.worker。
Worker 从 stdin 接收内部请求并在 stdout 返回候选；旧 Compilation 模块路径和 Dockerfile 已删除。
