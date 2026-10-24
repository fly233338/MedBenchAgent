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


## 冻结领域 Skill

[coding.skills].enabled 选择包内 tabular-inspection、medical-image、mask-geometry，可设为空列表。
创建 Run 时复制所选 SKILL.md 及随附资源到运行目录的 resources/coding-skills，run.json 只保存名称和逐文件摘要。
恢复及每次调用先校验快照；Worker 通过 SDK 1.49.6 原生加载器加载，使用 AgentContext 与 InvokeSkillTool。
加载结果必须与允许集合完全一致，解析失败不得静默跳过。公共、用户、项目 Skill 和记忆自动加载关闭。
Coding 按需调用技能，同一会话始终使用同一允许集合；Skill 不是 Workstream 生命周期或新调度层。

表格检查使用 data.tabular.inspect_table，来源分页使用 data.sources.inspect_source；影像与几何直接复用
现有 Reader/Adapter/Mask API。统计结果带扫描范围，缺失标签和不明确的关系不自动补全。

Worker 使用 SDK 1.49.6 的 Conversation callbacks 投影 ActionEvent、ObservationEvent 和会话错误。
只记录工具动作、安全参数/观察摘要和关联身份，不记录 thought/reasoning；事件带父 DP request ID。
该诊断通道与 stdout 候选协议分离，日志缺失不改变正式验收或 SDK 恢复事实。


用户 Run 的 artifacts 保存正式数据；Coding 的 workspace、conversations 和 logs 位于
.runtime/runs/compilation_id/coding/workstream_id。数据与运行目录通过上下文显式传入。
SDK 底层 LiteLLM 原生回调记录模型调用开始、结束、失败和 provider usage，覆盖主模型与 condenser；
模型返回 ID 可与 ActionEvent 关联。完整 completion 日志关闭。缺失 usage 不估算为实际值。
本地进程使用包内模型元数据，不在导入时下载价格表。来源只读是工作契约，不是系统权限保证。
