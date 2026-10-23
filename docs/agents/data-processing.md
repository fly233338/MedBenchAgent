# Data Processing Agent

使用 create_agent 接收来源、Profile、TaskPlan、DataRequirement 或已有数据，选择领域操作。
本轮只固定职责和工具契约，不固定“先 Profile 再整理”的协作顺序。

| 工具 | 能力 |
| --- | --- |
| inspect_sources / inspect_artifact | 检查显式委派的来源目录或正式 JSON 产品 |
| describe_data_contracts | 返回实际 Profile、TaskData、Case、Asset Schema 和后端能力 |
| invoke_coding / inspect_coding | 创建或续接内部 Coding Workstream，执行或检查调用收据 |
| inspect_data_candidate / inspect_candidate_cases | 检查真实候选 Manifest 与 Case 内容 |
| validate_data_candidate | 核验来源范围、成员摘要、Case 结构与实际覆盖 |
| submit_data_artifacts | 验证并提交正式 Profile / TaskData |

Coding 原生循环返回候选，Data Processing 决定是否接受。一次委派可以进行多次 Coding 调用；
Call ID 从持久化工具调用身份派生，重放沿用同一次 SDK 请求。私有 checkpoint 保存调用参数和文件收据，
不复制 SDK 内部历史或建立另一份执行账本。

候选来源只能位于显式委派的源目录或本 Coding Workspace。Profile 的来源身份与真实文件绑定，
TaskData 的统计由实际 Case 文件计算，正式 TaskData 复制为自包含快照。
自然语言接受条件仍由需求方判断；结构有效不等于任务语义已经成立。

静态指令位于 prompts/data_processing.md，Coding 使用现有 prompts/coding.md。
SDK 环境仍需预先配置，Local Runtime 不承诺操作系统沙箱或任意命令 exactly-once。
