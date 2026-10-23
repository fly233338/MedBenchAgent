# Data Processing 的候选检查与任务验收

Data Processing 是原生模型角色，拥有 profile 与 prepare_task_data 操作。OpenHands 仅执行内部
execute 请求，返回候选文件和真实执行收据；DP 可在同一委派内携带诊断多次调用 Coding 修复。

profile 形成来源观察、范围、unknowns 与证据，不要求完整规范化。prepare_task_data 按已绑定
Scenario/backend/需求准备 Cases；共享集合必须具有兼容粒度、输入关系及字段定义。

validate_data_candidate 检查候选类型、来源权限、Dataset/需求/Profile 绑定、成员与实际计数。
TaskData 逐条比较要求的字段类型/visibility、范围和最低容量。提交生成不可变自包含产物。
准备脚本及参数随 Manifest 保留；文件摘要由 Host 计算。来源引用只证明内容身份，不证明标签含义正确。

Planner 根据实际 TaskData 证据作任务适用性判断并记录 acceptance；DP completed 不自动替代验收。
输入顺序、标签解释、派生单位及间接答案泄漏由语义判断结合确定性检查处理。未知关系不能猜测。

旧 understand/text/image Preparation 阶段协议、Publisher 和独立 Worker 已移除。保留可复用的
asset_adapters、readers、mask_geometry 与源文件摘要工具供 Coding 明确调用。
表格处理直接使用 Coding 独立环境中的 pandas/openpyxl/xlrd；不维护表格读取薄包装或旧来源索引协议。
本轮只验证小型合成文件与 SDK 替身，不声明真实医学质量或外部执行 exactly-once。
