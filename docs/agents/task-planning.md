# Task Planning Agent

## 职责与输入输出

agents.semantic_planner 通过 create_agent 实现真实语义规划。模型选择操作并作语义判断，
planning.semantic 执行查询、组合构建和契约检查；不调用旧 BIR、Resolver 或 Planner 私有函数。
详见 [任务契约](../TASK_DISCOVERY.md)。

WorkRequest 必须绑定一个 canonical Dataset，可携带一个明确选定的 TaskPlan、DataRequirement
及 derived_view。最终输出为当前请求的正式 TaskPlan、数据需求、问题与摘要。

## 工具

| 类别 | 工具 | 行为 |
| --- | --- | --- |
| 读取 | inspect_dataset、read_evidence_catalog、read_field_values、read_case、read_mapping | 返回版本、范围、分页、截断及证据依据 |
| 读取 | inspect_task_plan、inspect_input_artifact | 读取当前草稿或显式委派产物；长对象可按 object_id/text_offset 继续 |
| 判断 | record_axis_assessment、record_scenario_decision | 接受模型判断与证据，不执行隐藏模型调用 |
| 组合 | build_candidate_scenarios | 确定性构造支持值的 Cartesian Product |
| 任务 | save_semantic_task、exclude_semantic_task | 创建、修订或有依据地排除任务 |
| 需求 | propose_data_requirement、accept_data_requirement | 保存要求；核验返回数据并记录需求方接受 |
| 发布 | publish_task_plan | 验证并提交不可变计划与需求，返回真实登记 ID |

运行身份、Reader、目录和工具调用 ID 由 ToolRuntime 注入，不作为模型参数。
同一轮只允许一个草稿写操作；每次写入要求 base_revision，成功后返回新 revision。
数据不足时发布 needs_data 任务及 DataRequirement，然后交还 Supervisor；不直接调用 Coding。

## 上下文与持久化

每个 Compilation 的 Planner 使用原生持久命名空间。messages/摘要与草稿引用保存在私有
checkpoint，领域草稿保存在 planner/drafts 的内容寻址文件；提交后生成请求所属正式产物。
Runtime Context 不作为业务数据库。

每次模型调用重新注入当前请求、绑定 Dataset、草稿摘要、已提交 ID 和查询限额。
新请求清除上次的 submitted 和最终响应，保留私人工作上下文；恢复同一请求保留已有进度。
不通过文件修改时间或全局 latest 选择业务输入。

[planner] 配置上下文字符限额、page_size、preview_chars、max_scenarios；
[planner.loop] 配置调用限额、token 预算、输出预留与摘要触发/保留长度。均随运行冻结。
查询没有匹配数据与未提供 Mapping 分别返回 absent_in_scope 和 unknown。
这套 Reader 面向小型本地数据，会执行内容校验，尚未提供大数据索引或图像推理工具。

## 失败与恢复

非法参数、过期 revision、未知引用作为工具反馈供模型修正；未预期的运行错误保留异常。
调用额度耗尽返回 failed，并恢复已经通过验证的本请求正式提交；不会声称已完成任务。
缺失结构化最终响应、选择上次请求产物或选择旧草稿的发布结果均被拒绝。
需求是否满足以当前正式计划的接受记录为准，不依赖摘要或 Coding 的完成状态。
