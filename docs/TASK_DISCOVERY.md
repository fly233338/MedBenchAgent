# Task Planning 契约

## 状态

首期只有纯任务契约与 Scripted Planner。完整 Planner、Axis 语义调用、Scenario Screening
和 Generation 尚未迁入 Core。示例通过不证明医学任务发现质量或模型自主性。

## 职责与产物

保留 Axis → Cartesian Product → Scenario Screening → Leaf。模型判断语义，
代码构造组合并检查结构。Task Family 在 Scenario 后选择，不作为新的 Axis。

DataRequirement 表达需求 ID、Dataset 输入身份、Case 范围、目的、所需表示、
已有 Evidence 引用和满足条件。不指定调度顺序，不保存会话，不要求 Planner 编程。

SemanticTaskSpec 表达任务 ID、Scenario、Task Family、输入需求、参考答案要求、
输出范式、证据、可行性和未满足需求 ID。Task Family 包括 closed_vqa、open_vqa、
report_generation、visual_grounding、detection；这里只定义语义，不实现 Backend。

任务语义不强制包含 Resolver、执行程序、最终模板或 Benchmark 记录格式。
程序由 Coding 处理，具体生成由后续 Generation 能力承担。

## 交接与续接

缺少数据表示时保存草稿和 DataRequirement，返回 needs_input + work 需求。
Supervisor 决定委派对象；Planner 不直接调用 Coding，也不 interrupt 整个协作图。
新请求携带需求及新数据引用，Planner 从私有 checkpoint 继续并判断需求是否满足。

Task Plan 保存任务、证据与可行性；checkpoint 保存消息、游标和待续接工作。
Coding completed 不能直接将任务标为 feasible。不可行任务可排除或调整，
不必使整个 Compilation 失败。

## 验证

首期用合成字段派生验证请求、文件和任务草稿的关联，只做基础身份/结构检查。
详见 [Planner](agents/task-planning.md) 与 [PLAN](PLAN.md)。
