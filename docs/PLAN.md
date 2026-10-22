# Compilation 实施计划

## 状态

- [x] 创建正常 refractor 分支及独立 Worktree，保留原工作区用户修改。
- [x] 第一步：架构、数据、任务和 Agent 文档先行；文档独立提交后开始代码。
- [x] 第二步：最小协作契约、Case 类型、DatasetReader 与任务契约。
- [x] 第三步：独立协作图、合成示例、一个多轮协作测试及一个最小恢复测试。
- [x] 本阶段第一步：更新架构、计划和 Supervisor 文档，随后再修改代码。
- [x] 本阶段第二步：统一包名、上下文名称、入口和测试引用。
- [x] 本阶段第三步：模型 Supervisor、相关配置冻结和模型适配测试。
- [ ] 在线验证：当前 Chat 服务连接失败，尚未执行真实模型协作。
- [ ] 后续：接入 OpenHands、Planner 和业务能力。
- [ ] 后续：切换 CLI/TUI，删除被替代实现。

当前阶段仅完成命名整理和真实 Supervisor 接入；OpenHands、医学数据和完整 Planner
不在本阶段范围。真实调用必须另有运行记录，不能用离线测试替代。

当前独立入口为 python -m medbenchagent.compilation，合成专业 Agent 位于 demo.py。
支持 --supervisor scripted|llm，默认 scripted；真实模式与专业 Agent 模式分别标示。
CLI/TUI 切换仍是后续工作。

## 提交顺序

1. 更新本阶段文档，记录目标与实现状态。
2. 更名为 compilation / CompilationContext，更新全部引用，不保留别名。
3. 接入模型 Supervisor、运行配置和有界产物预览，验证后提交。

只提交本次工作，不混入原工作区 AGENTS.md 修改。

## 验收

多轮协作测试使用同一 Graph 和两种初始证据条件，由 Scripted Supervisor 选择不同顺序。
覆盖重复调用、需求、实际文件产生、新证据改变草稿及私有状态隔离。
图不写死业务顺序；Scripted 结果不代表模型自主性。

恢复测试在已有产物后 interrupt 并关闭进程，新进程用同一 SQLite、Compilation ID
继续，检查私有进度、请求、引用及已 checkpoint 工作未重跑。
不增加真实调用次数要求或大量故障注入。

更名后继续运行两个既有协作测试。只新增一个聚焦模型适配的测试，复用模型通信测试。
非法 Agent、产物 ID 及结构错误必须反馈修正；模型错误不得触发 Scripted 回退。
可用配置下进行一次轻量真实 Supervisor 验证，不限定顺序或调用次数。
若配置不可用，明确记录在线验证未完成。

运行相关检查。Compilation 不导入 Workflow、Resolver、
Preparation Graph、BIRStore 或旧 harness。

## 后续资产迁移

| 资产 | 处理 |
| --- | --- |
| OpenHands 会话、事件、工作区 | 抽取到 Coding 普通节点，SDK 拥有长期状态 |
| Axis、Cartesian Product、Scenario、证据工具 | 迁入 Planner 私有子图 |
| 数据读取、图像转换、Mask 几何 | 普通工具，不按格式拆 Agent |
| Leaf Resolver/模板耦合 | 拆为语义任务、数据需求和生成能力 |
| BIR 写入/依赖校验 | 有业务需求时选取，首期不实例化 |
| Generation、Review、Audit | 相同契约或普通工具接入 |
| CLI/TUI | 新核心和真实能力就绪后切换 |
| 旧调度、阶段协议、冗余状态、无效测试 | 调用方切换后及时删除 |

旧代码可供参考，但 Compilation 不通过旧 Workflow 执行。
不建立兼容层、通用 Runtime、队列、新 Resolver DSL 或 Artifact 版本系统。

## 首期验证记录

- 新增协作测试仅两项：不同证据条件的多轮协作，以及新进程恢复。
- 两个协作测试：2 passed；本阶段更名后再次验证。
- uv run --locked ruff check .：通过。
- 修改共享 Case 类型后运行既有回归：328 passed、4 skipped；跳过项依赖可选医学读取库。
- 独立导入检查确认 Compilation/示例不加载旧 Workflow、Preparation、Resolver、BIRStore 或 harness。
- 文档相对链接检查通过；原工作区用户修改保留。

这些结果只验证合成协作和基础恢复，不是实际模型、OpenHands 或医学 Benchmark 的验收。

## 本阶段验证记录

- 包名、入口和上下文已统一为 compilation / CompilationContext；不保留旧别名。
- 新增一个模型适配测试，覆盖 Agent/产物 ID 修正、AskUser/Finish 解析、
  有界预览、配置快照、超限和模型不可用；复用已有模型通信与配置测试。
- 相关回归：73 passed；uv run --locked ruff check . 通过。
- CLI 离线检查使用 mock API：模型失败保留 checkpoint，恢复沿用原模式/模型，
  不读取修改后的配置、不重新发现模型，也不调用 Scripted 回退。
- 2026-10-22 检查当前 Chat 配置：默认本地端点 127.0.0.1:8005 的模型发现返回 ConnectError。
  原工作区与本 Worktree 均无其他本地模型配置。因此未执行真实模型协作，
  在线验收仍未完成；离线结果不替代这一项。
- 文档链接和命名引用检查通过。OpenHands、真实专业 Planner、医学数据及 CLI/TUI 切换仍未迁入。
