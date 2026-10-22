# Compilation 实施计划

## 状态

- [x] 首期在独立 Worktree 实施；当前已迁回主工作目录，工作分支为 refactor，用户修改保留。
- [x] 第一步：架构、数据、任务和 Agent 文档先行；文档独立提交后开始代码。
- [x] 第二步：最小协作契约、Case 类型、DatasetReader 与任务契约。
- [x] 第三步：独立协作图、合成示例、一个多轮协作测试及一个最小恢复测试。
- [x] 上阶段：统一 compilation 命名、模型 Supervisor、配置冻结和模型适配测试。
- [x] 本阶段第一步：确定 Workstream、Coding 职责和验收边界，先提交文档。
- [ ] 本阶段第二步：Workstream 契约及 Local OpenHands Coding 接入。
- [ ] 本阶段第三步：独立 Coding 模式、配置冻结、合成示例和针对性验证。
- [ ] 在线验证：当前 Chat 服务连接失败，尚未执行真实模型协作。
- [ ] 后续：接入真实 Planner 和业务能力。
- [ ] 后续：切换 CLI/TUI，删除被替代实现。

本阶段目标是基于 Workstream 的通用 OpenHands Coding Agent。医学数据、完整 Planner、
Benchmark Generation、Docker 和 Skill 系统不在本阶段范围。
真实调用必须另有运行记录，不能用离线测试替代。

当前独立入口为 python -m medbenchagent.compilation，合成专业 Agent 位于 demo.py。
新增 --coding scripted|openhands，与 --supervisor scripted|llm 独立，默认均为 scripted。
输出分别标示 Supervisor、Coding 和 Planner 模式；Planner 继续 scripted。
CLI/TUI 切换仍是后续工作。

## 提交顺序

1. 文档与接口边界：明确 Workstream 所有权、候选结果及恢复规则。
2. Coding 实现：最小工作契约扩展、会话映射、SDK 恢复及正式产物交接。
3. 入口与验收：独立模型覆盖、模式装配、合成示例、测试和使用说明。

只提交本次工作，不混入原工作区 AGENTS.md 修改。

## 本阶段接口与实施约束

- Delegate 的 workstream_id=null 表示为支持 Workstream 的 Agent 新建；已有 ID 表示续接。
  Host 分配 ID，目录与 WorkRequest 一同 checkpoint。未知 ID 或所属 Agent 不匹配均拒绝。
- AgentSpec 增加默认关闭的 uses_workstreams；Graph 按能力处理，不硬编码业务阶段。
  Supervisor 只看到初始目标、范围及从协调历史生成的最近结果，不接收 SDK 状态。
- 一个 Workstream 一个稳定 Conversation、Workspace 和 SDK 持久化目录。scope、工具集
  不决定会话身份；请求完成不关闭 Workstream，不新增 Session Manager 或 Agent 层级。
- 独立 worker 使用当前 OpenHands SDK/Tools 1.49.6，基础 Terminal、File Editor 和原生压缩。
  从 SDK 用户消息去重，已完成请求从 FinishTool 结构化事件恢复，不另存会话状态账本。
- Host 检查当前请求候选文件并复制为正式产物。跨 Workstream 仅通过 ArtifactRef 分享，
  不依赖其他 Workspace 的临时文件。工作失败返回 Supervisor，不回退 Scripted。
- [coding.llm] 覆盖已解析的公共 [llm]；两个 Agent 分别冻结配置，凭据只从环境读取。
  Coding 默认 80 次迭代、3600 秒、4096 输出 tokens、80 个压缩事件、24000 压缩 tokens。
- 运行入口只读取相关配置和现有 Coding 环境，不初始化 RAG 或旧准备链，不自动安装依赖。
  resume 使用冻结配置；Workstream 目录只存在于 checkpoint，不在 run.json 重复登记。

## 本阶段验收

- 保留现有协作和恢复测试，以 Fake SDK/Runner 聚焦验证 A → B → A 会话复用与隔离。
- 检查相同 scope 可新建不同 Workstream、跨 Compilation 隔离、跨正式产物读取，
  修改临时文件不影响已交接内容，不能提交其他 Workstream 的候选文件。
- 覆盖 Host 消息已持久化、SDK 已完成但 Graph 尚未记录结果两个恢复窗口。
- 扩展 Supervisor 的 Workstream 选择修正和入口四种模式组合、配置覆盖及冻结测试。
- 运行相关 Pytest 和 uv run --locked ruff check .；可用模型下执行轻量真实 Smoke Test，
  包含同一 Workstream 再次委派和跨 Workstream 正式产物读取，否则记录在线验收未完成。
- 保留有调用方的旧 Runtime；新 Compilation 不加载 Workflow、Preparation、Resolver、BIRStore。

## 既有骨架验收

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

## 模型 Supervisor 阶段验证记录

- 包名、入口和上下文已统一为 compilation / CompilationContext；不保留旧别名。
- 新增一个模型适配测试，覆盖 Agent/产物 ID 修正、AskUser/Finish 解析、
  有界预览、配置快照、超限和模型不可用；复用已有模型通信与配置测试。
- 相关回归：73 passed；uv run --locked ruff check . 通过。
- CLI 离线检查使用 mock API：模型失败保留 checkpoint，恢复沿用原模式/模型，
  不读取修改后的配置、不重新发现模型，也不调用 Scripted 回退。
- 2026-10-22 检查当前 Chat 配置：默认本地端点 127.0.0.1:8005 的模型发现返回 ConnectError。
  检查时主工作目录与独立 Worktree 均无其他本地模型配置。因此未执行真实模型协作，
  在线验收仍未完成；离线结果不替代这一项。
- 文档链接和命名引用检查通过。OpenHands、真实专业 Planner、医学数据及 CLI/TUI 切换仍未迁入。
