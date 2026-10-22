# Compilation 基础架构收敛

## 当前阶段

基线为 refactor / 72a7f66。保留 Workstream、OpenHands Coding 与现有协作语义，
本轮不迁入真实 Planner、Generation、医学 Skill、Docker 或旧 CLI/TUI。

- [x] 模块审计与图示架构规范。
- [x] 公共接口、运行记录、单一 Assembly、Prompt 资源与冻结。
- [x] Supervisor 有界上下文与私有 inspect 查询、可查询工作记录。
- [ ] Artifact 内容校验、执行关联、清理与回归验证。

## 模块调整

| 当前职责 | 目标模块 | 原因 |
| --- | --- | --- |
| graph 中跨模块使用的接口 | compilation.interfaces | 专业 Agent 不依赖 Graph 实现 |
| CLI 的配置与运行快照逻辑 | compilation.run | 运行记录有单独所有者，Assembly 不导入 CLI |
| CLI 与 demo_agents 的装配 | compilation.assembly | 按冻结模式创建，停止先装 Demo 再替换 |
| Supervisor / Coding 静态常量 | prompts 包资源 | 动态上下文不混入静态文件，恢复使用冻结原文 |
| 各 Agent 的上下文构造 | 保留在所属 Agent | 不建立万能 Context Schema 或 Manager |
| Artifact / 关联日志 | 原模块内增强 | 文件内容身份和诊断关联，不新增账本 |

## 施工与提交

1. Architecture 图、模块规范及本计划。保留用户 AGENTS.md 修改，只提交本轮新增约束。
2. 接口、Run、Assembly 与 Prompt/Context。Graph 历史保存 request/result，支持按 ID 查询；
   Supervisor 最终仍只返回 Delegate / AskUser / Finish。
3. Artifact digest、关联日志、打包与回归检查，清理被替代常量和定义，不留旧导入别名。

Supervisor 默认近期记录 8 条、目录页 20 条、每次决策最多 4 次 inspect；所有限额冻结。
目标与用户回答不静默截断，必需内容或查询超限明确报错并保留 checkpoint。
需求与问题是有来源的历史反馈，不根据后续 completed 自动推断旧需求已经解决。

## 验收

- 现有四种模式、跨进程恢复、Workstream 续接/隔离和正式产物行为不退化。
- 新 Agent 通过 Assembly 注册；Graph 不依赖具体实现，专业 Agent 不导入 Graph。
- 包内 Prompt 随 wheel 发布；修改源码资源后未完成运行使用原快照，新运行使用新文本。
- 大量历史/目录下实际模型请求有界，旧 Artifact、Workstream 和反馈仍可通过 inspect 找到。
- Artifact 文件变化在读取与交接时被识别，不重新登记摘要掩盖修改；不变内容可重放。
- 现有日志与 SDK 事件可通过标识关联，显式模型设置不触发旧配置加载，不记录凭据。
- 复用现有 Scripted / Fake SDK 测试，执行相关 Ruff / Pytest；公共契约修改后跑全量回归。
- 本轮不要求真实模型 API，不增加历史 Schema 或运行兼容分支。

## 本阶段验证记录

- 接口/装配/Prompt/Context 阶段：57 项相关测试通过，Ruff 通过。
- 1000 条历史与目录的请求预算、旧反馈/产物查询、单条目录页、多次 inspect 和查询超限已覆盖。
- 修改源码 Prompt 后，已有运行从冻结文本恢复，新运行读取新文本；已验证快照摘要损坏被拒绝。
- uv build --wheel 成功；主环境和独立 Coding Python 均从 wheel 读取两份 Prompt 资源，
  资源加载不导入 LangGraph 或 OpenHands。此验证不调用模型。
- 后续补齐 Artifact / 关联日志检查与全量回归；离线结果不能代替在线模型或医学 Benchmark 验收。
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

## Workstream Coding 阶段验证记录

- Workstream 目录与请求同 checkpoint，支持创建、续接、跨 Agent 校验和按历史生成摘要；
  scope 不参与会话身份。协调图没有新增固定业务阶段或独立 Session Manager。
- Local OpenHands Adapter 使用 SDK/Tools 1.49.6、每个 Workstream 的独立目录与原生
  FinishTool 结构化结果。正式文件由 Host 复制登记；SDK 消息不进入父图或另一个状态账本。
- 相关回归：57 passed，覆盖 A → B → A、跨 Compilation 隔离、产物复制、非法路径、
  内容冲突、缺失输入、环境不可用、两个 SDK/Graph 恢复窗口和四种入口组合。
- uv run --locked pytest -q：341 passed（4 条第三方/既有测试警告）。
  uv run --locked ruff check .：通过。
- 本机原生 SDK 离线检查确认 Conversation 保存/重新加载、FinishTool 恢复、PowerShell
  和 File Editor 执行，以及 SDK 持久化未写入提供的测试凭据。修正了旧 worker 示例中的
  过期 Condenser 导入路径，并在 Windows 显式使用 PowerShell，避免自动选到 tmux。
- 额外以本地模拟 HTTP 模型运行真实 SDK 子进程：3 次模拟模型响应驱动 File Editor 创建
  calculate.py、Terminal 执行并产生 answer.txt；Host 交接两份文件，同请求重放未再次调用模型。
  本机记录位于 .runtime/coding-transport-5b65d495/（忽略目录，不提交）。这不是在线模型验收。
- SDK 内置 PowerShell 探测有固定 5 秒超时；与其他检查并行时曾出现一次不可用反馈，
  单独复测通过。本轮没有改写第三方 SDK 的探测机制。
- 2026-10-22 再次检查默认 Coding 模型端点 http://127.0.0.1:8005/v1，模型发现仍为
  ConnectError。未完成真实模型 Smoke Test；离线结果不替代在线协作或医学数据验证。
- 用户已有 AGENTS.md 修改保留。已替换旧 agent_mode/单模型快照字段和单 Conversation 假设；
  仍有调用方的旧 Runtime 未删除，Planner、医学数据、Docker、Skill 系统及 CLI/TUI 迁移仍属后续。
