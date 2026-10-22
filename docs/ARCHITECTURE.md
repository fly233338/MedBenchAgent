# MedD2B Compilation 架构

本文是模块、依赖和状态归属的入口。图示对应当前实现，施工与验证进度见
[PLAN](PLAN.md)；交接细节见 [Agent 契约](agents/README.md)，命令与配置见 [README](../README.md)。

## 1. 模块与代码依赖

箭头表示主要的 **import / 使用依赖**，不是运行顺序；不穷举纯类型导入。`compilation` 包中的通用协调模块与
具体 Agent 适配分别分组；领域模块不反向依赖 Agent 或协调图。

```mermaid
flowchart TB
    subgraph Entry[独立 Compilation 入口]
        CLI[compilation.__main__]
        Run[compilation.run<br/>运行记录与配置 / Prompt 冻结]
        Assembly[compilation.assembly<br/>按模式装配 Agent]
        Demo[compilation.demo<br/>合成输入与 Scripted 实现]
    end
    subgraph Core[通用协调与交接]
        Graph[compilation.graph<br/>LangGraph State / 路由 / 收集]
        Interfaces[compilation.interfaces<br/>AgentSpec / 输入输出 / Context / View]
        Contracts[compilation.contracts<br/>WorkRequest / WorkResult / ArtifactRef]
        Artifacts[compilation.artifacts<br/>登记 / 内容校验 / 读取]
    end
    subgraph Agents[具体策略与运行适配]
        Supervisor[compilation.supervisor]
        Coding[compilation.coding<br/>Host Adapter]
        Worker[compilation.coding_worker<br/>独立 SDK 进程]
    end
    subgraph Services[领域能力与底层服务]
        Prompts[prompts<br/>静态 Markdown 与包资源加载]
        Settings[settings<br/>模型与执行配置]
        LLM[runtime.llm<br/>模型传输 / 请求预算]
        Logs[runtime.construction<br/>现有关联日志函数]
        Data[data.dataset / planning.contracts<br/>Reader 与任务契约]
        SDK[OpenHands SDK / Tools]
        LG[LangGraph / SQLite Saver]
    end
    CLI --> Run & Assembly & Graph & Demo
    Run --> Settings & Prompts
    Assembly --> Run & Interfaces & Supervisor & Coding & Demo
    Graph --> Interfaces & Contracts & Artifacts & LG & Logs
    Interfaces --> Contracts
    Artifacts --> Contracts
    Supervisor --> Interfaces & Artifacts & LLM & Logs
    Coding --> Interfaces & Artifacts & Worker & Logs
    Worker --> Contracts & SDK
    Demo --> Interfaces & Artifacts & Data & LG
    LLM --> Settings & Logs
    Logs --> Settings
    subgraph Legacy[仍在使用的旧入口与待迁移资产]
        OldCLI[medd2b CLI / TUI] --> Workflow[旧 Workflow / Preparation / Resolver / BIR]
    end
```

新 Graph 不导入具体 Agent 或旧业务链。Assembly 只装配已实现的能力，不生成数据；
当前独立 CLI 明确创建合成输入，Planner 仍选用 Scripted 实现。旧入口暂未切换。

## 2. 运行时协作与恢复

箭头表示 **运行时交接或持久化**。Supervisor 每次动态选择一个 Agent；没有固定的
Coding → Planner 业务流水线。内部 inspect 查询只读协调记录与正式产物。

```mermaid
flowchart LR
    Input[目标 / 初始 ArtifactRef] --> S[Supervisor]
    S -->|Delegate / checkpoint WorkRequest| A[选中的具名 Agent 节点]
    A -->|WorkResult| Collect[Host 校验并收集]
    Collect --> S
    S -->|inspect| Read[已登记目录 / 工作记录 / 有界预览]
    Read --> S
    S -->|AskUser| Pause[LangGraph interrupt]
    Pause -->|Command resume| S
    S -->|Finish| End[END]
    A -->|候选文件| Files[Host 复制 / 摘要校验 / ArtifactRef]
    Files --> Collect
    S -.-> Checkpoint[(SQLite checkpoint)]
    A -.-> Checkpoint
    Collect -.-> Checkpoint
    Frozen[(run.json<br/>模式 / 模型 / Prompt 快照)] -.-> S
    Frozen -.-> A
```

同请求重放沿用请求和 Workstream 身份；新请求可续接同一 Workstream。正式文件按请求
独立保存，修改形成新的交接。Checkpoint 是协调事实来源，日志不用于恢复决策。

## 3. Agent 内部结构

每个 Agent 选择适合自身的普通节点、子图或外部 Runtime；图中只展示当前实现。
Tool 和内部子图归属 Agent，复用的确定性业务能力归属领域模块。

```mermaid
flowchart TB
    subgraph SupervisorAgent[Supervisor]
        SP[冻结指令 + 目标 / 回答 / 有界上下文] --> SM[模型决策]
        SM -->|inspect| SQ[本 Agent 的只读查询]
        SQ --> SP
        SM --> SD[Delegate / AskUser / Finish]
    end
    subgraph CodingAgent[Coding]
        CW[WorkRequest + 冻结目标背景] --> CH[Host 解析输入 / 选择 Workstream]
        CH --> OH[OpenHands Conversation]
        OH --> Tools[Terminal / File Editor / FinishTool]
        Tools --> OH
        OH <--> Private[(本 Workstream 的 SDK 事件与 Workspace)]
        OH --> Candidate[结构化候选结果]
        Candidate --> Publish[Host 校验并交接正式文件]
    end
    subgraph PlannerAgent[当前 Scripted Planner]
        PI[WorkRequest] --> PN[plan 节点 / 私有规划进度]
        PN <--> Reader[DatasetReader / DataRequirement / SemanticTaskSpec]
        PN --> PO[WorkResult]
    end
```

Coding 使用 SDK 原生 Agent Loop，不再套一层子图。Planner 使用原生 per-thread 子图。
真实 Axis/Scenario/Leaf、Generation 与医学 Skill 是后续能力，不是图中已实现组件。

## 4. 状态所有权与扩展规则

| 所有者 | 保存内容 | 位置 |
| --- | --- | --- |
| Run | 初始目标、模式、有效模型/执行参数、静态 Prompt 文本与摘要 | run.json |
| Compilation Graph | 当前交接、可查询请求/结果历史、Workstream 目录、正式引用、用户回答 | SQLite checkpoint |
| Supervisor | 单次决策的有界上下文与只读查询结果 | 调用内临时对象，不另存记忆 |
| Scripted Planner | 私有规划轮次与待满足要求 | LangGraph 子图 checkpoint |
| OpenHands | 每个 Workstream 的消息、工具事件和执行进度 | `coding/<workstream_id>/conversations/` |
| Coding Workspace | 可修改脚本及按请求划分的候选输出 | `coding/<workstream_id>/workspace/` |
| 正式 Artifact | 单个已交接文件及 SHA-256 内容身份 | `artifacts/<request_id>/` + checkpoint 引用 |
| 诊断日志 | 模型调用与执行关联标识 | events.jsonl / Coding 请求日志，不是状态账本 |

新增功能先判断所属层级：跨 Agent 协调进入 Core；专业决策、Tool 包装和内部子图进入
所属 Agent；可复用的读取、运算和生成能力进入领域模块。静态指令进入包资源，动态目标、
约束、证据和 Tool Schema 由对应代码装配。新增文件须有独立职责，不要求每个 Agent
都有相同目录，也不为普通操作建立 Subgraph 或 Manager。

Skill 仅作为 OpenHands Coding 的可选领域扩展，与 Workstream 生命周期独立；本轮不实现
Skill Registry。正式 Artifact 的摘要仅覆盖被引用文件，不递归冻结 Dataset；Local Workspace
不是操作系统沙箱，不承诺外部命令 exactly-once。

详细边界：[Supervisor](agents/supervisor.md)、[Coding](agents/coding.md)、
[Task Planning](agents/task-planning.md)、[数据契约](DATA_LAYER.md)、[任务契约](TASK_DISCOVERY.md)。
