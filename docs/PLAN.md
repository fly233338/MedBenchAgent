# Compilation Core 实施计划

## 状态

- [x] 创建正常 refractor 分支及独立 Worktree，保留原工作区用户修改。
- [x] 第一步：架构、数据、任务和 Agent 文档先行；文档独立提交后开始代码。
- [ ] 第二步：最小协作契约、Case 类型、DatasetReader 与任务契约。
- [ ] 第三步：独立协作图、合成示例、一个多轮协作测试及一个最小恢复测试。
- [ ] 后续：接入真实 Supervisor、OpenHands、Planner 和业务能力。
- [ ] 后续：切换 CLI/TUI，删除被替代实现。

本期范围止于第三步。真实模型和医学数据接入不属于完成声明。

## 提交顺序

1. 文档：ARCHITECTURE、DATA_LAYER、TASK_DISCOVERY、PLAN、各 Agent 文档。
2. 契约/Reader：纯类型、文件引用、小型 JSON/JSONL 读取。
3. 骨架/验证：节点注册、决策循环、checkpoint、独立入口及两个测试。

只提交本次工作，不混入原工作区 AGENTS.md 修改。

## 验收

多轮协作测试使用同一 Graph 和两种初始证据条件，由 Scripted Supervisor 选择不同顺序。
覆盖重复调用、需求、实际文件产生、新证据改变草稿及私有状态隔离。
图不写死业务顺序；Scripted 结果不代表模型自主性。

恢复测试在已有产物后 interrupt 并关闭进程，新进程用同一 SQLite、Compilation ID
继续，检查私有进度、请求、引用及已 checkpoint 工作未重跑。
不增加真实调用次数要求或大量故障注入。

运行相关检查并复用已有数据类型测试。Core 不导入 Workflow、Resolver、
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

旧代码可供参考，但 Core 不通过旧 Workflow 执行。
不建立兼容层、通用 Runtime、队列、新 Resolver DSL 或 Artifact 版本系统。
