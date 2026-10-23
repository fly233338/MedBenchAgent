# 原生 Agent 与真实任务规划

## 本轮完成范围

- [x] Supervisor 改为 create_agent、ToolStrategy、只读工具和原生私有持久状态。
- [x] 模型 Planner：证据查询、Axis/Scenario 判断、任务修订、需求/接受、任务排除和正式发布。
- [x] 当前视图、原生摘要与调用限额、请求结果隔离、串行写入及恢复。
- [x] canonical Dataset 成员内容校验和可移植冻结入口。
- [x] --planner / --dataset-manifest、配置和 Prompt 冻结。
- [x] BrEaST 原始工作簿与图像的小型 canonical 样例脚本。
- [x] closed_vqa GenerationSpec 及纯确定性校验。
- [x] 清理被替代的 Supervisor JSON action 循环、重复 inspect 状态和旧模型传输测试。

## 实现边界

Compilation 只协调；专业 Planner 工具归属 agents.semantic_planner，纯领域规则归属 planning。
保留既有 OpenHands Coding、合成示例及仍有旧入口调用的业务资产。
未实现 Data Processing 包装、Generation 后端、Review/Audit/Delivery 执行链或 TUI 迁移。
不提供历史运行兼容或旧标注双格式 Reader。

TaskPlan 为唯一新规划文档；合成示例也已使用该聚合契约。
Draft 与正式发布都是不可变文档，草稿引用保存在私有 checkpoint，正式引用通过 WorkResult 交接。
DatasetManifest.members 覆盖被读取的实际文件；恢复不重新选择数据或从当前配置改写运行设置。

## 离线验证

验收使用 Fake ChatModel 驱动真正的 create_agent，不调用真实模型或实际 OpenHands 编程。
覆盖工具错误修正、持久角色 A → B → A、跨 Compilation 隔离、摘要后继续工作、
子图完成后重新打开 checkpoint 继续 collect、新请求拒绝旧结果、并行写拒绝、
调用额度耗尽、领域产物与内容篡改、生成契约错误绑定和输入/reference 分离。

已从本地 BrEaST 原始文件创建 20 条 Case 的开发快照并通过 FileDatasetReader 验证，
运行产物位于忽略目录 .runtime/breast-native-example-final/，包含 20 条 Case、44 个内容成员，
Mapping 保留原工作簿的 21 条数据字典说明及其来源位置。
这不表示医学任务语义或真实模型质量已经通过验收。

2026-10-23 收口验证：
- uv run --locked ruff check .：通过。
- uv run --locked pytest -q：357 passed，4 条第三方/既有测试警告。
- uv build --wheel：成功；确认 Supervisor、Planner、Coding 三份 Prompt 随 wheel 发布。
- 独立解释器导入 Planner 和 GenerationSpec，不加载旧 BIR、Resolver、Planner 或 Workflow。
- 原生限额恢复测试发现的 JSON 字段顺序影响提交身份问题，已统一为排序后的规范序列化并复测。
- 全部验证离线完成，未执行真实模型调用或真实 OpenHands 编程任务。

## 后续顺序

1. 用 GenerationSpec 接入一个真实生成后端，完成语义 Review、确定性 Audit 和 Delivery。
2. 将 OpenHands 放入 Data Processing 的局部修复循环，拆开候选与正式数据提交。
3. 验证下游反馈、局部返工与新版本复查，再切换旧入口和重整 TUI。
