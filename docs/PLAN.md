# 全部 Agent 能力接入与必要接口调整

## 已实现

- 五个 create_agent 角色：Supervisor、Data Processing、Task Planning、Generation、Review。
- Data Processing 拥有内部 OpenHands Coding；执行候选与正式数据提交分离，可在同一委派中多次执行。
- 原始 SourceManifest、可独立规划的 DatasetProfile、后端共享 TaskData / cases.jsonl。
- 场景级规划发布、Scenario 后端选择、数据需求范围和 TaskData 接受。
- Leaf 字段绑定与组装定义；GenerationSpec 改为执行绑定和参数。
- VQA（闭集/开放）、报告、定位、检测的确定性组装、数量分配、批次和隔离。
- 独立语义 Review、确定性 Audit、实际内容绑定与可移植 Delivery。
- 五角色配置与 Prompt 冻结、--all-agents / 单角色开关、原始来源入口和数量目标。
- BenchmarkItem 统一使用有序 inputs，直接消费者已同步调整，不保留 input_images 双格式读取。

## 本轮边界

Graph 继续只协调。基础 Prompt 仅包含职责、工具用途和提交/证据约束，不固定跨 Agent 调度顺序。
旧 Workflow、BIR、Resolver、Preparation 和旧 CLI/TUI 保留给当前调用方，统一清理及新工作流
Prompt 编排属于下一轮。本轮不声称实际模型已经能稳定自主完成所有协作路径。

新工具和领域操作有真实实现，不注册空后端。原始格式转换仍由 Coding 按当前需求编程完成；
生成后端只消费准备好的 TaskData。Local SDK 不承诺操作系统沙箱或任意命令 exactly-once。

## 轻量验证

新增两个关键离线用例：
1. Profile（无规范化样本）→ 场景级计划 → Data Processing 候选验收 → 多场景共享 TaskData
   → Leaf → 五类任务组装；同时检查全部角色装配及原生工具/交接调用。
2. 完整候选 Audit/Review → 可移植 Delivery 与独立读取；拒绝对新候选复用旧报告。

替换被 GenerationSpec 新契约取代的单一 closed_vqa 测试。复用直接受影响的旧 Coding、
Planner、生成、Audit、题目输入和交付检查，不运行全量 pytest 或真实模型/真实 OpenHands 编程。

2026-10-23 收口检查：
- 两个新增能力用例通过；与直接受影响的既有检查合计 17 项通过。
- 后续局部调整只复测相关能力/Coding 检查，未运行全量 pytest。
- uv run --locked ruff check . 通过；独立入口帮助和文档相对链接检查通过。
- uv build --wheel 成功，确认六份启用角色/Coding Prompt 随包发布。
- 独立解释器导入新增 Agent 不加载旧 BIR、Resolver、旧 Planner、旧 Review 或 Workflow。
- 未调用真实模型、真实 OpenHands 编程或大规模数据处理；能力用例使用小型合成文件。

## 下一轮

统一处理旧入口及遗留依赖，整理角色与跨 Agent 工作流 Prompt，之后观察真实运行并调整协作策略。
保留用户已有 AGENTS.md 修改，本轮提交只包含本次能力和接口变更。
