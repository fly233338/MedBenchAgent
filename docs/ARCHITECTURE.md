# MedBenchAgent 架构

MedBenchAgent 用 LangGraph 协调医疗数据准备和 benchmark 构建。Supervisor 接收完整的自然语言请求，根据当前产物和可用工具选择下一步；数据处理、规划、生成和评测保持独立边界。

## 运行流程

一次请求对应一个可恢复的 Run。Supervisor 可以在同一 Run 内：

1. 检查 Catalog、已登记数据集和配置来源；
2. 通过 `DataProcessingAgent` 准备或复用数据；
3. 提取 benchmark brief，选择数据集；
4. 为每个数据集规划任务并审核模板；
5. 全局分配样本并生成问题；
6. 由 `FinalizationAgent` 验证并发布 delivery。

缺少用户事实时，当前 Agent 通过 `ask_user` 中断；回答写入同一 Run 的 checkpoint 后继续。与系统无关的请求使用 Supervisor 的 `end` 工具结束。

## 模块边界

- `runtime/`：会话、Run、checkpoint、事件、Agent harness、依赖和运行产物。
- `workflow/graph.py`：Supervisor 工具、产物就绪判断和子 Agent 委派。
- `agents/`：数据处理、任务规划、模板、分配、生成和最终化 Agent。
- `data/`：来源、annotation、图像、Profile、Card、Catalog 和确定性验证。
- `planning/`：需求提取、数据集选择、任务树、协议和样本计数。
- `generation/`、`audit/`、`rag/`、`evaluation/`：各自领域的执行和评测。

模型负责语义判断，代码负责结构校验、文件操作、产物依赖和确定性验证。共享状态只保存协调摘要和产物引用；Agent 的详细上下文写入私有状态。

## 新运行布局

新实验和 Run 使用版本化根目录 `experiments-v2/` 与 `.runtime/`。现有旧目录由当前实现忽略。

```text
<run>/
  control/
    effective_config.json
    contract.json
    artifacts.json
    events.jsonl
    operations.jsonl
  inputs/
    blueprint.json
    catalog_snapshot.json
    candidate_cards/
  agents/<agent>/<scope>/
    state.json
    semantic outputs
  workers/<dataset>/
    inputs/ work/ outputs/ state/ logs/ conversations/
```

`operations.jsonl` 保存工具调用、参数、依赖和结果，用于幂等重放。语义产物仍按 Agent 和数据集组织，便于审阅和恢复。

## 数据准备

数据准备的 canonical 产物使用 `annotation.v5`、`asset_manifest.jsonl` 和
`preparation_report.json`。Case grouping、Evidence ref 和 Asset 完整性由宿主确定性校验；
OpenHands Coding Session 只负责在 Worker workspace 中探索来源和编写可重复转换器。旧的
image-centric 产物不属于 v2 数据处理运行时。

原始来源以只读方式访问。OpenHands Coding Session 在 Worker workspace 中探索来源、编写适配器并提交候选；宿主重新核验来源、locator、总体、annotation 和图像资产，再注册 Dataset、Profile、Card 和 Catalog。

有效的已登记数据默认复用。`prepare_data` 只有在用户明确要求时才使用 `rebuild`。数据源决议和准备候选绑定当前 Run，成功后才进入 benchmark 选择。

## Benchmark 构建

`specify` 生成带证据的 Requirement。`select_datasets` 冻结候选 Card 和名称解析输入。每个 annotation.v5 数据集由 TaskPlanningAgent 完成 Axis Analysis、Task Node、Task Leaf 和 Resolver 编译，再由 v2 allocation/generation pipeline 生成并审计 `benchmark-manifest.v2`。Generation 只生成有确定答案依据的问题，Delivery 记录完整性、审计和短缺统计。

## 恢复与界面

Run 的配置、产物索引、事件日志和 Agent 状态都位于 Run 目录。GraphRunner 使用 LangGraph checkpoint 恢复；TUI 右侧显示 Run 列表、每个 Agent 的状态表和可过滤事件日志。

## 评测边界

benchmark 构建和模型推理是两个独立阶段。evaluation 读取已发布 benchmark 和图像 manifest，模型适配器、预测解析和指标计算不参与构建流程。
