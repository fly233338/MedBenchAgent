# Generation（后续接入）

## 目标与输入输出

消费 SemanticTaskSpec、Case 取样和任务派生数据，生成 Benchmark Item。
Backend 接受 task_spec、data_refs、output_directory，返回产物引用。
closed_vqa、open_vqa、report_generation、visual_grounding、detection
目前仅定义 Task Family 语义，不实现全部 Backend。

## 状态与边界

确定性生成可作为普通工具；自主 Generation Agent 仍通过 WorkRequest/WorkResult 接入。
独立工作优先 per-invocation，不强制长期会话。模型输入和参考证据必须分开。

## 反馈、恢复与实施状态

数据不足或格式不满足返回结构化结果，不捏造样本。
后续只实现实际副作用需要的幂等性，首期不建设发布、审计或版本体系。
旧 generation 是参考资产，新 Core 不调用其 Resolver/BIR 路径。
