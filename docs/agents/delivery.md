# Delivery

agents.delivery 是薄的确定性具名节点，generation.publication 实现无 BIR/Harness 依赖的打包。
不新增第六个模型 Agent。

输入必须明确选择一个 generation_candidate、匹配的 audit_report、review_report、
benchmark_request，以及候选引用的 GenerationSpec、TaskPlan、TaskData。
节点检查数量目标一致、Review 通过、无阻断发现、Audit 与当前真实内容一致。

非空且有数量不足或隔离项时，Supervisor 可在 WorkRequest.scope 中加入 allow_limited_delivery，
并在 objective/constraints 说明限制；说明原样进入报告。空候选不得发布。

默认输出可移植包：benchmark.jsonl、benchmark_manifest.json、Audit/Review 报告、
quarantine.jsonl、TaskPlan/GenerationSpec 依据、compilation_report.md 及实际输入资源。
complete.json 绑定所有文件摘要。正式 delivery_manifest ArtifactRef 指向该完成标记。
evaluation.loader.load_benchmark_package 可独立验证和读取，无需构造 checkpoint。

Finish 仍只表示协调结束，WorkResult 包装完成不改变原目标或实际 shortfall。
旧 generation.delivery 保留给旧入口；本轮不切换旧 CLI/TUI，也不定义完整调度工作流。
