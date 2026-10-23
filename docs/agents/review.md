# Review Agent

agents.benchmark_review 使用独立 create_agent 审查 TaskPlan 或 generation_candidate。
它读取显式绑定的任务、准备数据和题目，只写自己的 Audit/Review 产物，不修改被审查对象。

工具包括 inspect_artifact、inspect_task_case、inspect_candidate、run_candidate_audit 和 finalize_review。
语义 findings 保存实际对象 ID、问题类别、证据 Artifact ID、阻断状态和建议责任方。

候选审查必须有针对同一内容身份的完整 Audit。批准需要：
- Audit 通过，且检查数量等于当前候选数量。
- 覆盖当前任务和 Scenario，至少检查每个有生成结果任务的一条真实题目。
- 没有阻断 findings 或未完成审查范围。

Report 记录实际 checked_item_ids，Host 标明是否为语义抽样。
TaskPlan 审查不要求候选 Audit，但批准仍需覆盖声明的任务和 Scenario。
旧候选通过不授权新候选，Review 的 completed 仅表示审查工作结束。

本轮完成能力和基础角色 Prompt；具体何时启动 Review、如何安排多轮复查由后续工作流 Prompt 决定。
不执行对被评测模型回答的 LLM-as-judge，也不声称替代医学专家验收。
