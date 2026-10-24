```markdown
1. runtime.graph_runner.execute_compilation
2. compilation.assembly.assemble_compilation
3. compilation.graph.build_compilation_graph
4. compilation.graph.supervisor
   → agents.supervisor.agent.ModelSupervisor.__call__
       → Delegate(data_processing, profile)
5. agents.data_processing.agent.node
   → agents.data_processing.tools.inspect_sources
   → agents.data_processing.tools.invoke_coding
       → coding.adapter.OpenHandsCodingAgent.execute_candidate
           → coding.worker.main / OpenHands SDK
   → agents.data_processing.tools.submit_data_artifacts
6. compilation.graph.collect
   → 登记 DatasetProfile
7. compilation.graph.supervisor
   → Delegate(task_planner, discover_scenarios)
8. agents.task_planning.agent.node
   → tools.record_axis_assessment
   → tools.build_candidate_scenarios
       → planning.semantic.build_scenarios
   → tools.record_scenario_decision
   → tools.set_scenario_backends
   → tools.propose_data_requirement
   → tools.publish_task_plan
9. compilation.graph.collect
   → 登记 TaskPlan 和 Requirements
10. compilation.graph.supervisor
    → Delegate(data_processing, prepare_task_data)
11. agents.data_processing.agent.node
    → Coding 候选 → DP 验收 → TaskData
12. compilation.graph.collect
13. compilation.graph.supervisor
    → Delegate(task_planner, design_leaves)
14. agents.task_planning.tools.accept_task_data
    → tools.save_semantic_task
    → tools.publish_task_plan
15. compilation.graph.collect
16. compilation.graph.supervisor
    → Delegate(generation, generate)
17. agents.generation.tools.prepare_task_execution
    → agents.generation.tools.allocate_generation_budget
    → agents.generation.tools.generate_task_batch
        → generation.execution.generate_batch
            → generation.backends.assemble_item
    → agents.generation.tools.finalize_generation
18. compilation.graph.collect
19. compilation.graph.supervisor
    → Delegate(review, review_candidate)
20. agents.review.tools.run_candidate_audit
    → audit.generation.audit_candidate
    → agents.review.tools.finalize_review
21. compilation.graph.collect
22. compilation.graph.supervisor
    → Delegate(delivery, publish)
23. agents.delivery.node
    → generation.publication.publish_delivery
24. compilation.graph.collect
25. compilation.graph.supervisor
    → Finish
    → compilation.policy.check_goal
```

## Run / Runtime 补强（2026-10-24）

- 删除 Experiment 多运行容器入口：一个命名 Run 绑定一个 Dataset 和一次 Compilation，重做使用 Clone。
- runs 保存配置、冻结定义、输入和正式产物；.runtime/runs 保存 Checkpoint、UI 草稿/回执、事件、规划草稿、Skill 快照及 Coding 会话。
- 上下文显式区分数据目录与运行目录；输入 Manifest 摘要冻结，Profile 派生成员随正式发布固化。
- 复用 LangChain callbacks、工具 middleware 和 SDK/provider callbacks，记录可关联的实际用量及执行事件；不建立新状态账本。
- Textual 改为 Run 列表入口、英文短文案、独立 TCSS、命令候选及 Dataset/配置/产物入口。已有问题回答、安全暂停和恢复路径保留。
- 本地 Coding 保持宿主用户权限；候选路径、摘要和来源检查由 Host 执行。本轮不实现 Docker 或 OS 级只读隔离。

验收记录：完整回归 77 项通过（保留库的 pynrrd 有一条 datetime.utcnow 弃用提示）；
后续输入路径边界修复的 28 项相关复测通过。Ruff、sdist/wheel 构建以及独立 wheel 导入与 Textual 启动检查通过。
80×24、100×35、140×45 的交互、缩放及深浅主题渲染已执行；最终英文界面文案和截图样例通过文本检查。
Chat 与 Coding 的配置端点探测均为 ConnectError，未执行真实模型/真实数据编译；本地 BrEaST 夹具的 Scripted Agent 端到端已验证。
