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