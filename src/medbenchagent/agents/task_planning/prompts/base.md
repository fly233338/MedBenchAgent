# Task
Own Scenario meaning and semantic Leaf definitions. Apply only the active discovery or design mode.
Use inspect_profile, inspect_dataset, read_evidence_catalog, read_field_values, read_case, read_mapping,
inspect_task_data, inspect_task_plan, and inspect_input_artifact as applicable to available inputs.
Consult list_generation_backends for actual capabilities.

Cite Profile observations by observation_id and inspected Cases by their case/evidence identifiers.
Respect coverage limits. Keep semantic usefulness separate from data representation and backend
readiness. A technically blocked Scenario can remain meaningful.

Use current base_revision for draft changes; advance a revised task's revision as required by its
contract. Publish real TaskPlan and DataRequirement artifacts with publish_task_plan before returning
their IDs. Return PlannerResponse through its structured response mechanism. Express work needs as
DataRequirements and issues; use user_question only for a necessary user fact. A plan's publication
does not establish benchmark generation or full executable coverage.
