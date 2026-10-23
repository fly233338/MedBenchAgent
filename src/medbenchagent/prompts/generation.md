# Operation: generate
Assemble benchmark items from published Leaves and their accepted TaskData using the selected
backend. Read exact bound TaskPlan, TaskData, and benchmark_request artifacts. Inspect current backend
contracts with list_generation_backends and concrete data with inspect_task_data.

Use prepare_task_execution to bind a Leaf without changing its meaning. Honor the Host-validated
Scenario population. Case selection controls sampling, not semantic eligibility: do not remove Cases
to conceal missing required fields, incompatible references, or unsupported labels. Report those
failures for Planner or Data Processing work.

Use allocate_generation_budget with the unchanged requested total, then generate_task_batch for
allocated slices. Inspect outputs through inspect_generation_product. Preserve the Leaf's templates,
input order, references, options, and output specification. Never generate missing medical answers
or put reference-only data into evaluated-model inputs.

Quarantine defective items with quarantine_generated_items and retain the reasons. Batches selected
for finalization must have disjoint attempted task/Case pairs, including failed or quarantined attempts.
Select a replacement set instead of superseded overlapping batches; retain superseded artifacts as
history. Semantic repairs return to Planner; missing
representations can be reported through propose_data_requirement.

Use finalize_generation to publish a real candidate. Return WorkResponse selecting actual artifact IDs;
use its existing summary and issues fields to describe the candidate's accepted quantity, failures,
and shortfall. A partial candidate is not full goal completion, and a
finalized candidate is not an independently reviewed or published delivery.
