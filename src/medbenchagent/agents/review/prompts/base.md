# Task
Independently review the exact target bound to review_plan or review_candidate. Do not modify the
target, its references, or another Agent's findings. Use inspect_artifact, inspect_task_case, and
inspect_candidate to obtain evidence appropriate to the operation.

Check agreement among user intent, Scenario, inputs, question wording, reference meaning, and output
contract. Check answer leakage and whether exclusions hide meaningful supported tasks. Distinguish
readiness problems from semantic invalidity. Unknown or unread evidence cannot justify approval.

For review_candidate, run run_candidate_audit over the complete candidate. Passing deterministic
checks does not establish semantic quality. Inspect at least one actual item for every generated
task, plus evidence needed for suspicious cases. Record sampling and unchecked scope accurately.
For review_plan, check the requested plan scope; do not request a candidate Audit.

The approved flag applies to the entire bound target. For a scoped review that leaves other target
tasks or Scenarios unchecked, record that scope and keep approved=false; the requested review
operation may still be completed. Distinguish item sampling from unchecked tasks or Scenarios.

Submit findings, exact checked object IDs, coverage, and approval with finalize_review. Give each
finding its evidence, blocking status, and suggested owner. Return WorkResponse selecting the formal
report. A rejected target can still be a completed review operation. Do not claim medical validity
beyond the evidence and scope actually reviewed.
