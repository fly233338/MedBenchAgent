# Task Planning Agent

**Input:** one immutable dataset access receipt, optional user constraints, frozen Registry/extensions and any scoped revision request.

**Output:** one canonical Task Plan with Axis decisions, Scenarios, validated targets, executable Leaves, reviews and categorized exclusions.

**Private state:** evidence/read caches bound to BIR input digests, Scenario and rework round. Full dataset facts are cached on disk by binding digest. Leaf prompt context is assembled for each request; it is not checkpoint state. The canonical plan owns progress.

Plan writes update BIR `task_plan.json` and return its reference. There is no local-file fallback, checkpoint Plan copy, bulk-Axis entry point or default Registry. Extension selection is owned by `planning_extensions.json`; changing it invalidates dependent planning work.

**Model-visible context:** facts, source-backed field semantics, relations, category/value coverage, representative Cases, user constraints and validation feedback. Required/query fields are retained first. Omissions and continuation tools are explicit. Source facts are separate from requirements; model assumptions cannot create missing annotations.

**Request budget:** `planning.context_chars` covers the final serialized request, including instructions, output schema, tools, source evidence, tool data and all repair messages. The direct Axis/screening/review calls and Leaf tool loop use the same check before each model request. An oversized request fails explicitly with size/limit diagnostics; required evidence and repair output are not silently truncated.

**Tools:** catalog, exact value, metadata and saved-task paging; Leaf save/exclusion and Scenario review. Axis and screening have direct semantic calls rather than an outer model call that only chooses the predetermined tool. Open-ended Leaf design retains autonomous queries and actions.

**Feedback:** typed Resolver/capacity diagnostics repair the same Leaf. Source concerns are returned for supervision. Exclusions distinguish missing evidence, unsupported semantics and user exclusions. A read failure cannot silently exclude a Scenario.

**Recovery:** one pending Scenario queue in BIR, checked dependency digests and scoped caches. Reopened Scenarios preserve other accepted work; changed data/requirements invalidate dependent plans.

**Acceptance:** stable ordered inputs, deterministic references, complete answer domains, boundary-valid equivalent templates and approved local reviews. See [Task Planning contract](../TASK_DISCOVERY.md); tests include evidence flow, task input, masks, task plans and context budgets.
