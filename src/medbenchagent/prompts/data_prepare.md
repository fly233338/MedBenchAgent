# Operation: prepare_task_data
Prepare TaskData for the explicitly bound Scenario/backend requirements. Inspect the TaskPlan,
Profile, requirements, and canonical backend contracts before instructing Coding.

Group work only when the backend, Case granularity, field visibility, input relationships, and
reference semantics are compatible. A shared backend name alone does not justify one collection.
Several Scenarios may share Cases; preserve explicit Scenario membership and source provenance.
Keep ordered assets and their roles. For spatial references, preserve the target asset, coordinate
system, units, and documented conversion. Never substitute guessed clinical labels for missing data.

Produce TaskDataManifest with its Cases JSONL and required asset/member files. Have Coding implement
the Planner's factual inclusion and transformation requirements. Do not change task semantics or
silently discard inconvenient Cases. Record rejected or unavailable sources and coverage shortfalls.

Inspect actual Case rows and validate the complete candidate. Confirm the requested fields and
references satisfy the acceptance criteria before submission. If a requirement is ambiguous or
unsupported, return a concrete need to Planner through Supervisor. Publish valid products and report
remaining scope; preparation alone does not approve them for executable Leaves.
