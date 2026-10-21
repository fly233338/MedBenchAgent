# Independent semantic review

**Input:** a canonical discovered/complete Task Plan, the same dataset evidence interface used for design, and unchanged user requirements.

**Output:** a BIR review document with checked Scenario IDs, findings, ownership and approved/rejected status. The reviewer cannot modify the plan or prepared data.

**Private state:** each semantic call has a fresh context. Checked-Scenario progress is persisted against exact plan/data/requirement/Registry dependencies.

**Model-visible evidence:** complete Leaf definitions and deterministic validation summaries, source semantics, representative Cases, validated targets and cross-Scenario task summaries. Excluded non-structural Scenarios can be screened independently for missed coverage.

**Feedback:** task/coverage findings belong to Planning. Source concerns identify known fields and evidence references and belong to Preparation. Blocking findings prevent publication; Supervisor routes the repair.

**Recovery:** current checked Scenarios are reused. A changed plan invalidates the review receipt, so an old approval cannot authorize a new candidate.

**Acceptance:** exact Scenario and Leaf coverage, source evidence present in the review payload, rejected review preventing publication and bounded targeted revision. Integration tests use scripted reviewers over real prepared image fixtures.
