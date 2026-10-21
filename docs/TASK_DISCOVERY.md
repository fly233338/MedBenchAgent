# Task Planning contract

The canonical product is `datasets/<dataset>/planning/task_plan.json` inside a compilation BIR. The plan binds the dataset receipt, compilation requirements and frozen Registry. Derived axis/scenario/leaf exports are not a restore source.

## Semantics and stages

The Registry defines Axis values and operation meanings. Models assess supported values; code constructs structurally feasible Scenarios. Screening validates atomic clinical targets and reference fields. Leaf design binds one validated target to inputs, a deterministic Resolver, output contract and equivalent question templates. Review checks task meaning, source grounding, wording, boundary applicability and duplication.

The canonical phases are access, axes, screening, leaves, complete and blocked. `pending_scenario_ids` is the one active queue. During screening it contains unreviewed candidates; during leaves it contains Scenarios requiring design/review. The model cannot bypass validation by changing a phase.

No explicit clinical goal means evidence-driven design. Explicit goals constrain screening and design, but do not imply a positive-case population filter or allocation preference. Exclusions distinguish insufficient evidence, unsupported tasks and user exclusions.

## Evidence interface

Planning constructs bounded packets from verified dataset facts, frozen annotation metadata, relations, observed values and representative Cases. Current target/input fields and actively queried fields have priority. Omitted fields/examples/tasks are explicitly reported; required evidence exceeding the configured budget raises a diagnostic rather than disappearing.

Tools expose catalog pages, exact observed-value pages, annotation-metadata pointer/text pages and saved Leaf pages. Long value excerpts are explicitly marked as excerpts, never answer labels. Representative samples cover observed categories, missingness, relations and numeric boundaries; a bounded sample is not proof of complete dataset coverage.

Axis and screening receive their own source facts. Design and independent review share `scenario_evidence_context`; review gets original semantics and representative Cases, not only a generated Leaf summary. Dynamic Leaf evidence is injected once by the planning adapter. The generic harness has no planning-specific context logic.

## Deterministic contracts

- Model inputs and reference evidence are separate. Mask geometry and other immutable reference features never become model inputs.
- Input role order comes from Resolver declarations and the frozen Sampling manifest.
- `eligible_count` counts successfully resolved SamplingUnits. Scenario structural capacity counts Cases.
- Closed-label options cover the Resolver answer domain; missing and explicit negative values remain distinct.
- Explicit `map_label` transformations are validated; unmapped values stay ineligible.
- Template count comes from configuration. Generation records the seeded template choice; Audit recomputes it.
- Duplicate Leaf semantics are detected across Scenarios.
- An excluded Scenario has a reason and evidence. All-excluded plans can be valid plans, but cannot publish an empty benchmark.

## Revision and review

Supervisor can reopen named non-structurally-excluded Scenarios with a reason and current issue IDs. A revision preserves unrelated leaves, records its history atomically in the plan and invalidates dependent allocation, generation, review and candidate receipts. The per-Scenario rework budget is enforced.

Independent review records findings and checked Scenarios in its own BIR document. Findings identify their owner and whether they block delivery. Source concerns require known fields and evidence references. Publication cannot override a rejected semantic review.

See [Task Planning Agent](agents/task-planning.md) and [Review](agents/review.md).
