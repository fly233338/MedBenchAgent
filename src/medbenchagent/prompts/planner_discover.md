# Operation: discover_scenarios
Discover supported task situations from source evidence and the user's benchmark goal. Do not
design Leaves or assume unprepared data already satisfies a backend.

Assess input_structure, operation, and clinical_target with record_axis_assessment. Distinguish
observed support from unknown relationships. Use build_candidate_scenarios to construct candidates,
then record_scenario_decision for semantic screening. Preserve evidence and reasons for exclusions.
The Cartesian product is a candidate space, not proof that every combination is meaningful.

After interpreting each kept Scenario, use set_scenario_backends to select supported output paths.
Backend selection is not an additional Axis. If a meaningful Scenario lacks an implemented backend,
keep its meaning and report the technical blocker; do not disguise that blocker as semantic exclusion.

Use propose_data_requirement to specify each required Scenario/backend representation: Case unit,
ordered inputs, field visibility, reference semantics, factual inclusion rules, provenance, and
acceptance conditions. Name Scenarios without inventing future task IDs.

Discovery completion requires every kept Scenario within the delegated scope to have an implemented
backend selection, and every in-scope Scenario/backend pair to have a concrete requirement or accepted
reusable TaskData. Preserve unresolved needs outside this scope; this operation's completion does not
resolve them. Otherwise publish useful progress and report the unmet condition. A discovery plan may
contain no Leaves.
Pending requirements remain explicit work needs after discovery completion; completed Leaf design
requires acceptance of the requirements relevant to its scope.

Interpret input_structure as what the evaluated model will actually receive. One selected slice is
single-image; an ordered spatial volume is volume. Multiple files alone do not establish multi-view
correspondence, temporal pairing, or a clinically meaningful group. Require source evidence for those
relationships and ask Data Processing to preserve them in the prepared Case.
Operation describes the primary reasoning demand: recognition, localization, quantification,
comparison, correspondence, or description. Quantification needs an annotated or reproducibly derived
reference with scope and units. Comparison needs a dimension and direction. Description needs an
actual report/reference, not a verbose restatement of one label. These are semantic guidelines,
not backend or answer-format restrictions.
Clinical target is what is predicted: acquisition, anatomy, finding, an entity's attribute, or an
assessment supported by source labels. A fixed organ name or available metadata is not by itself a
useful task. Screen candidate combinations for answerability and value rather than retaining every
Cartesian product entry. Keep unknown evidence separate from evidence of absence.
