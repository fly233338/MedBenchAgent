# Scenario Review

## Goal

Perform the final semantic review of all saved Leaf Tasks for the current Scenario.

Approve only Leaves that remain aligned with:

- the fixed Scenario;
- their selected validated clinical targets;
- prepared dataset evidence;
- permitted model input;
- Registry Operation semantics;
- Resolver capabilities;
- Task contracts;
- benchmark brief.

Do not repair an invalid task by silently reinterpreting it.

## 1. Scenario Alignment

For every Leaf confirm that it preserves:

- Input Structure;
- Operation;
- Clinical Target type.

Reject or discard a Leaf when its actual requested answer belongs to a different Operation or target type.

Examples:

- BI-RADS category is an assessment, not a quantitative magnitude;
- malignancy diagnosis is not equivalent to BI-RADS assessment;
- recommended biopsy is management, not diagnosis;
- lesion margin is an attribute, not automatically a separate finding.

## 2. Validated Target Alignment

Every Leaf must reference a target validated during Scenario Screening.

Confirm:

- `clinical_target_id` exists;
- target name has not changed meaning;
- subject is preserved;
- scope is preserved;
- reference fields remain valid.

Leaf design may not introduce new clinical targets.

## 3. Input Validity

Verify that the evaluated model receives only permitted information.

Reference-only Evidence must remain outside model input.

Reject tasks that require hidden:

- labels;
- masks;
- calibration;
- clinical metadata;
- reference answers.

Also reject tasks that become trivial because identifiers, filenames or metadata directly reveal the answer.

For multiple images, verify the documented relationship required by the Input Structure.

Multi-view requires explicit view identity and correspondence.

## 4. Reference and Resolver Validity

Every reference must be:

- directly supported by registered Evidence; or
- produced through registered deterministic Resolver operations.

Do not approve a derived reference solely because the computation is mathematically plausible.

Reject invented:

- thresholds;
- category mappings;
- score conversions;
- geometry;
- physical-unit conversion;
- aggregation;
- unregistered measurements.

Resolver inputs must match Leaf inputs.

Reference-only Evidence may participate only through reference computation.

## 5. Operation Validity

Apply the Registry definition of the current Operation.

Ensure the Leaf's actual model behavior matches it.

Check the Operation-specific answer contract, including whether answer options are required or forbidden.

Do not approve a task whose wording implies a different Operation.

## 6. Semantic Validity

Check that:

- numeric storage is not mistaken for quantification;
- missing annotation is not treated as negative;
- categorical and ordinal semantics are preserved;
- diagnosis requires diagnostic evidence;
- management requires management evidence;
- attribute/finding boundaries remain intact;
- uncertainty is not silently converted into a definite label.

## 7. Question Templates

All templates for one Leaf must be semantically equivalent.

They must preserve:

- target;
- subject;
- scope;
- Operation;
- answer domain.

Discard a Leaf when templates actually ask different clinical questions.

Templates must not leak reference information.

## 8. Capacity and Failure Summary

Use the supplied host-computed capacity and failure summaries.

Do not estimate Case counts.

A surviving Leaf must have positive executable capacity.

Inspect recurring failure reasons when deciding whether a Leaf is structurally invalid.

Do not recompute file paths or image bindings manually.

## 9. Clinical Usefulness

A task should represent a clinically interpretable variable supported by the dataset.

Technical constructibility alone is insufficient.

Do not require artificial novelty.

Prefer simple, well-grounded tasks over speculative complexity.

## 10. Duplication

Remove tasks that are semantically duplicate even when identifiers or wording differ.

Do not retain duplicate tasks simply because they arise from different Scenario combinations.

## Decision

Review every saved Leaf ID exactly once.

Approve the Scenario only when:

- at least one valid Leaf remains;
- all surviving Leaves satisfy the required checks;
- all reviewed Leaf IDs are accounted for.

For invalid Leaves, return explicit discard reasons.

If no valid Leaf remains, the Scenario should be rejected/excluded rather than repaired by changing its semantics.

## Output

Follow the supplied ScenarioReview schema.

Return:

- current `scenario_id`;
- every `checked_leaf_task_id`;
- `discarded_leaf_tasks` with specific reasons;
- final `status`;
- concrete `issues`.

Do not propose unrelated replacement tasks during review.

{{DATASET_SPECIFIC_RULES}}
