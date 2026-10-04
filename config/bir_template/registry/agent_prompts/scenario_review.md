# Scenario and Leaf Task Review

## 1. Purpose

Review every saved Leaf Task in the supplied Scenario.

Determine whether each task is clinically meaningful, evidence-supported,
consistent with its contract, and ready for deterministic generation.

Do not rewrite Scenario fields or redesign Leaf Tasks.

Apply all injected planning rules below.

{{PLANNING_RULES}}

Every question template must be valid for every eligible case, including
normal, negative, and other supported boundary cases. Reject wording that
presupposes a lesion, a unique object, a fixed image count, a time relation,
or a narrower lesion-level scope when the supplied reference is Case-level or
image-level. For an image-level BI-RADS assessment, wording must refer to the
image or Case rather than assuming a visible lesion.

## 2. Review Inputs and Limits

Use the supplied Leaf definitions, question templates, input selectors,
Resolvers, reference evidence, and deterministic validation summaries.

Distinguish:

- `benchmark_model_input`: the input permitted for the evaluated model.
- Reference evidence: information used to construct or verify the answer.

The planning and review models may inspect reference evidence.
That does not authorize including it in benchmark model inputs.

Review only what the supplied material establishes. Do not claim to have
inspected rendered images, cross-Scenario tasks, or validation results
that were not provided.

## 3. Review Each Leaf in This Order

### 3.1 Scenario Alignment

1. Check the primary requested operation and clinical target.
   - `assessment` is a standardized grade, score or risk category. A Leaf must reference a validated clinical target ID from its Scenario.
   - `diagnosis` is a disease or pathology classification with a stated basis.
   - `finding` is an observed sign or pattern.
   - `attribute` is a property such as shape, margin, composition or echogenicity.
   - Assessment and diagnosis may coexist when they ask different questions.
2. Check that the input structure matches the task.
3. Check that every question template preserves the same task semantics.
4. Treat wording changes and equivalent direction reversals as templates,
   not independent capabilities.

### 3.2 Evidence and Leakage

1. Check that the target, visible-input scope, and reference scope agree.
2. Check explicit entity relationships, measurement rules, and input roles.
3. Check that all requested facts are covered by the reference.
4. Reject inference of absence, completeness, or zero from missing annotations.
5. Check that reference-only information is excluded from the intended
   benchmark model input.

### 3.3 Reference and Output Contract

1. Accept direct annotations and supported deterministic derivations.
2. Do not reject a task solely because its final answer is not a raw label.
3. Check the Resolver's operands, operations, result, and output type.
4. Check comparison direction, equality behavior, units, and A/B roles.
5. Check Task Paradigm, OutputSpec, options, and exact answer encoding.
6. If `map_label` is used, check that every source-to-output mapping is
   explicit, evidence-supported and preserves unmapped and missing values.
7. Use supplied deterministic results; do not substitute a semantic judgment
   for a reported execution failure.

### 3.4 Benchmark Validity

1. Check that the intended input is necessary for the task.
2. Reject unsupported clinical claims and obvious reference leakage.
3. Check ambiguity, redundant inputs, and unintended shortcuts.
4. Inspect the supplied answer distribution and its coverage.
   Do not infer a dataset-wide constant from a small sample.
5. Check duplicates within the supplied task set.
   Cross-Scenario conclusions require the supplied other-Scenario Leaf
   summaries. Sharing one evidence field alone is not sufficient.

### 3.5 Completeness and Consistency

1. Cover every saved Leaf Task.
2. Check the required number and semantic equivalence of templates.
3. Check positive usable capacity and unresolved validation failures.
4. For `describe`, require multiple supported observations and a coherent
   reference, not an expanded categorical label.
5. Record concrete blockers rather than general requests for improvement.

## 4. Review Decision

1. List every reviewed Leaf ID in `checked_leaf_task_ids`, including Leaves
   recommended for discard.
2. Put invalid Leaves in `discarded_leaf_tasks`, with a specific non-empty
   reason for each.
3. Approve only if at least one valid Leaf remains and no blocking issue
   remains for the retained Leaves.
4. Otherwise reject and explain the unresolved blockers in `issues`.
5. Do not discard a valid task solely because it uses a novel but supported
   composition of evidence.

## 5. Output

Return one ScenarioReview matching the supplied schema:

- `scenario_id`
- `checked_leaf_task_ids`
- `discarded_leaf_tasks`
- `status`: `approved` or `rejected`
- `issues`

Return JSON only. Do not add rewritten Leaf definitions or extra fields.
