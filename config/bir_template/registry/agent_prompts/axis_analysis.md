# Axis Support Assessment

## Purpose

Assess every allowed value of the current Axis using the supplied dataset evidence,
benchmark requirements, and normalized field semantics. Do not construct
Scenarios or design complete Leaf Tasks.

## Meaning of Support

Mark a value supported only when the evidence establishes at least one clinically
meaningful task that fits the registered Axis meaning, the permitted benchmark
input, and a direct or registered deterministic reference route.

Field presence, numeric counts, medical relevance, or a final answer label alone
are insufficient. Treat `field_semantics` as authoritative for semantic roles;
unknown roles must remain unknown. Do not infer clinical measurements by
subtracting counts or guessing from field names.

## Assessment Procedure

For each allowed value:

1. Read the Registry definition.
2. Inspect entities, properties, input visibility, scope, units, coverage, and explicit relationships.
3. Check whether the permitted input can determine the requested answer.
4. Identify direct evidence or a supported derivation route.
5. Record specific evidence references and a concise reason.

Assess each value independently. Do not require every combination with other Axes
to be feasible.

## Constraints

1. Follow the benchmark input policy.
2. Distinguish image-grounded evidence from clinical context, administrative metadata, calibration, and reference-only evidence.
3. Annotation masks and reference overlays are not additional primary images.
4. Do not infer pairing, temporal order, correspondence, comparability, absence, completeness, or zero counts from aggregate facts.
5. Derived references require available evidence and registered operations.
6. Management requires an explicit recommendation or management reference.
7. Ordinal assessment codes remain assessment targets.

Apply all injected planning rules below.

{{PLANNING_RULES}}

## Output

Return one AxisDecisionSet with `axis`, `dataset_revision`, and exactly one
decision for every allowed value. Each decision must contain `value`,
`supported`, `evidence_basis`, and `reason`. Return JSON only.
