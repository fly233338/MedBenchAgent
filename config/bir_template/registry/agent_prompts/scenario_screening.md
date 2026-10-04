# Scenario Feasibility Screening

## Purpose

Review one candidate Scenario before Leaf Task design. Determine whether the
current evidence supports one or more concrete, atomic clinical targets.
Screening constrains later Leaf design; it does not brainstorm task ideas.

Do not design Leaf Tasks, question templates, answer options, or speculative
derived measurements.

## Validated Clinical Target

Return only targets that are:

1. Atomic and specific rather than broad phrases such as "lesion characteristics".
2. Consistent with the Scenario clinical target Axis value.
3. Bound to an explicit subject and scope such as Case, image, lesion, or region.
4. Grounded in named annotation fields, assets, regions, or a registered deterministic derivation.
5. Determinable from the permitted model input; reference computability alone is insufficient.
6. Executable through the registered Resolver operations when `reference_kind` is `derived`.

Numeric category codes do not become measurements. BI-RADS and other ordinal
systems remain `assessment`. Management requires an explicit recommendation or
management reference. Missing evidence is not a negative finding.

## Decision

- `keep` requires at least one validated clinical target.
- `exclude` is appropriate when the supplied evidence establishes no viable route
  or establishes a blocking input, leakage, or reference incompatibility.

Apply all injected planning rules below.

{{PLANNING_RULES}}

## Output

Return JSON only with exactly:

- `decision`: `keep` or `exclude`.
- `reason`: a concise explanation.
- `supporting_fields`: field names supporting the decision.
- `validated_clinical_targets`: an array containing `target_id`, `name`,
  `target_type`, `subject`, `scope`, `reference_kind`, and `reference_fields`.
- `evidence_basis`: references and details supporting the decision.

When `decision` is `exclude`, `validated_clinical_targets` may be empty.
