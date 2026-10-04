# Axis Support Assessment

## 1. Purpose

Assess every allowed value of the current Axis using the supplied
dataset evidence and benchmark requirements.

Determine support at the Axis level. Do not construct Scenarios or
design complete Leaf Tasks.

## 2. Meaning of Support

Mark a value as supported when the evidence establishes at least one
clinically meaningful task possibility that:

1. Fits the registered meaning of the Axis value.
2. Can be answered from the permitted benchmark input.
3. Has a reference that is directly annotated or deterministically
   derivable through supported operations.

A final answer label does not need to exist in the original dataset.

Field presence, multiple files, or medical relevance alone is not
sufficient evidence of support.

## 3. Assessment Procedure

For each allowed value:

1. Read its Registry definition.
2. Identify relevant entities, properties, input assets, and explicit
   relationships in the supplied evidence.
3. Check whether these facts support the intended model capability.
4. Identify the reference evidence or a supported derivation route.
5. Record the decision with specific evidence references and a concise reason.

Assess each value independently. Do not require every possible combination
with values from other Axes to be feasible.

## 4. Constraints

1. Follow the supplied benchmark input policy. A Registry value does not
   itself authorize an otherwise prohibited input type.
2. Distinguish image-grounded evidence from clinical context, administrative
   metadata, and calibration information.
3. Annotation masks and reference overlays are not additional primary images.
4. Do not infer pairing, temporal order, correspondence, or comparability
   from field counts or shared patient identity alone.
5. Do not infer absence, completeness, or zero counts from missing annotations.
6. Derived references require available evidence and supported operations;
   do not assume an unimplemented transformation exists.
7. Cite only evidence references supplied by the planning context.

## 5. Output

Return one AxisDecisionSet matching the supplied schema:

- `axis`: the current Axis.
- `dataset_revision`: the supplied evidence revision.
- `values`: exactly one decision for every allowed value.

Each decision must contain `value`, `supported`, `evidence_basis`, and `reason`.

When `supported` is false, distinguish in `reason` between:
- an explicit incompatibility; and
- insufficient evidence to establish support.

Do not add new status values or extra fields. Return JSON only.
