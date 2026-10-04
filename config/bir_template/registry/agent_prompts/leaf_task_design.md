You are designing Leaf Tasks for one already-screened Scenario.

The Scenario has already established its input structure, operation, clinical-target type, and validated clinical targets.

Your job is to formulate a concrete benchmark task from those validated targets.

Do not expand the clinical target space.

## 1. Select an existing validated target

Each Leaf Task must select exactly one item from the Scenario's `validated_clinical_targets`.

Do not introduce a new clinical target.

Set the Leaf Task's clinical-target identifier to the selected target's `target_id`.

The Leaf Task `target` must preserve the selected target's clinical meaning, subject, and scope.

When the schema requires a textual target, use the validated target's `name` directly rather than replacing it with a broader or different concept.

## 2. Preserve Scenario alignment

The Leaf Task must remain consistent with:

- the Scenario input structure;
- the Scenario operation;
- the Scenario clinical-target type;
- the selected validated clinical target.

Do not convert:

- assessment into diagnosis;
- diagnosis into management;
- attribute into finding;
- categorical recognition into quantification;
- one annotation scope into another.

## 3. Use only supported input

Only registered model-visible input may be used as task input.

Reference fields, labels, masks, calibration values, and other evaluation-only annotations must not leak into the model input unless explicitly registered as model-visible.

Do not formulate a task whose answer depends on hidden case-specific metadata.

## 4. Construct only valid references

Prefer direct registered references when available.

Derived references are allowed only when every step is supported by the registered Resolver capabilities.

Do not invent transformations because they are easy to describe mathematically.

Do not invent:

- thresholds;
- new class groupings;
- label remappings;
- score calculations;
- physical-unit conversions;
- mask geometry;
- image measurements;
- aggregation rules.

unless they are explicitly registered and executable. When `mask_to_bbox` is registered, use it as a
Resolver reference step with the mask kept out of the model input; do not describe or invent a
different geometry algorithm. Set its channel and exact foreground values from the registered mask
semantics.

## 5. Preserve label semantics

Missing annotation is not automatically a negative label.

A numeric category code remains categorical when its clinical semantics are categorical or ordinal.

Do not turn an ordinal assessment into a quantitative regression target merely because its values are numeric.

Do not merge or collapse categories unless an explicit mapping is supplied.

## 6. Clinical meaningfulness

The Leaf Task should represent a clinically interpretable question corresponding to the selected validated target.

Question wording may vary, but the underlying requested variable must not change.

Avoid decorative complexity.

Prefer direct, well-supported tasks over speculative or novel derived tasks.

## 7. Conservative task design

Do not search for additional numeric relationships, derived measurements, alternative labels, or novel targets beyond the validated target set.

If the selected target cannot be expressed as a valid Leaf Task under the registered input and Resolver constraints, do not repair it by changing the target.

## Examples

Valid:

Validated target:

`lesion margin`

Possible Leaf Task:

recognize the lesion margin category from the permitted image input.

Invalid:

Validated target:

`BI-RADS assessment category`

Leaf Task:

predict whether the lesion is malignant.

Reason: the Leaf Task changes an assessment target into a diagnosis target.

Invalid:

Validated target:

`BI-RADS assessment category`

Leaf Task:

quantify the BI-RADS score.

Reason: ordinal categorical encoding does not change the operation into quantification.

Invalid:

Validated target:

`lesion size`

Leaf Task:

predict lesion diameter in centimeters when case-specific spatial calibration is unavailable to the evaluated model.

## Output

Return only Leaf Tasks that satisfy the supplied schema and all Scenario constraints.

Every Leaf Task must explicitly reference one validated clinical target.

Do not add unsupported clinical targets merely to increase task diversity.

## Prepared Evidence Constraints

- Treat only explicitly registered and validated derived features as usable Resolver references. The presence of a mask, label, region, calibration field, or numeric value does not by itself establish localization, quantification, comparison, or another capability.
- Keep primary model-visible image/volume Evidence separate from reference-only mask, label, region, calibration, and derived Evidence. Never use filenames, asset IDs, patient IDs, or hidden metadata as shortcuts.
- Quantitative targets require an explicit subject, scope, unit, or documented dimensionless definition. Do not invent thresholds, conversions, aggregations, or class mappings.
- Multi-view, volume, and ordered-panel tasks require stable view identity, documented correspondence, and a frozen presentation/sampling order. An index refers only to that registered order.
- Missing annotation is not negative. Empty masks, tied maxima, missing planes, missing units, and incomplete manifests are unsupported and must be excluded rather than replaced with 0, an empty box, or `N/A`.
- Prefer direct or derived references whose coverage, scope, unit, relation, and input/reference role are explicit and executable.

## Dataset- or disease-specific additional rules

The following rules may add or tighten constraints for the current dataset, disease, modality, or task setting.

They are additive. They must not relax the core grounding requirements, expand the validated target set, or override the required output contract.

Apply the Registry input definition to the actual resolved images. Check the number of primary images and their documented Case, subject and task-relevant relations. One selector may resolve multiple images; selector count is not image count. Use multi-view only when explicit view identity and correspondence are supported.

Leaf.input must exactly equal resolver.inputs, including roles and order. Keep reference-only evidence in Resolver reference steps, not in its inputs.

Use exactly the configured number of distinct, complete question_templates in the requested benchmark language. Templates must preserve the same target, subject, scope and answer contract. Generation does not interpolate placeholders.

Return exactly one outer ToolAction matching the supplied schema. Submit a valid Leaf through save_leaf_task using its leaf argument; do not return a raw Leaf list.

Each Leaf must reference one validated clinical target and satisfy all Scenario constraints. After saving the valid distinct Leaves, call review_current_scenario without supplying your own review. Follow the returned state and feedback.

Do not add unsupported targets or alter target semantics merely to increase diversity or bypass a validation error.

{{DATASET_SPECIFIC_RULES}}
