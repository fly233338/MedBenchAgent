You are reviewing one candidate Scenario produced from the supported task-planning axes.

The Scenario's input structure, operation, and clinical-target type are fixed.

Your goal is to determine whether the supplied dataset supports at least one concrete clinical target under this exact Scenario.

Scenario Screening does not design future Leaf Tasks and does not brainstorm possible task directions.

Keep the Scenario only when at least one atomic, evidence-grounded clinical target can be identified.

A medically plausible question is not sufficient.

## 1. Respect the fixed Scenario

Do not reinterpret the Scenario in order to make it feasible.

In particular:

- do not change the operation;
- do not change the clinical-target type;
- do not change the input structure;
- do not justify one target type using evidence for another target type.

The clinical target is determined by the requested clinical information itself, not by another entity mentioned in the context.

For example:

- measuring a property of a diagnosed lesion does not make the target `diagnosis`;
- a numeric BI-RADS category remains an `assessment`;
- a property of a lesion is normally an `attribute`, not a separate `finding`;
- a management action requires management evidence.

## 2. Identify concrete clinical targets

A valid clinical target must satisfy all of the following.

### Atomic

It represents one clearly defined clinical variable.

Avoid broad targets such as:

- lesion characteristics;
- clinical assessment;
- morphology;
- disease severity in general.

### Axis-consistent

Its clinical meaning must match the Scenario's fixed `clinical_target`.

### Subject-bound

The clinical entity or Case to which the target applies must be explicit.

### Scope-bound

The annotation scope must be supported by the evidence, such as Case, image, study, lesion, or region.

Do not silently change scope.

### Reference-grounded

The exact field, asset, or registered deterministic derivation used as reference must be identifiable.

### Input-compatible

The permitted model input must contain enough information to make the target a legitimate prediction problem.

Ground-truth computability alone does not establish input feasibility.

### Executable

If the reference is derived, every required transformation must exist in the registered Resolver capabilities. A prepared mask Evidence may support a localization target only when its validated geometry and the registered `mask_to_bbox` operation are available.

Do not accept a target because a human could theoretically calculate it.

## 3. Quantitative targets

Numeric storage does not by itself make a target quantitative.

Do not create quantitative targets from:

- category codes;
- ordinal assessment categories;
- demographic variables;
- IDs;
- calibration metadata;
- unexplained numeric evidence;
- aggregate evidence counts.

For `quantify`, a concrete quantity, subject, scope, unit or dimensionless definition, and valid reference rule must be available.

If a physical quantity depends on case-specific calibration that is hidden from an image-only model, exclude that target.

## 4. Categorical mappings and derived labels

Do not invent:

- thresholds;
- merged classes;
- benign-versus-suspicious groupings;
- score groupings;
- label mappings;
- arithmetic scoring rules.

Such transformations are valid only when explicitly documented or registered.

For example, an assessment field containing categories 1–6 does not authorize grouping selected categories into a new binary label unless that mapping is explicitly provided.

## 5. Diagnosis and management

A diagnosis target requires an authoritative diagnostic reference supported by the dataset semantics.

Do not convert imaging descriptors or standardized assessments into a diagnosis unless a diagnosis reference actually exists.

A management target requires an explicit recommendation, next-step, follow-up, procedure, or treatment reference.

Do not infer management from assessment, diagnosis, or general clinical knowledge.

## 6. Decision rule

Return `keep` only if at least one concrete validated clinical target exists.

Return `exclude` when:

- no concrete target exists;
- the only available target requires unsupported inference;
- required information is unavailable to the model input;
- required derivation is not executable;
- evidence scope does not align;
- the proposed interpretation changes the fixed Scenario.

When uncertain, prefer `exclude`.

## Output

Return JSON only.

For a kept Scenario, provide:

- `decision`: `keep`
- `reason`
- `supporting_fields`
- `evidence_basis`
- `validated_clinical_targets`

Each item in `validated_clinical_targets` must contain:

- `target_id`: stable identifier unique within the Scenario;
- `name`: concise noun phrase naming the clinical variable;
- `target_type`: exactly the Scenario's `clinical_target`;
- `subject`: entity or Case to which the target applies;
- `scope`: supported annotation scope;
- `reference_kind`: `direct` or `derived`;
- `reference_fields`: exact names from the provided Evidence Catalog. `derived_capabilities` paths (for example `mask_geometry.area_fraction`) are capability coverage summaries, not Evidence names and must never be used here.

`reference_fields` may only contain names listed in `allowed_reference_fields` and must match the provided Evidence Catalog exactly. For a derived target, use the registered mask or other reference Evidence name; the downstream Resolver reads the requested feature (such as `area_fraction`) from that Evidence.

The target `name` must not be a task instruction.

Good:

`BI-RADS assessment category`

Bad:

`Classify the BI-RADS category from the ultrasound image`

When `decision` is `keep`, at least one validated clinical target is required.

When `decision` is `exclude`, do not invent placeholder targets.

Do not return task directions, future question ideas, answer-option suggestions, or proposed Leaf Tasks.


## Dataset- or disease-specific additional rules

The following rules may add or tighten constraints for the current dataset, disease, modality, or task setting.

They are additive. They must not relax the core evidence-grounding requirements, change the fixed Scenario, or override the required output contract.

Apply the Registry input-structure definition to the images actually selected. For multiple images, verify documented Case membership, task-relevant relations and alignment between the selected images and the target reference. For multi-view, also require explicit view identity and correspondence. A Case-level label must not be silently reassigned to an unrelated image or lesion.

## Additional examples

- A recognize / attribute Scenario has separately documented shape and margin references: these can be two separate atomic targets when both are supported. Do not combine them into a vague target such as lesion characteristics.
- Two linked time points of the same tracked lesion can form a multi-image input when the target and references apply to those images. Different time points do not establish multi-view without documented view correspondence.

{{DATASET_SPECIFIC_RULES}}
