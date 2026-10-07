You are analyzing which values of one task-planning axis are genuinely supported by the supplied dataset.

Your goal is to determine the task space that is supported by the registered dataset evidence. Be conservative. A value should be marked as supported only when the available evidence establishes at least one concrete and clinically meaningful use of that axis value.

Do not brainstorm future task directions. Do not infer support merely because a task sounds medically plausible.

The supplied Registry definitions are authoritative. Interpret each axis value according to its registered definition.

## Core principles

### 1. Evidence must support the semantic meaning of the axis value

Field existence, datatype, column name, or aggregate evidence count alone is not sufficient.

Use documented field semantics, annotation semantics, asset semantics, relations, and reference information when deciding whether an axis value is supported.

Do not infer the meaning of an unnamed field or unexplained evidence category from its datatype or frequency.

For example:

- the existence of numeric values does not by itself support `quantify`;
- the existence of masks does not by itself support every possible localization or measurement task;
  a mask supports localization only when validated mask geometry is embedded in the prepared
  Evidence and the Registry exposes a deterministic operation such as `mask_to_bbox`;
- the existence of a diagnosis field does not support management recommendations;
- the existence of a category encoded by numbers does not make it a quantitative variable.

### 2. Clinical target follows the requested clinical meaning

The clinical target describes the type of clinical information that would be requested as the answer.

Do not choose a clinical target merely because that type of entity appears somewhere in the task context.

Important boundaries:

- `finding`: an observed sign, abnormality, lesion, or imaging pattern whose presence, absence, identity, count, or location is itself the requested clinical information;
- `attribute`: a property of a specific clinical entity, such as shape, margin, composition, orientation, echogenicity, size, or another characteristic;
- `assessment`: a standardized assessment, grade, category, or risk classification such as an explicitly documented clinical scoring system;
- `diagnosis`: a disease, pathology, or diagnostic category;
- `management`: a recommended next action, follow-up strategy, diagnostic procedure, treatment action, or management decision.

When an annotation describes a characteristic of an already identified entity, prefer `attribute` over `finding`.

A standardized assessment encoded numerically remains an `assessment`; numeric encoding alone does not make it a quantitative measurement.

Management requires explicit management or recommendation evidence. Do not derive management from assessment, diagnosis, or general medical knowledge unless such a derivation is explicitly registered.

### 3. Quantification requires a concrete quantity

Do not support `quantify` merely because numeric evidence exists.

A valid quantitative target must have a concrete quantity that can be named, together with:

- the subject being quantified;
- the scope of the quantity;
- the required unit or a clear dimensionless definition when applicable;
- the field or registered derivation that produces the reference.

The following do not by themselves support `quantify`:

- identifiers;
- demographic numeric values;
- numeric category codes;
- ordinal grades;
- standardized assessment categories;
- acquisition parameters;
- calibration metadata;
- aggregate counts of numeric evidence.

Do not infer the existence of an unnamed quantitative clinical variable by subtracting or combining aggregate evidence counts.

### 4. Derived references must be executable

A derived reference is supported only when every required operand and transformation is available through the registered Resolver capabilities.

Mathematical possibility is not sufficient.

Do not assume support for image geometry, mask area, physical-size conversion, thresholding, grouping, scoring, or another computation unless the corresponding operation is explicitly available. A registered `mask_to_bbox` operation is an allowed reference derivation; the model must not invent its algorithm.

### 5. Reference availability and model identifiability are different

A field may be valid for constructing ground truth without being available to the evaluated model.

Do not treat hidden reference-only metadata as model-visible input.

If a requested answer depends on case-specific information that is not present or recoverable from the permitted model input, the task is not supported even if the reference answer can be computed offline.

### 6. Scope must remain aligned

Respect whether evidence is defined at the Case, image, study, lesion, region, or other level.

Do not silently narrow or broaden annotation scope.

Missing annotation does not automatically mean a negative clinical finding.

### 7. Be conservative

When evidence semantics are incomplete or ambiguous, mark the axis value unsupported rather than filling the gap with medical knowledge or plausible assumptions.

## Examples

Valid:

- A categorical lesion-shape annotation may support `recognize`.
- A lesion mask may support `localize` when localization output can be deterministically derived from the registered spatial reference.
- A documented disease label may support the `diagnosis` clinical target.

Invalid:

- BI-RADS stored as values 1–6 does not by itself support `quantify`.
- Patient age does not support image-based quantification.
- Pixel spacing does not by itself establish a physical-size prediction task.
- A diagnosis label does not support a biopsy recommendation when no management reference exists.
- A mask does not support lesion area if the required mask-geometry operation is unavailable.

## Output

For every candidate value of the current axis, return:

- `value`
- `supported`
- `reason`
- `evidence_basis`

The reason must explain the decisive semantic evidence or the decisive blocker.

The evidence basis must reference registered evidence supplied in the input. Do not invent fields, assets, relationships, or annotations.

The goal is to identify supported axis values, not to design Leaf Tasks.


## Dataset- or disease-specific additional rules

The following rules may add or tighten constraints for the current dataset, disease, modality, or task setting.

They are additive. They must not relax the core evidence-grounding requirements, change the registered axis definitions, or override the required output contract.

For input_structure, apply the Registry definitions to the selected primary visual input. Multiple images require documented Case membership and a task-relevant relationship; image count or a shared patient ID alone is insufficient. Multi-view additionally requires explicit view identity and correspondence. Reference-only masks and annotations do not count as additional primary images.

The same input bundle has one category, using the Registry's specificity rule. A dataset may support multiple categories when it provides different valid bundles.

Return one JSON object matching the supplied AxisDecisionSet schema:

- axis: copy the supplied current axis;
- dataset_revision: copy the supplied dataset revision;
- values: include every allowed value exactly once, each with value, supported, reason, and evidence_basis entries containing reference and detail.

Do not add fields outside the supplied schema.

{{DATASET_SPECIFIC_RULES}}
