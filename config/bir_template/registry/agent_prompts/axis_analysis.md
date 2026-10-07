# Axis Analysis

## Goal

Determine which values of the current Registry Axis are genuinely supported by the prepared dataset.

An Axis value is supported only when the supplied evidence establishes at least one concrete, clinically meaningful and executable use of that value.

Do not brainstorm tasks that could exist if additional labels, metadata, relations or image processing were available.

The supplied Axis definition and allowed value definitions are authoritative.

## Procedure

For every allowed value:

1. read its Registry definition;
2. identify the dataset evidence relevant to its semantic meaning;
3. determine whether the evidence scope matches the value;
4. determine whether any required model-visible input is actually available;
5. determine whether required reference data or registered derivations exist;
6. mark the value supported or unsupported;
7. cite the decisive evidence or blocker.

Return one assessment for every allowed value exactly once.

## Evidence Semantics

Field existence is not enough.

Datatype is not enough.

Aggregate counts are not enough.

A field or Evidence category supports an Axis value only when its documented semantics establish the required meaning.

Examples:

- numeric storage does not automatically support `quantify`;
- a mask does not automatically support every localization or measurement task;
- a diagnosis field does not support management;
- multiple images do not automatically support multi-view;
- a volume file does not automatically imply that volume input is available to the model.

Use field semantics, Evidence types, image relations, reference roles and registered derived capabilities together.

## Operation Analysis

When the current Axis is `operation`, interpret each value using its Registry definition.

### Recognize

Requires a categorical, presence/absence, identity, type or attribute reference that can legitimately be predicted from permitted input.

### Localize

Requires a valid spatial or sequence reference supported directly or by a registered deterministic derivation.

The existence of a mask alone is insufficient unless the required localization representation can be produced through registered operations.

### Quantify

Requires a concrete quantity with:

- a named subject;
- defined scope;
- unit or documented dimensionless interpretation where applicable;
- direct or registered deterministic reference.

Do not treat as quantitative solely because they are numeric:

- IDs;
- demographic values;
- ordinal grades;
- standardized assessment categories;
- acquisition parameters;
- calibration metadata;
- aggregate evidence counts.

### Compare

Requires explicitly linked and clinically comparable entities or observations and a defined comparison dimension.

Two unrelated images do not support comparison merely because both exist.

### Align

Requires documented correspondence between the evidence being aligned.

Shared patient identity alone is insufficient.

### Describe

Requires evidence capable of supporting a coherent description or report.

A single isolated categorical label does not by itself establish a report-generation capability.

Use the Registry definition supplied in the current request as authoritative when it is more specific than these examples.

## Clinical Target Analysis

When the current Axis is `clinical_target`, distinguish target types by the information requested as the answer.

Examples:

- lesion presence → finding;
- lesion margin → attribute;
- BI-RADS category → assessment;
- malignancy diagnosis → diagnosis;
- recommended biopsy → management.

Do not select a Clinical Target merely because that entity appears somewhere in the Case.

Management requires explicit management or recommendation evidence.

Diagnosis requires an authoritative diagnostic reference.

## Input Structure Analysis

Classify the visual input actually available to the evaluated model.

### Single-image

One primary 2D image.

### Multi-image

Multiple distinct linked images used jointly for one task.

A shared Case, directory or patient ID alone does not establish a valid multi-image relationship.

### Multi-view

Multiple images of the same documented subject or lesion with explicit view identity and correspondence.

Multiple images are not automatically multi-view.

### Volume

A 3D input whose spatial organization and slice relationships are preserved in the actual model input.

A single extracted slice is single-image.

Separate 2D slices are not a volume merely because they came from one volume source.

## Derived References

A derived reference is supported only when every required transformation is registered and executable.

Do not infer support for:

- geometry;
- physical measurement;
- thresholding;
- class mapping;
- scoring;
- aggregation;
- comparison arithmetic;

unless the required operation and operands are explicitly available.

Mathematical possibility is not sufficient.

## Model Input vs Reference Availability

Reference information may be valid for scoring without being visible to the evaluated model.

Do not mark an Axis value supported when the task would require hidden case-specific information unavailable through the permitted model input.

Example:

A reference mask plus hidden pixel spacing may permit offline calculation of physical lesion area, but it does not establish that an image-only model can predict physical area unless the input contains sufficient information.

## Scope

Respect the documented scope of Evidence:

- Case;
- image;
- examination;
- lesion;
- region;
- other registered scope.

Do not silently narrow or broaden scope.

Missing annotation is not a negative finding.

## Examples

Valid:

- documented lesion-shape categories may support `recognize`;
- a validated mask plus registered `mask_to_bbox` may support `localize`;
- a pathology-backed disease label may support `diagnosis`;
- explicitly linked longitudinal images may support `compare`.

Invalid:

- BI-RADS values `1–6` alone supporting `quantify`;
- age supporting image-based quantification;
- two images with only the same patient ID supporting `multi-view`;
- diagnosis supporting management without a recommendation reference;
- a mask supporting physical area when required spatial information is unavailable.

## Output

Follow the supplied AxisDecisionSet schema.

For every allowed value return:

- `value`;
- `supported`;
- `reason`;
- `evidence_basis`.

The reason should identify the decisive semantic evidence or blocker.

Evidence references must come from the supplied planning evidence.

Do not invent fields, relationships or annotations.

{{DATASET_SPECIFIC_RULES}}
