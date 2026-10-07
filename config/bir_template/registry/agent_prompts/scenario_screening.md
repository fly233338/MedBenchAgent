# Scenario Screening

## Goal

Review one candidate Task Scenario and determine whether the dataset supports at least one concrete clinical target under that exact Scenario.

The Scenario is already fixed as:

`Input Structure × Operation × Clinical Target Type`

Do not change any of these dimensions to make the Scenario feasible.

Scenario Screening identifies valid targets. It does not design Leaf Tasks or question templates.

## Authoritative Context

Use the supplied:

- Scenario;
- Operation definition;
- Clinical Target definition;
- benchmark brief;
- Evidence Catalog;
- field semantics;
- relations;
- derived capabilities;
- permitted reference fields;
- annotation semantics.

The supplied Operation and Clinical Target definitions are authoritative.

## Valid Clinical Target

A validated clinical target must be:

### Atomic

It represents one clearly defined clinical variable.

Good:

- lesion margin;
- BI-RADS assessment category;
- presence of pleural effusion;
- lesion bounding box.

Too broad:

- lesion characteristics;
- imaging findings;
- clinical status;
- morphology in general.

### Axis-consistent

Its meaning matches the Scenario's fixed Clinical Target type.

Do not use evidence for one target type to justify another.

### Subject-bound

The entity or Case to which the target applies is clear.

### Scope-bound

Its annotation scope is supported by the dataset.

Do not move a Case-level label onto a lesion or a lesion-level label onto an unrelated image.

### Reference-grounded

The exact reference Evidence must be identifiable.

For a direct target, identify the actual reference field.

For a derived target, identify the source Evidence and required registered operations.

### Input-compatible

The permitted model-visible input must contain enough information for this to be a legitimate prediction problem.

Ground-truth availability alone is insufficient.

### Executable

Every required reference transformation must be registered.

Do not keep a target because a human could theoretically compute the answer.

## Operation Alignment

Apply the current Operation definition supplied in the request.

Do not reinterpret the Operation.

Examples:

- `recognize × assessment` may support BI-RADS category recognition;
- `quantify × assessment` does not become valid merely because BI-RADS is stored numerically;
- `localize × finding` requires a supported spatial reference;
- `compare` requires explicitly comparable entities and a comparison dimension.

## Quantitative Targets

For `quantify`, require:

- named quantity;
- subject;
- scope;
- unit or documented dimensionless interpretation;
- valid deterministic reference.

Do not create quantitative targets from:

- IDs;
- demographic values;
- numeric category codes;
- ordinal assessment categories;
- acquisition parameters;
- unexplained numeric fields.

## Categorical Mapping

Do not invent:

- thresholds;
- benign/suspicious regrouping;
- merged classes;
- binary labels;
- score groupings;
- arithmetic scoring rules.

A mapping is valid only when explicitly documented in accepted semantics or supported by a registered mapping operation.

## Diagnosis and Management

Diagnosis requires diagnostic reference evidence.

Do not turn imaging descriptors or assessment categories into diagnosis without such evidence.

Management requires explicit recommendation, procedure, follow-up, treatment or next-step evidence.

Do not infer management from diagnosis, assessment or general medical knowledge.

## Input Structure

Judge the actual visual input selected for the task.

Multiple images require:

- documented Case membership;
- task-relevant relationship;
- target/reference scope that applies to those images.

Multi-view additionally requires explicit view identity and correspondence.

Different time points are not automatically multi-view.

Reference-only masks do not count as model-visible additional images.

## Decision

Return `keep` only when at least one valid clinical target exists.

Return `exclude` when:

- no atomic target exists;
- only unsupported inference makes the target possible;
- required input is hidden;
- required reference derivation is unavailable;
- annotation scope is incompatible;
- keeping the target would require changing the fixed Scenario.

When evidence is genuinely insufficient, exclude rather than fill gaps with medical knowledge.

## Output

Follow the supplied ScenarioScreeningResult schema.

For `keep`, provide one or more `validated_clinical_targets`.

Each target must include:

- stable `target_id`;
- concise clinical `name`;
- `target_type` matching the Scenario;
- explicit `subject`;
- supported `scope`;
- `reference_kind`;
- actual `reference_fields`.

`reference_fields` must use exact Evidence names permitted by the supplied Evidence Catalog.

A target name is a clinical variable, not a question.

Good:

`BI-RADS assessment category`

Bad:

`Classify BI-RADS from this ultrasound`

For `exclude`, do not create placeholder targets.

Do not propose Leaf Tasks, answer choices or question templates here.

{{DATASET_SPECIFIC_RULES}}
