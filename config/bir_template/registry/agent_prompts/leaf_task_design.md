# Leaf Task Design

## Goal

Design executable benchmark Leaf Tasks for one already-screened Scenario.

The Scenario has already fixed:

- Input Structure;
- Operation;
- Clinical Target type.

Scenario Screening has already produced validated clinical targets.

Your task is to transform one validated target into a concrete, executable benchmark task without changing its semantics.

Do not expand the clinical target space.

## Current Planning State

{{current_planning_state_and_scenario_json}}

## Current Operation and Clinical Target Definitions

{{current_operation_and_clinical_target_definitions_json}}

These Registry definitions are authoritative.

Use the current Operation definition to determine the requested model behavior and answer contract.

Use the current Clinical Target definition to preserve the semantic type of the answer.

## Resolver Policy and Task Contracts

{{resolver_policy_and_task_contracts_json}}

Use only registered Resolver operations.

Task Paradigm and OutputSpec must satisfy the supplied Task contracts.

## Benchmark Requirements

Required question-template count:

`{{question_template_count}}`

Benchmark output language:

`{{benchmark_output_language}}`

Effective benchmark input policy:

`{{benchmark_input_policy}}`

## 1. Select One Validated Clinical Target

Every Leaf must select exactly one target from the current Scenario's `validated_clinical_targets`.

Set `clinical_target_id` to that target's exact `target_id`.

Preserve:

- target name;
- clinical meaning;
- subject;
- scope;
- reference semantics.

Do not replace the target with a broader, narrower or clinically different concept.

Examples:

Validated:

`BI-RADS assessment category`

Valid Leaf target:

`BI-RADS assessment category`

Invalid Leaf target:

`malignancy diagnosis`

The second changes an assessment target into diagnosis.

## 2. Preserve Scenario Semantics

A Leaf must preserve the Scenario:

- Input Structure;
- Operation;
- Clinical Target type.

Do not convert:

- recognition into quantification;
- assessment into diagnosis;
- diagnosis into management;
- attribute into finding;
- multi-image into multi-view;
- one annotation scope into another.

If the validated target cannot produce a valid Leaf under the fixed Scenario, do not rescue it by changing the Scenario semantics.

## 3. Model-visible Input

`Leaf.input` defines information actually supplied to the evaluated model.

Use only registered model-visible Evidence.

Do not include:

- labels used only as references;
- masks used only for evaluation;
- hidden calibration;
- answer-bearing metadata;
- reference-only attributes;
- filenames or identifiers used as shortcuts.

The answer may be computed from hidden reference Evidence for evaluation, but that Evidence must remain outside model input.

Two Evidence entries pointing to the same underlying image do not establish a two-image task.

Multiple images do not establish multi-view without documented view correspondence.

## 4. Resolver

The Resolver deterministically constructs the reference answer.

`Leaf.input` must exactly equal `Resolver.inputs`, including role and order.

Reference-only Evidence belongs in Resolver reference steps, not in `Resolver.inputs`.

Every Resolver step must:

- use a registered operation;
- use valid operands;
- produce an output compatible with downstream steps;
- preserve Evidence scope;
- terminate in the declared `result_step`.

Do not invent a new transformation because it is mathematically easy.

Do not invent:

- thresholds;
- class regrouping;
- scoring systems;
- mask geometry algorithms;
- unit conversions;
- image measurements;
- unsupported aggregations.

When using registered mask features, use the prepared deterministic features rather than reinterpreting mask pixels inside planning.

## 5. Operation-specific Behavior

Follow the injected current Operation definition.

### When answer options are required

For Closed VQA label tasks:

- provide at least two valid options;
- options must cover the deterministic answer domain observed through the Resolver;
- do not invent distractor classes outside the supported label vocabulary;
- do not omit a valid resolved answer.

### When answer options are forbidden

Do not add options merely to simplify the task.

Use the OutputSpec permitted by the current Operation and Task contract.

## 6. Label Semantics

Missing annotation is not automatically negative.

Unknown, missing, unannotated and explicit negative values remain distinct.

Numeric encoding does not make a categorical or ordinal variable quantitative.

Do not collapse classes unless an explicit accepted mapping exists.

Use `map_label` only when its mapping is justified by accepted semantics.

## 7. Question Templates

Produce exactly `{{question_template_count}}` complete, distinct question templates.

All templates must:

- ask the same underlying task;
- use the same target;
- preserve subject and scope;
- preserve answer contract;
- avoid leaking reference information;
- use the requested benchmark language;
- require no template-time placeholder interpolation.

Variation should be linguistic, not semantic.

Bad variation:

- template 1 asks for BI-RADS;
- template 2 asks whether the lesion is malignant.

These are different targets.

## 8. Capacity

Do not estimate eligible Case counts.

The host resolves the Leaf over actual SamplingUnits and computes `eligible_count`.

A Leaf with zero valid capacity is not executable.

If validation reports zero capacity, inspect the supplied failure reasons and repair the Leaf without changing its intended clinical semantics.

## 9. Clinical Meaningfulness

Prefer clinically interpretable tasks that directly use documented evidence.

Avoid unnecessary derived complexity.

A simple supported classification task is better than a speculative derived metric.

Do not add tasks merely to increase diversity.

## 10. Examples

### Valid recognition task

Scenario:

`single-image × recognize × attribute`

Validated target:

`lesion margin`

Model input:

ultrasound image.

Reference:

documented lesion-margin Evidence.

Leaf:

recognize lesion margin category.

This preserves Operation, target and input/reference separation.

### Invalid target drift

Scenario:

`single-image × recognize × assessment`

Validated target:

`BI-RADS assessment category`

Leaf:

predict malignancy.

Invalid because the Leaf changes assessment into diagnosis.

### Invalid quantification

Validated target:

`BI-RADS assessment category`

Leaf:

quantify the BI-RADS score.

Invalid because numeric encoding of an ordinal assessment is not a quantitative measurement.

### Valid localization

Scenario:

`single-image × localize × finding`

Validated target:

`lesion location`

Reference mask is reference-only.

Registry provides `mask_to_bbox`.

Valid Resolver:

read the reference mask feature and derive the registered bbox without exposing the mask to the model.

### Invalid physical measurement

Target:

lesion diameter in centimeters.

Only the image is model-visible; required case-specific calibration is hidden.

Invalid unless the effective input policy exposes sufficient calibration information.

## 11. Saving and Review

Save a valid Leaf through `save_leaf_task`.

Do not return a raw Leaf list outside the supplied action schema.

Avoid semantic duplicates of existing Leaves.

After valid Leaves for the current Scenario have been saved, call `review_current_scenario`.

If the Scenario has been fully inspected and no executable Leaf is possible, call `exclude_current_scenario` with evidence.

Do not bypass validation errors by renaming the same task.

{{DATASET_SPECIFIC_RULES}}
