# Task Planning

Task Planning uses an immutable DatasetAccessReceipt and staged
annotation evidence artifacts. The model receives only the evidence slice
needed for its current decision.

~~~text
DatasetSelection
  -> DatasetAccessReceipt
  -> DatasetOverview + EvidenceCatalog + RelationSummary + CaseSamples
  -> Axis Value decisions
  -> deterministic Task Scenarios
  -> current Scenario evidence
  -> Leaf Tasks + Task Paradigm + question templates
  -> Scenario review
  -> Benchmark Items
~~~

## Registry-backed levels

1. **Task Axis** — the experiment-frozen BIR Registry defines order and values;
   the model selects values. The default axes are input_structure, operation,
   and clinical_target.
2. **Axis Value** — the model marks every registered value as supported or
   unsupported and cites bounded evidence.
3. **Task Scenario** — code forms the cross-product of supported values, checks
   structural feasibility and records exclusions.
4. **Leaf Task** — the model proposes a concrete target such as
   pleural-effusion presence; code validates selectors, Resolver and capacity.
5. **Task Paradigm** — selected at Leaf Task time and checked against OutputSpec.
6. **Benchmark Item** — code instantiates a Leaf Task for one concrete Case and selects one
   question template with the frozen generation seed.

A plan is resumable and stores only stage references, decisions, scenarios,
leaf tasks, reviews and revisions. Large dataset facts are kept in separate
evidence artifacts and paged through read tools.

The Evidence Catalog and annotation mapping are complete dataset facts, not a
task candidate list. Task Planning must distinguish image-grounded evidence
from patient/clinical context, device or administrative metadata, and
calibration-only spacing. Medical relevance alone does not make a field a
valid image-only task target.

Value evidence may also expose deterministic `observed_values` entries with
normalized values and counts. These entries are the only source for Closed VQA
option vocabularies. VQA label leaves store the complete finite vocabulary in
`options`; Report Generation and Visual Grounding leaves do not.

Each new Leaf Task stores the configured number of distinct
`question_templates` (between three and five). Review checks their semantic
equivalence. Generation uses a seeded shuffle cycle and records the selected
`template_index` on each Benchmark Item so output diversity is reproducible.

## Operation and target semantics

Operation 表示“要求模型完成什么任务”，`clinical_target` 表示“针对什么对象或属性”，输出格式再单独规定。

`recognize` covers presence, absence, identity, type, or categorical attributes of clinically relevant entities or observations, including anatomical structures, tissues, cells, findings, devices, evidence-supported assessment or diagnostic categories, visually assessable acquisition characteristics, and image quality.

`localize` grounds a specified anatomical structure, tissue, cell, finding, or device in an explicit spatial or sequence location and returns a supported bounding box or slice/frame index.

`quantify` determines a count, magnitude, extent, proportion, or other numeric quantity with an explicit measurement or counting scope, unit where applicable, and reference rule.

`compare` determines similarities, differences, ordering, or change between explicitly linked and clinically comparable entities or observations, including the comparison dimension and direction.

`align` establishes correspondence or concordance between explicitly related evidence. Multiple inputs alone do not make a task an alignment task.

`describe` produces a coherent paragraph or structured report synthesizing multiple evidence-supported clinical observations and their relationships. An isolated label or verbose restatement of one categorical fact does not qualify.

Task Planning must distinguish `benchmark_model_input` from reference evidence. Planning and review models may inspect the Resolver and reference contract; reference evidence must not be placed in the benchmark model input.

## Capability Category

Capability Category is a horizontal reporting label, not a hierarchy level.
The frozen BIR Registry maps each Operation to exactly one label:

| Operation | Capability Category |
| --- | --- |
| recognize | Recognition |
| localize | Localization |
| quantify | Quantification |
| compare | Comparison |
| align | Integration |
| describe | Generation |

The six categories cover recognition, localization, quantification, comparison,
multi-image or image-text integration, and description/report generation. The
category is inherited by Scenario, Leaf Task and Benchmark Item.

## Contracts and recovery

Axis tool arguments use nested Pydantic models, including AxisValueDecision
and evidence basis objects. Leaf and review arguments are typed as well. A
tool failure that did not advance the persisted plan remains blocked;
Supervisor does not relabel it as progress or automatically repeat it.

Reference Evidence is used by the deterministic Resolver to reconstruct an
answer and is never included in model input. Every evidence artifact is bound
to the selected dataset revision and access digests.

Scenario Review processes one pending Scenario at a time by its `scenario_id`.
The queue head is the only valid ID, which makes resumed operation receipts
specific to the Scenario being reviewed. Review may atomically discard invalid
Leaf Tasks with a required reason. The discarded IDs are removed from the
Scenario and canonical plan; an approved Scenario must retain at least one
valid Leaf Task.

