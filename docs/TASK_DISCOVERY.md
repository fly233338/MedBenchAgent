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
  -> Global Allocation
  -> Per-dataset Generation
  -> Global Finalization and Benchmark Items
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

Screening `keep` means that the Scenario is worth exploring; it does not
promise that a valid Leaf will be found. A Leaf-design Scenario may be
explicitly excluded after its bound context has been read when no valid Leaf
exists, including after review discards every saved Leaf. The exclusion
reason and evidence are persisted. A data or model failure never performs
this exclusion automatically. A completed plan may therefore contain zero
executable tasks when every Scenario was excluded.

Scenario counts have separate meanings: `feasible_case_count` counts Cases
that satisfy deterministic structural Scenario constraints; `eligible_count`
counts Cases resolved successfully by one Leaf Resolver; Scenario Review
decides whether the task meaning, wording, reference contract and templates
are clinically valid.

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
Every template must remain valid for every eligible Case, including normal,
negative and other supported boundary Cases; wording must not presuppose a
lesion, unique object or narrower annotation scope.

Question templates are complete strings; generation does not interpolate
placeholders. A Leaf's `input` and `resolver.inputs` must match exactly.
Resolver roles are unique and are resolved in declaration order. Multiple
asset roles cannot resolve to the same Evidence ID. The generated item keeps
ordered role bindings, flattened input Evidence IDs/assets, options and
reference Evidence IDs so downstream consumers do not need another semantic
decision.

Leaf duplicate detection uses a Scenario-independent semantic fingerprint
covering ordered inputs, target, reference rule, Resolver, output contract and
answer vocabulary. Scenario names or axis combinations alone do not create
new task coverage; assessment and diagnosis remain distinct when their target
or reference meaning differs.

For Closed VQA labels, direct tasks use the registered observed vocabulary.
Derived comparison or computation tasks use the encoded values produced by
the Resolver. The same encoding and option-domain check is applied when a
Leaf is saved, when items are generated and when items are audited.

## Operation and target semantics

Operation 表示“要求模型完成什么任务”，`clinical_target` 表示“针对什么对象或属性”，输出格式再单独规定。

`recognize` covers presence, absence, identity, type, or categorical attributes of clinically relevant entities or observations, including anatomical structures, tissues, cells, findings, devices, evidence-supported assessment or diagnostic categories, visually assessable acquisition characteristics, and image quality.

`localize` grounds a specified anatomical structure, tissue, cell, finding, or device in an explicit spatial or sequence location and returns a supported bounding box or slice/frame index.

`quantify` determines a count, magnitude, extent, proportion, or other numeric quantity with an explicit measurement or counting scope, unit where applicable, and reference rule.

`compare` determines similarities, differences, ordering, or change between explicitly linked and clinically comparable entities or observations, including the comparison dimension and direction.

`align` establishes correspondence or concordance between explicitly related evidence. Multiple inputs alone do not make a task an alignment task.

`describe` produces a coherent paragraph or structured report synthesizing multiple evidence-supported clinical observations and their relationships. An isolated label or verbose restatement of one categorical fact does not qualify.

Task Planning must distinguish `benchmark_model_input` from reference evidence. Planning and review models may inspect the Resolver and reference contract; reference evidence must not be placed in the benchmark model input. Prepared mask features are immutable derived references, not model inputs.

Clinical target selection follows the requested prediction: `assessment` is a
standardized grade, score or risk category such as BI-RADS; `diagnosis` is a
disease or pathology class with an explicit reference basis; `finding` is an
observed sign or pattern; and `attribute` is a property such as shape,
margin, composition or echogenicity. Assessment and diagnosis may coexist.
A fixed organ name alone is not an anatomy task.

Closed-label tasks may use the deterministic `map_label` Resolver step when
an explicit evidence-supported mapping groups raw encoded labels into a
single answer domain. Unmapped, missing, unknown, not-applicable and explicit
negative values remain distinct unless the mapping explicitly states
otherwise. The same Resolver output is used for planning capacity, generation
and audit.

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
to the selected dataset revision and access digests. Ordered comparison and index
tasks additionally require a frozen presentation or sampling manifest.

Scenario Review processes one pending Scenario at a time by its `scenario_id`.
The queue head is the only valid ID, which makes resumed operation receipts
specific to the Scenario being reviewed. Review may atomically discard invalid
Leaf Tasks with a required reason. The discarded IDs are removed from the
Scenario and canonical plan; an approved Scenario must retain at least one
valid Leaf Task.

The canonical `pending_scenario_ids` projection follows the active workflow
queue. During `scenario_review` it contains the Scenarios still awaiting
candidate review; after that queue is exhausted it contains the accepted
Scenarios still awaiting Leaf design and review. The persisted
`scenario_review_pending_ids` field remains the authoritative review queue.


## Validated Clinical Targets and Rules

Scenario Screening is an evidence gate. It returns atomic `validated_clinical_targets` with an
ID, target type, subject, scope, reference kind, and reference fields. `keep` requires at least
one target; it does not ask the model to brainstorm Leaf Tasks. Leaf design must select a target
ID from the current Scenario and cannot create a new clinical target.

The four core planning Prompts are complete Markdown files containing common
constraints and validated examples. Each has one final
`{{DATASET_SPECIFIC_RULES}}` placeholder. Optional disease or dataset extensions
are frozen under `registry/planning_extensions/` and selected by the Supervisor
through the human Dataset Selection-style flow. Selected extension text is
filtered by stage and appended at the end of the core Prompt. Extensions may
add or tighten constraints but cannot relax evidence grounding, Resolver
validation, or the output contract. Deterministic code still owns all structural
and Resolver validation.
