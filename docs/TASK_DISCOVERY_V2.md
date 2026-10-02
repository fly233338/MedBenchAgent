# Task Planning

Task Planning uses an immutable DatasetAccessReceipt and staged
annotation.v5 evidence artifacts. The model receives only the evidence slice
needed for its current decision.

~~~text
DatasetSelection
  -> DatasetAccessReceipt
  -> DatasetOverview + EvidenceCatalog + RelationSummary + CaseSamples
  -> Axis Value decisions
  -> deterministic Task Scenarios
  -> current Scenario evidence
  -> Leaf Tasks + Task Paradigm
  -> Scenario review
  -> Benchmark Items
~~~

## Six fixed levels

1. **Task Axis** — human-defined registry; the model selects values. The
   current axes are input_structure, operation, and clinical_target.
2. **Axis Value** — the model marks every registered value as supported or
   unsupported and cites bounded evidence.
3. **Task Scenario** — code forms the cross-product of supported values, checks
   structural feasibility and records exclusions.
4. **Leaf Task** — the model proposes a concrete target such as
   pleural-effusion presence; code validates selectors, Resolver and capacity.
5. **Task Paradigm** — selected at Leaf Task time and checked against OutputSpec.
6. **Benchmark Item** — code instantiates a Leaf Task for one concrete Case.

A plan is resumable and stores only stage references, decisions, scenarios,
leaf tasks, reviews and revisions. Large dataset facts are kept in separate
evidence artifacts and paged through read tools.

## Capability Category

Capability Category is a horizontal reporting label, not a hierarchy level.
Code maps Operation to exactly one label:

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
