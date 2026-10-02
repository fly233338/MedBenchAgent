# Task Discovery

Task Planning uses annotation.v5 Cases, Evidence and Assets as its only data
source. The registered `config/blueprint.json` Blueprint supplies capability
dimensions after the Task Tree has been reviewed.

```text
Dataset Profile + representative Case
  → Dataset Facts
  → analyze_axis(input_structure)
  → analyze_axis(operation)
  → analyze_axis(clinical_target)
  → deterministic Task Nodes
  → ordered current Node
  → evidence-backed Task Leaves
  → Node review
  → complete Task Tree
  → Blueprint capability assignments
  → Resolver / capacity / generation / audit
```

The Task Planning Agent keeps one resumable `task_space_plan.json` draft. The
draft contains the Axis decisions, Nodes, Leaves, Node reviews, current Node
pointer and capability assignments. Derived axis/node/leaf files are views for
inspection; they are not independent sources of truth.

The three axes are fixed. The model calls `analyze_axis` exactly once for each
axis in the order shown above. Code validates registry coverage, evidence
references, deterministic Node IDs and structural feasibility. Excluded
combinations remain recorded but do not require Leaves.

Each executable Node must receive at least one Leaf. A Leaf has one canonical
`question_template`, explicit input selectors, a typed Resolver, target,
OutputSpec and TaskParadigm. Resolver resolution, input/reference separation
and positive Case capacity are deterministic checks. After all Leaves for a
Node are saved, the model submits one review covering every Leaf. A rejected
review returns the current Node to design; an approved review advances the
ordered pointer.

The final plan cannot publish while any executable Node is pending, lacks a
positive-capacity Leaf, lacks a complete review or lacks a registered Blueprint
capability assignment. Capability assignments may contain multiple dimensions.
TaskParadigm and OutputSpec remain separate reporting dimensions.

The Resolver is shared by capacity counting, generation and audit. Reference
Evidence is used to reconstruct answers and is never included in model input.
