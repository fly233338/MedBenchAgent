# Task Discovery v2

Task Discovery uses annotation.v5 Case/Evidence/Asset data as its only source of truth.

```text
Dataset Profile → Axis Analysis → Task Nodes → Node Evidence Context
→ Task Leaves → Resolver → Capacity/Audit → Benchmark Items
```

The fixed axes are `input_structure`, `operation`, and `clinical_target`. Dataset analysis may select registered values but cannot redefine them. Code constructs Task Nodes through a constrained cross-product and assigns the stable Capability Category from the Operation.

Task Leaves contain a canonical question, input Evidence roles, target, reference rule, OutputSpec and TaskParadigm. A Leaf is executable only after its typed ResolverSpec compiles and resolves at least one Case. The Resolver is shared by capacity counting, generation and audit; it never calls an LLM or external medical source.

Generation produces `benchmark-manifest.v2` items with `case_id`, input Evidence/Assets, OutputSpec, answer and reference Evidence IDs. Reference Evidence is audited separately and is never part of model input.

Evaluation metrics are configured after benchmark construction through `evaluation-config.v2`. Task Planning, Generation and Audit do not select or require metrics.
