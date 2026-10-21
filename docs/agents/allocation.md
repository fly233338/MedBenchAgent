# Allocation

**Input:** one complete dataset Task Plan, requested sample budget and optional explicit allocation preferences.

**Output:** a canonical BIR allocation with leaf quotas, capacities, requested/allocated counts and shortfall.

The entry point lives in `planning.allocation`. Counts and capacities are indexed directly by Leaf ID; it writes only BIR `allocation.json`. Allocation has no separate Agent module, private session or dataset aggregation layer.

**State and execution:** deterministic capacity-aware allocation. Only interpreting explicit preferences uses a model; it receives task names, targets/capacities and quoted preference IDs. Non-default weights require those IDs and must be finite/nonnegative.

**Feedback:** zero capacity and shortfall are factual results. They do not authorize fabricated examples, new priorities or cross-dataset redistribution.

**Recovery:** receipt dependencies bind requirements, dataset and plan. A plan change invalidates the allocation.

**Acceptance:** quotas never exceed eligible SamplingUnits, no cross-dataset budget, exact preference coverage and explicit shortfall. Generation consumes the BIR allocation rather than a UI/checkpoint copy.
