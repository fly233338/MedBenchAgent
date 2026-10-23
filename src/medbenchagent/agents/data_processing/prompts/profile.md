# Operation: profile
Produce an evidence-grounded DatasetProfile that enables scenario discovery without requiring full
normalization. Establish actual source formats, available fields and assets, annotation meaning,
case granularity, and supported relationships. Record unresolved joins or temporal relationships
as unknown instead of inferring them from filenames.

Delegate a bounded inspection program to Coding. Request reproducible observations and source
references, representative samples, inspected scope, and explicit sampled/complete/unknown coverage.
Statistics must identify what was measured; never present sample statistics as population totals.
Distinguish data suitable as evaluated-model input from labels or other reference-only information.

Inspect the candidate Profile's observations against its inspected scope and evidence. Record
limitations that will matter to task planning. Publish an accepted Profile with submit_data_artifacts.
Do not create Scenarios, Leaf tasks, or backend-specific Cases in this operation. If evidence is
insufficient, preserve useful observations and report the precise additional inspection or user fact
needed; do not fabricate a complete Profile.
