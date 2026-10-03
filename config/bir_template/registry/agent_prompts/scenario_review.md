Review the supplied Scenario and Leaf summaries. Return a complete ScenarioReview covering every Leaf.
Use these five dimensions in order: (1) Scenario Alignment, including semantics and question
templates; (2) Evidence and Leakage; (3) Output Contract and Closed VQA options; (4) Benchmark
Validity; (5) Completeness and Consistency. Do not redesign Scenario fields or Leaf Tasks.
Accept only tasks that depend on the intended image or allowed input and can enter generation.

Check whether the question, benchmark-model input, and reference answer cover the same scope. Check target ambiguity, constant answers, duplicates across Scenarios, and whether every input is necessary.

The review package must include:
- Complete Leaf, questions, options, and output requirements.
- The input actually visible to the benchmark model.
- Resolver and reference evidence.
- A bounded summary of successful resolutions, failure reasons, and answer distribution.
- Deterministic validation results already completed by code.

Planning and review models should inspect the reference answer; only the benchmark VLM input must exclude reference evidence. Use `benchmark_model_input` for the latter concept.

The host supplies `describe_synthesis_allowed` and `describe_support_rule`. Follow them exactly. When synthesis is disabled, require a direct report-level or structured narrative reference and reject a narrative assembled from separate categorical values, text annotations, region annotations, metadata, or evidence counts.
