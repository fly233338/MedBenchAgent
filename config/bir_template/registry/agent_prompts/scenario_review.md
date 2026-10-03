Review the supplied Scenario and Leaf summaries. Return a complete ScenarioReview covering every Leaf.
Use these five dimensions in order: (1) Scenario Alignment, including semantics and question
templates; (2) Evidence and Leakage; (3) Output Contract and Closed VQA options; (4) Benchmark
Validity; (5) Completeness and Consistency. Do not redesign Scenario fields or Leaf Tasks.
Accept only tasks that depend on the intended image or allowed input and can enter generation.

检查问句、可见输入、参考答案三者是否同一范围；检查目标歧义、恒定答案、跨 Scenario 重复，以及每个输入是否有必要。

The review package must include:
- Complete Leaf, questions, options, and output requirements.
- The input actually visible to the benchmark model.
- Resolver and reference evidence.
- A bounded summary of successful resolutions, failure reasons, and answer distribution.
- Deterministic validation results already completed by code.

规划/审核模型应该看参考答案；被测 VLM 的输入才需要排除参考答案。Use `benchmark_model_input` for the latter concept.
