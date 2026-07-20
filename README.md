# MedBenchAgent

MedBenchAgent builds medical VLM benchmarks from a natural-language brief,
image-level `annotation.jsonl` data, and PDF medical references. The workflow screens
supported tasks, designs them with PDF RAG, and writes closed VQA, detection, finding
description, and report items into one mixed JSONL. It first writes an immutable,
unshuffled `benchmark_items.raw.jsonl`, then by default writes a separately finalized
`benchmark_items.jsonl` copy. Both files contain exactly the requested `sample_count`.

## Dataset input

The workflow accepts only an `annotation.jsonl` file. Each line represents one image
and must contain `image_id`, `image_path`, `patient_id`, `case_id`, `study_id`, and an
object-valued `annotation`. `image_path` is stored as opaque metadata: dataset
exploration does not resolve it or check whether an image file exists. Benchmark items
are identified by `image_id`; mapping IDs to image files is the responsibility of an
external module. The versioned input contract is at
`src/medbenchagent/schemas/annotation.v1.schema.json`.

Validated annotation text is cached by JSONL content hash and cache-format version
under `benchmarks/dataset_cache/` as a DuckDB database with one `images` row per JSONL
line and a deterministic `task-capability-profile.v1`. The profile describes annotation
structures and every scalar leaf field, but exposes raw values only for low-cardinality
categorical fields; identifiers, DICOM metadata, numeric values, and coordinates are
not included as examples. CSV and XLSX inputs are not supported.

## Planning review

The workflow computes the complete Task Plan in one model call, then presents it for
review. Respond in natural language to confirm it or describe one task to revise;
DeepSeek maps every response to a validated confirm/revise action. BenchmarkInfo review
uses the same natural-language interpretation. The working Blueprint is updated only
after confirmation: dropped tasks are removed and retained tasks receive their approved
`supporting_fields`. The resulting `task_plan_report.json` retains decisions for both
retained and dropped tasks.

## Development

Copy `.env.example` to `.env`. The default configuration targets the local vLLM
deployment at `http://127.0.0.1:8005/v1` with model `deepseek-v4-flash`, plus the
Qwen3 embedding and reranker models under `/public/home/rxa-jj-wzk/models/Qwen-RAG`.
`DEEPSEEK_API_KEY` may remain empty when vLLM authentication is disabled.
The reranker uses the Qwen3 causal-LM `yes`/`no` scoring procedure documented on
ModelScope, so an original ModelScope checkpoint works without Sentence Transformers
adapter metadata such as `modules.json` or `1_LogitScore/`. Embedding and reranking
models are loaded lazily once per model path in each worker process and then reused.
Set `RAG_MIN_RERANK_SCORE` to filter weak passages; the default is `0.5`.

For Web VS Code, start the development server without browser launch and forward port
8123 through the IDE:

```bash
uv run langgraph dev --host 127.0.0.1 --port 8123 --no-browser --allow-blocking
```

The graph entry point is `medbenchagent.workflow.graph:compile_benchmark_graph`.

## Question design and benchmark outputs

RAG processes retained tasks one at a time and writes its result directly into each
working Blueprint task. The embedded fields are `question_templates`,
`generation_logic`, `medical_task_description`, `citations`, and, for letter tasks,
`options`. There is no separate Question Design artifact or duplicate generation spec.
Natural-language review can revise one or more task designs without changing other
Blueprint tasks.

For each task/image candidate, the generator projects only the task's approved
`supporting_fields` from the annotation while preserving array source indices. DeepSeek
uses that projection and the embedded task design to return one question, normalized
answer, and source references. Code validates the response shape and an independent
model reviews annotation fidelity, medical normalization, bbox alignment, and answer
leakage before the item is accepted. Each task/image produces at most one item.

The raw file keeps task/source-row order and original closed-task option order. Every
item records `task_type` and `response_format`; `answer` is a choice letter, reference
text, or an `xyxy` box. The final copy balances correct option positions inside each
closed task while detection and free-text items pass through unchanged.

Balance an existing raw file without modifying it:

```powershell
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```
