# MedBenchAgent

MedBenchAgent builds medical VLM benchmarks from a natural-language brief,
catalog-selected image-level `annotation.jsonl` data, and PDF medical references. The workflow screens
supported tasks, designs them with PDF RAG, and writes closed VQA, detection, finding
description, and report items into one mixed JSONL. It writes an immutable,
unshuffled `benchmark_items.raw.jsonl`. Per-task `sample_count` values are generation targets:
the files contain the accepted items produced before all targets are met or the
dataset is exhausted.

## Dataset input

The workflow selects datasets from the repository-level `dataset_catalog.json`. After
BenchmarkInfo confirmation, DeepSeek compares the confirmed benchmark request with every
catalog entry's `id`, `name`, `description`, and `available` status. It recommends one
available dataset for natural-language confirmation or reselection. Catalog paths are
never included in model prompts or user review. After confirmation, code resolves the
selected ID to its `annotation_path` and continues automatically.

Each catalog entry must contain exactly `id`, `name`, `description`, `annotation_path`,
and boolean `available` fields. Relative annotation paths are resolved from the project
root; absolute paths are also accepted. A dataset should be marked available only after
its path contains a valid normalized `annotation.jsonl`. Invalid catalog entries or
selected annotation files are treated as configuration errors rather than falling back
to manual path input.

The downstream workflow accepts only an `annotation.jsonl` file. Each line represents one image
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

The retained tasks then receive a stable, even allocation of the benchmark target.
The complete allocation is shown for natural-language review; named task adjustments
are fixed and the remaining count is redistributed across unnamed tasks. Counts are
written to the working Blueprint only after confirmation. A zero-count task remains in
the Blueprint but skips RAG design and generation.

## Medical references

The workflow always scans the repository `materials/` directory after sample-count confirmation.
Store extractable PDFs in one of these top-level categories: `clinical_guidelines`,
`imaging_reporting_and_classification`, `imaging_acquisition_and_quality`,
`consensus_statements`, `evidence_reviews`, or `regulations_and_policy`. The user is shown
every categorized PDF and can select one or more files in natural language. DeepSeek maps that
response to exact relative paths, code validates the selection, and only the selected PDFs enter
RAG retrieval, indexing, and citations. PDFs outside these categories are configuration errors.

## Development

Copy `.env.example` to `.env`. The default configuration targets the local vLLM
deployment at `http://127.0.0.1:8005/v1` with model `deepseek-v4-flash`, plus the
Qwen3 embedding and reranker models under `/public/home/rxa-jj-wzk/models/Qwen-RAG`.
`DEEPSEEK_API_KEY` may remain empty when vLLM authentication is disabled.
Model requests reuse one OpenAI-compatible client, default to a 120-second timeout,
and disable SDK retries. Set `DEEPSEEK_TIMEOUT_SECONDS` to override the timeout.
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

The generator traverses source rows in a deterministic seeded order and projects only
each unfinished task's approved `supporting_fields`, preserving array source indices.
For each image, code removes completed or unsupported tasks, deterministically shuffles
the rest from the seed and image ID, and attempts them one at a time until three are
accepted. Letter VQA generation receives only its task context and projected annotation;
it returns one question and one exact option, then receives an independent review.
Bbox items retain their deterministic coordinate check, while free-text items retain one
model review. Failed attempts do not consume the per-image acceptance limit.

Accepted items are merged idempotently into `benchmark_items.progress.jsonl` using
deterministic item IDs. The generation subgraph checkpoints after every processed row,
so replay after a checkpoint/file mismatch safely deduplicates accepted items. If the
dataset is exhausted before every target is reached, generation still succeeds with
the actual items and records the per-task shortfall in graph state.

The raw file keeps original closed-task option order. Each item contains only `id`,
`task_id`, `image_id`, `image_path`, `question`, `options`, and `answer`; `answer` is a
choice letter, reference text, or an `xyxy` box. Run the balance command manually to
write a separate balanced copy; it recovers the correct semantic option from raw's
letter and option order, while detection and free-text items pass through unchanged.

Balance an existing raw file without modifying it:

```powershell
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```
