# MedBenchAgent

MedBenchAgent builds medical VLM benchmarks from a natural-language brief,
catalog-selected image-level `annotation.jsonl` data, and PDF medical references. The workflow screens
supported tasks, designs them with PDF RAG, and writes closed VQA, detection, finding
description, and report items into one mixed JSONL. Each run is initialized at
`benchmarks/runs/<run_id>/` before the first model call and writes an immutable,
unshuffled `benchmark.<dataset_id>.raw.jsonl`. The same directory contains provenance,
construction events/report, confirmed RAG designs, an `effective_config.json` provenance snapshot,
and a SHA-256 manifest. Per-task `sample_count` values are generation targets:
the files contain the accepted items produced before all targets are met or the
dataset is exhausted.

生成后会执行确定性 Audit：PNG 仅由 `image_id` 经同目录 `image_manifest.jsonl` 定位；audit 不读取源 DICOM、不以 `image_path` 定位 PNG，也不判断医学答案正确性。

## Dataset input

The workflow selects datasets from `datasets/dataset_catalog.json`; the repository-level
`dataset_catalog.json` remains a template. After
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

The working Blueprint keeps only `dataset.dataset_id`; it never embeds catalog metadata,
cache details, hashes, or local paths. `benchmark_manifest.json` carries the final dataset
record assembled from the catalog, profile, access manifest, and adjacent
`image_manifest.jsonl`, including annotation/image-manifest hashes and the fixed
`8bit_grayscale_png` image-format contract.

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

The retained tasks then receive a stable category-weighted allocation of the benchmark
target. The six task-category weights are normalized across categories that retain at
least one task, then distributed by largest remainder; tasks within each category split
that category's integer allocation evenly. The complete allocation is shown for
natural-language review; named task adjustments are fixed and the remaining count is
redistributed across unnamed tasks with the same weighting. Counts are written to the
working Blueprint only after confirmation. A zero-count task remains in the Blueprint
but skips RAG design and generation.

## Medical references

The workflow always scans the repository `materials/` directory after sample-count confirmation.
Store extractable PDFs in one of these top-level categories: `clinical_guidelines`,
`imaging_reporting_and_classification`, `imaging_acquisition_and_quality`,
`consensus_statements`, `evidence_reviews`, or `regulations_and_policy`. The user is shown
every categorized PDF and can select one or more files in natural language. DeepSeek maps that
response to exact relative paths, code validates the selection, and only the selected PDFs enter
RAG retrieval, indexing, and citations. PDFs outside these categories are configuration errors.

## Configuration

Configuration uses four ordered layers: immutable code fallbacks, committed
`config/default.toml`, one explicit local TOML file (`MEDBENCH_CONFIG` or
`load_settings(config_path=...)`), and non-empty environment-variable overrides.
Relative paths are resolved from the project root. `config/local.toml` is never discovered
implicitly; copy `config/local.toml.example` and set `MEDBENCH_CONFIG` to use it.

`seed`, `max_tasks_per_image`, and `question_temperature` in a LangGraph input override
the configured generation defaults for that one run. The run directory records the resulting
non-sensitive values, configuration-file paths and SHA-256 hashes, and applied override keys in
`effective_config.json`; the final manifest records its hash. API and LangSmith keys are never
written to this snapshot.

## Development

Copy `.env.example` to `.env`. The committed default configuration targets the local vLLM
deployment at `http://127.0.0.1:8005/v1` with model `deepseek-v4-flash`; put local model
directories in an explicit local TOML file.
`DEEPSEEK_API_KEY` may remain empty when vLLM authentication is disabled.
Model requests reuse one OpenAI-compatible client, default to a 120-second timeout,
and disable SDK retries. Set `DEEPSEEK_TIMEOUT_SECONDS` to override the timeout.
The reranker uses the Qwen3 causal-LM `yes`/`no` scoring procedure documented on
ModelScope, so an original ModelScope checkpoint works without Sentence Transformers
adapter metadata such as `modules.json` or `1_LogitScore/`. Embedding and reranking
models are loaded lazily once per model path in each worker process and then reused.
Set `RAG_MIN_RERANK_SCORE` to filter weak passages; the default is `0.5`.

To trace model calls in LangSmith, set `LANGSMITH_TRACING=true` with a valid
`LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`). The OpenAI-compatible DeepSeek
client is wrapped so RAG design, generation routing, and review calls appear as nested
LLM spans. The workflow still runs normally when tracing is disabled.

For Web VS Code, start the development server without browser launch and forward port
8123 through the IDE:

```bash
uv run langgraph dev --host 127.0.0.1 --port 8123 --no-browser --allow-blocking
```

The graph entry point is `medbenchagent.workflow.graph:compile_benchmark_graph`.
The deterministic root graph runs `SpecificationAgent`, `CapabilityAgent`,
`ClinicalDesignAgent`, `GenerationAgent`, and `AuditAgent` in that fixed order, with
only run initialization/finalization and artifact adapters between them. Agent-internal
confirmation, revision, RAG, and generation loops are visible with `xray=True`.

## Question design and benchmark outputs

RAG processes retained tasks one at a time and writes its result directly into each
working Blueprint task. The embedded fields are `question_templates`,
`generation_logic`, `medical_task_description`, `citations`, and, for letter tasks,
`options`. There is no separate Question Design artifact or duplicate generation spec.
Natural-language review can revise one or more task designs without changing other
Blueprint tasks.

The generator runs three independent seeded scans in a fixed order: letter VQA,
Grounding bbox, then direct-reference free text. Each stage begins from the database
start, projects only unfinished tasks' approved `supporting_fields`, and independently
accepts up to three tasks per image after a stable seed/image-ID shuffle. Letter VQA
normalizes one exact option; Grounding accepts only candidates with one valid annotated
box. Detection asks only for a generic lesion or abnormal region, while visual grounding
requires a named finding paired with that same box; the two tasks are mutually exclusive.
Free text accepts only one direct reference string. Each stage has its own prompt,
validator, review, and temporary progress file; failures do not consume the per-image
acceptance limit.

Accepted items are merged idempotently into a stage-specific progress JSONL using
deterministic item IDs. The generation subgraph checkpoints after every processed row,
so replay resumes the active stage safely. After all stages finish, their progress is
merged in stage order into raw JSONL and the temporary files are removed. If a dataset
is exhausted before every target is reached, generation still succeeds with the actual
items and records per-task shortfalls in graph state.

## Progress and trace display

The root graph exposes five expandable role stages. Specification and capability review
loops, RAG queue state, the private `GenerationRequest`, and generation record state remain
inside their owning Agent checkpoints, so LangSmith shows a compact root run that can be
expanded to inspect each role and its model calls or deterministic steps.
Clients can consume live progress with `stream_mode=["updates", "custom"]` and
`subgraphs=True`: RAG emits the completed/total task count and remaining queue length;
generation emits its cursor, completed and target task counts, rejection counts, and
exhaustion status after each processed record.

The raw file keeps original closed-task option order. Each item contains only `id`,
`task_id`, `image_id`, `image_path`, `question`, `options`, and `answer`; `answer` is a
choice letter, reference text, or an `xyxy` box. Run the balance command manually to
write a separate balanced copy; it recovers the correct semantic option from raw's
letter and option order, while detection and free-text items pass through unchanged.

## Construction records and Audit

`construction_events.jsonl` records only metadata. LLM calls use the fixed agent names
`SpecificationAgent`, `CapabilityAgent`, `ClinicalDesignAgent`, and `GenerationAgent`,
with their operation, logical `DEEPSEEK_MODEL` value, token usage, and latency; deterministic
Audit events use `AuditAgent`. Validated human confirmations and revisions are recorded without
their natural-language input. `construction_report.json` reduces these events to total and
per-agent LLM usage, generation/audit item counts, wall-clock time, and per-stage human counts.
Run lifecycle events use the `orchestrator` component and are not attributed to a sixth Agent.

Audit verifies the image-manifest mapping, streams PNG hashes, decodes 8-bit grayscale PNGs,
and preserves the existing contract and provenance checks. It rejects repeated item IDs and
canonical-identical items while retaining repeated question text; `audit_report.json` reports
normalized-question repetition separately in `duplicate_statistics`.

Balance an existing raw file without modifying it:

```powershell
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```
