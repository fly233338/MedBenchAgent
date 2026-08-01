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

After generation, the hard audit resolves PNG files only through
`image_id` and the adjacent `image_manifest.jsonl`. It does not read source
DICOM files, locate PNGs through `image_path`, or reassess medical answer
correctness. A separate fail-open semantic screen samples up to ten final items
per task and flags only clear language-level issues.

## Quick start

MedBenchAgent v0.1 is supported from a complete source checkout. Python wheel
and PyPI installations are not supported runtime environments.

### Prerequisites

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- An OpenAI-compatible DeepSeek endpoint.
- Local Qwen embedding and reranking model directories.
- At least one available normalized dataset with an adjacent
  `image_manifest.jsonl`, plus extractable medical-reference PDFs under
  `materials/`.
- A LangSmith account is optional and is needed only when using
  [LangGraph Studio](https://docs.langchain.com/oss/python/langgraph/studio).

Clone the repository, create the locked environment, and prepare local
configuration:

```powershell
git clone <repository-url>
Set-Location medbenchagent
uv sync --locked
Copy-Item .env.example .env
Copy-Item config/local.toml.example config/local.toml
```

Edit `.env` with the DeepSeek endpoint settings. The example already sets
`MEDBENCH_CONFIG=config/local.toml`; edit that TOML file with the local Qwen
model directories. Keep credentials in `.env`, never in TOML files.

Start the configured DeepSeek-compatible service, then launch the terminal
workflow:

```powershell
uv run medbenchagent build
```

The command first asks for the benchmark requirement:

```text
Describe the Medical VLM benchmark to build: Build a 100-question breast ultrasound benchmark.
```

The CLI prints the effective non-sensitive configuration, validates required
local resources, and presents every graph interrupt as a terminal question.
After initialization it prominently prints the exact run ID (the UUID directory
name) and absolute `benchmarks/runs/<run_id>` directory before asking for the
first confirmation.
Confirm or revise benchmark information, dataset selection, task planning,
sample allocation, source selection, and question design in natural language.
It also streams Agent, RAG, generation, and audit progress before printing the
audited JSONL, manifest, reports, and run directory.

## Dataset input

The workflow selects datasets from `datasets/dataset_catalog.json`. After
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
and contains `patient`, `case`, `study`, and `image` objects. Every entity has its own
`annotation` object; image identity, path, modality, and anatomy are stored directly in
`image`. Patient, case, and study identifiers may be null, and no cross-record
consistency is inferred or enforced for those entities. `image.image_path` is stored as
opaque metadata: dataset exploration does not resolve it or check whether an image file
exists. Benchmark items are identified by `image.image_id`; mapping IDs to image files
is the responsibility of an external module. The versioned input contract is at
`src/medbenchagent/schemas/annotation.v2.schema.json`.

Validated annotation text is cached by JSONL content hash and cache-format version
under `benchmarks/dataset_cache/` as a DuckDB database with one `images` row per JSONL
line and a deterministic `task-capability-profile.v2`. The profile describes scalar
leaves under each entity annotation plus `image.modality` and `image.anatomy`. Every
eligible semantic scalar field includes up to three stable top values; strings are
limited to 200 characters. Identifiers, file references, DICOM metadata, split/fold
metadata, and bounding-box coordinates do not expose example values. CSV and XLSX inputs
are not supported.

## Planning review

The workflow computes the complete Task Plan in one model call, then presents it for
review. Respond in natural language to confirm it or describe one task to revise;
DeepSeek maps every response to a validated confirm/revise action. BenchmarkInfo review
uses the same natural-language interpretation. The working Blueprint is updated only
after confirmation: dropped tasks are removed and retained tasks receive their approved
`supporting_fields`. For `anatomy_attribute` and `finding_attribute`, Task Plan first
identifies one question type per observable semantic property, then materializes each type
as an independent task such as `finding_attribute.shape`. Every expanded task has one
answer-bearing field, while `parent_task_id` preserves its planning family. The resulting
`task_plan_report.json` is written after the first model plan and updated after each review revision; it retains decisions for both retained and dropped tasks. `rag_designs.json` is likewise written before RAG-design review and updated after each revision. Detection and visual grounding are independent capabilities:
both may be retained when the dataset has the required bounding-box and finding evidence.

The retained tasks then receive a stable category-weighted allocation of the benchmark
target. The six task-category weights are normalized across categories that retain at
least one task, then distributed by largest remainder. Within each category, allocation is
split first across original parent tasks and then evenly across the valid question types of
each parent, so decomposition does not increase a parent task's weight. The complete allocation is shown for
natural-language review; named task adjustments are fixed and the remaining count is
redistributed across unnamed tasks with the same weighting. Counts are written to the
working Blueprint only after confirmation. A zero-count task remains in the Blueprint
but skips RAG design and generation.

## Medical references

The workflow always scans the repository `materials/` directory after sample-count confirmation.
Store extractable PDFs under any meaningful first-level category directory, for example
`Standard` or `Guideline`. The directory name is used as the material category and displayed
to the user. The user is shown every categorized PDF and can select one or more files in natural
language. DeepSeek maps that response to exact relative paths, code validates the selection, and
only the selected PDFs enter RAG retrieval, indexing, and citations. PDFs placed directly under
`materials/` are invalid because they have no category directory.

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
written to this snapshot. `generation.question_template_count` is configuration-only and defaults
to three templates per newly designed task.

## Runtime configuration

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

To trace model calls in LangSmith when using Studio or the local API, set
`LANGSMITH_TRACING=true` with a valid
`LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`). The OpenAI-compatible DeepSeek
client is wrapped so RAG design, generation routing, and review calls appear as nested
LLM spans. The workflow still runs normally when tracing is disabled.

Set `LANGSMITH_TRACING=false` to prevent application traces from being
uploaded to LangSmith. A LangSmith account is still required to use the hosted
Studio interface with the local Agent Server. `medbenchagent build` always
disables LangSmith tracing, Hugging Face online downloads, and Chroma telemetry
for its process, regardless of these optional tracing settings.

The graph entry point is `medbenchagent.workflow.graph:compile_benchmark_graph`.
The deterministic root graph runs `SpecificationAgent`, `CapabilityAgent`,
`ClinicalDesignAgent`, `GenerationAgent`, and `AuditAgent` in that fixed order, with
only run initialization/finalization and artifact adapters between them. Agent-internal
confirmation, revision, RAG, and generation loops are visible with `xray=True`.

## Usage modes

### Interactive local CLI

`uv run medbenchagent build` is the supported interface for offline and
air-gapped source-checkout environments. It compiles the same
`benchmark_builder` graph in-process with a `MemorySaver`, translates its
interrupts into terminal questions, and consumes nested `tasks`, `updates`, and
`custom` streams for progress. Configuration comes exclusively from the normal
TOML and environment precedence; the command has no seed or generation tuning
arguments.

The checkpoint exists only for the lifetime of the CLI process. Ctrl+C, EOF, or
a fatal model error preserves files already written to the run directory but
cannot resume the graph after restarting the command.

### LangGraph Studio

Research users with LangSmith access may alternatively start
`uv run langgraph dev --host 127.0.0.1 --port 8123 --no-browser --allow-blocking`
and connect Studio to the `benchmark_builder` graph. Studio presents the same
human-in-the-loop interrupts with Agent Server-managed resumable state. The
local OpenAPI documentation is available at `http://127.0.0.1:8123/docs`.

### Local API

Advanced clients may call the Agent Server through its REST API or the
LangGraph SDK at `http://127.0.0.1:8123`. The API exposes the same
`benchmark_builder` graph and requires clients to handle interrupts and resume
runs. This repository does not currently provide a separate API client.

### Production deployment

`langgraph up` can provide Docker-based, production-like validation, and
LangSmith Deployment can host an Agent Server. Neither mode is supported or
tested by MedBenchAgent v0.1; production use also has separate infrastructure
and licensing requirements. See the official
[local development and testing](https://docs.langchain.com/langsmith/local-dev-testing)
and [LangGraph CLI](https://docs.langchain.com/langsmith/cli) documentation.

### Wheel and PyPI installation

Wheels produced by `uv build` are packaging checks only. They contain the
Python package and annotation schema, but not repository-level configuration,
the Blueprint template, dataset catalog, medical materials, or runtime
workspace. The current path configuration also derives the project root from
the source-tree layout. Install and run MedBenchAgent from a complete source
checkout instead of installing its wheel.

## Question design and benchmark outputs

RAG processes retained tasks one at a time and writes its result directly into each
working Blueprint task. The embedded fields are `question_templates`,
`generation_logic`, `medical_task_description`, `citations`, and, for letter tasks,
`options`, `answer_field`, and `answer_value_mapping`. There is no separate Question Design artifact or duplicate generation spec.
Natural-language review can revise one or more task designs without changing other
Blueprint tasks. New designs contain exactly the configured number of unique, semantically
equivalent question templates. Generation deterministically selects one template for each
task/image/source-row tuple from the run seed before asking the model for a controlled paraphrase;
legacy Blueprint tasks containing one valid template remain runnable.

Before designing a new letter task, RAG requests the exact categorical values of its selected
approved answer field. Code returns at most 32 frequent scalar values with counts, truncation
status, and the total distinct-value count; it never returns source rows, identifiers, bounding
boxes, DICOM metadata, or long free text. The resulting mapping preserves each raw JSON value
exactly and may merge values only when one annotated variable unambiguously determines a coarser
option. Unmapped values are ineligible for generation.

Capability screening retains a letter-response task only when its approved supporting fields
include at least one field with two or more distinct non-empty values and the values represent at
least two semantic answer classes. The profile count is a deterministic necessary condition; the
model must still reject spelling variants, synonyms, or several raw values that collapse to one
answer. Finding-presence tasks additionally require explicit positive and negative evidence.

The generator runs three independent seeded scans in a fixed order: letter VQA,
Grounding bbox, then direct-reference free text. Each stage begins from the database
start, projects only unfinished tasks' approved `supporting_fields`, and independently
accepts up to three tasks per image after a stable seed/image-ID shuffle. For new mapped letter
tasks, code reads the declared answer field, requires one uniquely located mapped raw value, and
fixes the semantic answer and evidence before the model call; the model generates only the
question stem. Legacy letter designs without a mapping retain their existing generation path.
Free text deterministically
locates its unique direct-reference string, while grounding locates the four source
coordinates of its unique annotated box. Code verifies every witness against the
original annotation before review; only path, indices, and hashes enter provenance.
Detection asks only for a generic lesion or abnormal region, while visual grounding
requires a named or described finding paired with that same box; neither task accepts a
mask as generation evidence. Diagnosis candidates are deterministically skipped before
generation when they are placeholders, delimiter-joined combinations, or resolve to
multiple scalar evidence locations. Each stage has its own prompt, validator, review,
and temporary progress file; failures do not consume the per-image acceptance limit.

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

`construction_events.jsonl` records LLM metadata plus every complete natural-language user input,
including inputs that fail parsing or validation. These raw inputs are intentionally unredacted and
may contain sensitive information, so run directories must be handled accordingly. Each successful
revision writes a linked `revision_applied` event containing the actually changed object IDs. LLM
calls use the fixed agent names, including the sampled semantic portion of `AuditAgent`, with their
operation, logical model value, token usage, and latency. Hard Audit events use `AuditAgent` without
an LLM call. `construction_report.json` uses `construction-report.v2` and reduces these events to
total and per-agent LLM usage, generation/audit item counts, wall-clock time, final confirmation
status, applied revision rounds, actual modified-object counts, and the selected material count.
Run lifecycle events use the `orchestrator` component and are not attributed to a sixth Agent.

Hard Audit verifies the image-manifest mapping, streams PNG hashes, decodes 8-bit grayscale PNGs,
and checks every item's source row against the exact annotation file used by Generation. For
every response format it replays the evidence path and source indices, then verifies the witness
hash. Free-text answers must exactly equal the replayed string, and bbox answers must equal the
four replayed coordinates. For letter items it does not reinterpret the evidence or judge its
mapping to an option for legacy designs; for structured mappings it deterministically verifies
that the replayed raw value maps to the semantic option represented by the saved answer letter.
It rejects repeated item IDs and canonical-identical items while retaining
repeated question text; `audit_report.json` reports normalized-question repetition separately in
`duplicate_statistics`. After answer balancing,
Audit deterministically samples at most ten items per task for one permissive semantic call per
task. Clear task drift, answer leakage, ambiguity/option overlap, and semantic duplication/template
collapse are copied to `benchmark.<dataset_id>.flagged.jsonl` and
`semantic_audit_records.jsonl`. Model or response failures are recorded as not-run and never
remove items from the audited output; flagged items are therefore an observation-only subset of
the audited JSONL. `audit_report.json` uses the versioned `audit-report.v1` structure with separate
`hard_audit` and `semantic_audit` sections so their counts are not treated as competing outcomes.

Balance an existing raw file without modifying it:

```powershell
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```

## Model inference and evaluation

The evaluation package runs generated benchmark artifacts against local Medical
VLM checkpoints. Use the audited JSONL when available. The loader requires the
matching `benchmark_manifest.json` and the normalized dataset's
`image_manifest.jsonl`; it verifies their hashes and resolves every PNG by
`image_id`. The opaque source `image_path` stored in benchmark items is never
used for inference.

Run one process per GPU with `torchrun`. Each process loads one complete model
copy on its `LOCAL_RANK`, processes a round-robin item shard in ordinary
batches, and writes a temporary rank file. Rank zero creates one ordered,
self-contained inference JSON after every rank succeeds:

```powershell
uv run torchrun --standalone --nproc-per-node=4 --module medbenchagent.evaluation.inference `
  --adapter qwen2_5_vl `
  --model-path models/Qwen2.5-VL-7B-Instruct `
  --benchmark benchmarks/runs/RUN_ID/benchmark.DATASET.audited.jsonl `
  --benchmark-manifest benchmarks/runs/RUN_ID/benchmark_manifest.json `
  --image-manifest datasets/NORMALIZED_DATASET/image_manifest.jsonl `
  --output results/qwen2_5_vl.json `
  --batch-size 4
```

For one GPU, the equivalent `uv run medbench-infer ...` command is available.
Inference is deterministic greedy decoding. The defaults are 512 new tokens,
SDPA, 256-1280 Qwen visual tokens, and correctness thresholds of 0.5 for both
sentence BLEU and bbox IoU. Use the CLI flags to change these values. Existing
outputs and temporary parts are protected unless `--overwrite` is explicitly
passed; interrupted jobs retain their parts for diagnosis but do not resume
them.

Generate the task, metric-group, and category report from the inference file:

```powershell
uv run medbench-eval `
  --input results/qwen2_5_vl.json `
  --output results/qwen2_5_vl.report.json
```

The evaluator recomputes item scores and checks the recorded result. It reports
accuracy, semantic-option macro-F1, SacreBLEU, and mean IoU according to each
task's manifest metric. Heterogeneous task metrics are kept in separate groups;
only thresholded correctness is summarized across all tasks. Optional
`--bleu-threshold` and `--bbox-iou-threshold` values recompute report
correctness without rerunning the model.

Model integrations live under `medbenchagent.evaluation.models`. Add one model
module implementing `ModelAdapter.generate`, then add its explicit name and
factory to the registry in that package. Adapters own checkpoint loading, chat
templates, batch preprocessing, decoding, and any model-specific coordinate
space; common parsing and scoring remain model-independent.

## License

MedBenchAgent is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) for copyright information.
