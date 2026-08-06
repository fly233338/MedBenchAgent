# MedBenchAgent

MedBenchAgent builds medical VLM benchmarks from a natural-language brief,
catalog-selected image-level `annotation.jsonl` data, and PDF medical references. The workflow screens
supported tasks, designs them with PDF RAG, and writes closed VQA, detection, finding
description, and report items into one mixed JSONL. Each run is initialized at
`benchmarks/runs/<run_id>/` before the first model call and writes an immutable,
unshuffled `benchmark.<dataset_id>.raw.jsonl`. The same directory contains provenance,
construction events/report, versioned Task Plan and RAG-design snapshots, an `effective_config.json` provenance snapshot,
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

The workflow computes the Task Plan by sending each predefined task to one sequential
model call, then runs one lightweight global consistency review over the aggregated result
before presenting it for human review. Each task call preserves the support-screening rules
and chooses one registered metric compatible with the task's response format. The global
review may remove obvious semantic duplicates or clearly unsupported retained decisions and
may rewrite reasons, but it cannot add fields or metrics. Respond in natural language to
confirm the complete plan or describe one task to revise; DeepSeek maps every response to a
validated confirm/revise action.
BenchmarkInfo review uses the same natural-language interpretation. The initial Blueprint
template does not preselect metrics. The working Blueprint is updated only after
confirmation: dropped tasks are removed, retained tasks receive their approved
`supporting_fields`, and each retained final task receives its selected metric. For
`anatomy_attribute` and `finding_attribute`, Task Plan first identifies one question type
per observable semantic property, then materializes each type as an independent task such
as `finding_attribute.shape`; every expanded task selects its own metric. Every expanded
task has one answer-bearing field, while `parent_task_id` preserves its planning family.
The resulting Task Plan drafts are stored as immutable `task_plans/v1.json`, `v2.json`, and
so on; RAG designs use the matching `rag_designs/vN.json` layout. The first per-task generation
and global review create `v1`, and each human review revision creates the next version without overwriting earlier
results. `construction_events.jsonl` records task-specific calls and the global reconciliation, and links each
revision's natural-language request, changed task IDs, artifact path, and SHA-256. Detection and visual grounding are independent capabilities:
both may be retained when the dataset has the required bounding-box and finding evidence.

`finding_presence` is used for binary presence/absence when no direct categorical finding
label is available; when such a label exists, the category-identification task is preferred.
Metric candidates and their response-format compatibility are maintained in the shared
metric registry. The task prompt receives the compatible candidates and their selection
rules; the final benchmark manifest continues to store one primary metric per task.

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
equivalent question templates. `generation_logic` is one semantically complete instruction
sentence describing how to use approved annotation semantics and the selected question template
to generate the question; `medical_task_description` remains one paragraph describing the
assessed capability followed by its medical basis. Generation deterministically selects one
template for each task/image/source-row tuple from the run seed before asking the model for a
controlled paraphrase; letter tasks require a structured answer field and answer-value mapping
before generation.

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
accepts up to twenty tasks per image after a stable seed/image-ID shuffle. All three formats
pass `medical_task_description` and `generation_logic` to their generation and review prompts.
Code fixes the answer and evidence before the model call, and every generation model returns
only the question stem. For mapped letter tasks, code reads the declared answer field, requires
one uniquely located mapped raw value, and fixes the semantic answer and evidence before the
model call. Letter designs without a mapping are rejected at the generation boundary.
Free text deterministically locates its unique direct-reference string, while grounding locates
the four source coordinates of its unique annotated box. Code verifies every witness against
the original annotation before review; only path, indices, and hashes enter provenance.
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
four replayed coordinates. For letter items it deterministically verifies that the replayed raw
value maps to the semantic option represented by the saved answer letter.
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

Run evaluation in a separate Conda environment rather than the project's locked
`uv` environment:

```bash
conda create -n medbench-eval python=3.11 -y
conda activate medbench-eval
python -m pip install -r requirements/evaluation.txt
```

If the server needs a CUDA-specific PyTorch wheel, install the matching `torch`
and `torchvision` build first, then install the requirements file. The repository
does not assume a CUDA version. The evaluation environment does not need the full
MedBenchAgent dependency set or an editable package install.

To use FlashAttention-2, install a build compatible with the active PyTorch and
server CUDA toolchain after the base requirements:

```bash
python -m pip install packaging psutil ninja
python -m pip install flash-attn --no-build-isolation
```

Keep `flash-attn` outside the base requirements because its build is specific to
the server GPU, CUDA, and PyTorch versions. Use the CLI's default SDPA when
FlashAttention-2 is unavailable.

From the repository root, verify the source modules directly:

```bash
PYTHONPATH=src python -m medbenchagent.evaluation.inference --help
PYTHONPATH=src python -m medbenchagent.evaluation.eval --help
```

Supported adapter keys are:

| Adapter key | Model family | Parsed bbox coordinate space |
| --- | --- | --- |
| `qwen2_5_vl` | Backward-compatible Qwen2.5-VL key | Processed Qwen image |
| `qwen2_5_vl_7b_instruct` | Qwen2.5-VL-7B-Instruct | Processed Qwen image |
| `qwen2_5_vl_32b_instruct` | Qwen2.5-VL-32B-Instruct | Processed Qwen image |
| `internvl3_8b` | InternVL3-8B | `[0,1000]` |
| `internvl3_5_8b` | InternVL3.5-8B | `[0,1000]` |
| `qwen3_vl_8b_instruct` | Qwen3-VL-8B-Instruct | Processed Qwen image |
| `qwen3_vl_8b_thinking` | Qwen3-VL-8B-Thinking | Processed Qwen image |
| `qwen3_vl_32b_instruct` | Qwen3-VL-32B-Instruct | Processed Qwen image |
| `lingshu_7b` | Lingshu-7B | Processed Qwen image |
| `lingshu_32b` | Lingshu-32B | Processed Qwen image |
| `hulu_med_7b` | Hulu-Med-7B | Original PNG |
| `hulu_med_32b` | Hulu-Med-32B | Original PNG |
| `qoq_med_vl_7b` | QoQ-Med-VL-7B | Processed Qwen image |
| `qoq_med_vl_32b` | QoQ-Med-VL-32B | Processed Qwen image |
| `chiron_o1_8b` | Chiron-o1-8B | `[0,1000]` |

All adapters retain the original shared task prompts. Qwen3 Thinking and InternVL3.5 extract
the suffix after the last `</think>`, Chiron extracts the suffix after the last
`### The final answer is:`, and Hulu uses its processor's `use_think=False`
decode as the answer. Full reasoning remains in `raw_prediction`; an absent
official separator falls back to the full response, while an empty extracted
answer fails strict parsing. Letter responses must be one bare valid letter or
the Lingshu-style form with one trailing ASCII period, such as `A.`. Explanations,
multiple letters, and other wrappers remain parse errors. Free text must be
non-empty, and bbox responses must be exactly one
`{"bbox":[x1,y1,x2,y2]}` JSON object.

The InternVL3.5 adapter does not install InternVL's optional Thinking System
Prompt. It alone appends a model-specific instruction forbidding reasoning and
`<think>` tags to the shared task prompt. Because the checkpoint can still emit
`<think>` blocks, the complete response is retained in `raw_prediction` and only
the suffix after the final closing tag is parsed.

For a single model/dataset combination, run one process per GPU with the current
Conda environment's Python:

```bash
PYTHONPATH=src python -m torch.distributed.run \
  --standalone \
  --nproc-per-node 4 \
  --module medbenchagent.evaluation.inference \
  --adapter qwen2_5_vl_7b_instruct \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --benchmark /path/to/benchmark.DATASET.audited.jsonl \
  --benchmark-manifest /path/to/benchmark_manifest.json \
  --image-manifest /path/to/image_manifest.jsonl \
  --output results/qwen2_5_vl_7b_instruct.json \
  --batch-size 4

PYTHONPATH=src python -m medbenchagent.evaluation.eval \
  --input results/qwen2_5_vl_7b_instruct.json \
  --output results/qwen2_5_vl_7b_instruct.report.json
```

Each process loads one complete model on its `LOCAL_RANK`, processes a
round-robin item shard, and writes a temporary rank file. Rank zero creates the
ordered inference JSON after every rank succeeds. Generation is greedy, with
defaults of 512 new tokens, SDPA, and 0.5 correctness thresholds for both BLEU
and bbox IoU. Existing inference outputs and temporary parts are protected
unless `--overwrite` is supplied; interrupted jobs do not resume.

The CLI default remains SDPA for portability. Pass
`--attn-implementation flash_attention_2` to opt in. PyTorch SDPA may itself
select a fused CUDA attention kernel, so compare end-to-end throughput on the
target server rather than assuming a fixed speed ratio. The repository ignores
the entire `scripts/` directory; local batch automation and server paths are not
tracked by Git.

Each torchrun rank loads a complete checkpoint. `NPROC_PER_NODE` adds independent
data-parallel replicas and does not shard a 32B model across GPUs. A 32B BF16
checkpoint therefore requires a GPU that can hold the complete model plus runtime
memory; multi-GPU tensor parallelism and quantization are outside this evaluator.

The evaluator recomputes item scores and checks the recorded result. It reports
accuracy, semantic-option macro-F1, SacreBLEU, and mean IoU according to each
task's manifest metric. Heterogeneous task metrics are kept in separate groups;
only thresholded correctness is summarized across all tasks. Optional
`--bleu-threshold` and `--bbox-iou-threshold` values recompute report
correctness without rerunning the model. The report's per-item `items` list
retains both the normalized `prediction` and the model's complete
`raw_prediction` for output auditing.

Every letter task also records semantic-option `macro_f1`, regardless of its
primary metric. Free-text and bbox tasks record `macro_f1: null`. Category and
summary macro-F1 values average the available letter-task macro-F1 values and are
`null` when no letter task is present.

Model integrations live under `medbenchagent.evaluation.models`. Add one model
module implementing `ModelAdapter.generate`, then add its explicit name and
factory to the registry in that package. Adapters own checkpoint loading, chat
templates, batch preprocessing, decoding, and any model-specific coordinate
space; common parsing and scoring remain model-independent.

## License

MedBenchAgent is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) for copyright information.
