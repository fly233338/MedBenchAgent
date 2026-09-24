# MedBenchAgent v2

MedBenchAgent uses LangGraph to construct annotation-driven Medical VLM benchmarks.
A Supervisor selects prepared datasets, delegates dataset-local planning and template review, allocates globally,
and generates every dataset before a global FinalizationAgent checks and publishes the benchmark.
It asks for clarification when needed instead of requiring confirmation at every stage.
Model inference and scoring remain a separate subsystem.

## Setup and use

Use Python 3.11 or newer and install the project dependencies:

```powershell
uv sync
uv run medbenchagent
# Optional: use an existing TOML file as the starting config for newly created experiments.
uv run medbenchagent --config config/local.toml.example
# The same conversation and run service is available without the TUI.
uv run medbenchagent create pilot
uv run medbenchagent run pilot "Build a chest X-ray benchmark in English"
uv run medbenchagent resume pilot SESSION_RUN_ID
```

The terminal interface creates `experiments/<name>/config.toml` from visible defaults and lets you edit
non-secret fields with the keyboard. `--config` seeds newly created experiments only; reopening an experiment
uses its saved TOML. Nonempty supported field environment variables still override TOML values in a new run.
In the experiment list, use Up/Down and Enter or click a row; Ctrl+F searches and Ctrl+N focuses new experiment input.
In a conversation, requests submitted while a Run works enter a serial queue. During clarification, the main
input answers the current Run; the separate Queue task field submits another request. The CLI `run` command
requires an existing experiment and prints the session Run IDs accepted by `resume`.
Keep secrets outside version control. The CLI disables optional external tracing, downloads
and telemetry; the configured language-model API is still used.
LLM overrides are `MEDBENCH_LLM_API_KEY`, `MEDBENCH_LLM_MODEL`, `MEDBENCH_LLM_BASE_URL`,
`MEDBENCH_LLM_TIMEOUT_SECONDS` and `MEDBENCH_LLM_MAX_RETRIES`. Existing `DEEPSEEK_*` aliases remain
supported; nonempty neutral names take precedence. Empty overrides are ignored; numeric overrides are
parsed and range-checked. API keys are excluded from configuration snapshots.
Optional RAG service keys are `MEDBENCH_EMBEDDING_API_KEY` and `MEDBENCH_RERANKER_API_KEY`.

Preparation and construction runs share local preflight checks. The Blueprint is required for construction,
not preparation alone. PDF materials and configured embedding/reranker endpoints are optional: when absent,
retrieval tools are disabled and planning starts with empty Materials. Configured services are called only
when retrieval runs; an HTTP failure stops the run with its checkpoint preserved for recovery. The Planner
selects references, and retrieval defaults to `rag.result_limit`. These settings are saved with the run.
Post-generation semantic auditing has been removed; remove the obsolete `[audit]` section from local overrides.

The static task template is `config/blueprint.json`. Runs never write back to this file.
Each run freezes its effective configuration, Blueprint, PDF materials, and the catalog and candidate Cards
visible at its start. The terminal stays in one experiment conversation after a run ends. Type a new task,
answer a pending clarification in the same input area, or select a run to inspect its status and frozen paths.
Ctrl+E opens the configuration editor; Ctrl+R resumes an unfinished run. A single request for preparation
and construction creates two independent runs in sequence. A build request does not automatically prepare
missing raw data: request preparation first, then start a new build run.
Conversation history and run references live in `experiments/<name>/session.json`. SQLite checkpoints remain
under the run root and are the recovery authority. Reopening an interrupted run uses its saved effective
configuration even if experiment TOML changed. Resume verifies code, source identities and selected assets.
An experiment TOML edit affects a queued Run when it starts and never changes a Run already in progress.
The global Dataset Catalog registers prepared data, while each Run freezes its own candidate view. A published
Dataset Bundle remains the data product referenced by that catalog.
The runtime contract is `agent-runtime.v14`; old checkpoints are not migrated. Code or runtime-contract
changes require a new run. Existing prepared bundles without v2 integrity bindings must be explicitly prepared again.
Keep the run directory and prepared dataset assets intact; external input changes are picked up by a new run.

The RAG service base URLs include `/v1`. vLLM provides the [OpenAI embeddings endpoint](https://docs.vllm.ai/en/latest/models/pooling_models/embed/)
and [rerank endpoint](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/). For the original Qwen3
reranker, start its service with the classification overrides required by vLLM:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8001
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8002 --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```

Use a new model name or fresh run when service weights change behind an existing model name; the Chroma
index is keyed by the configured embedding endpoint and model.

The brief describes benchmark-level `clinical_scope`, optional `diseases`, `modalities` and
`evaluation_focus`, plus the required output `language` and positive `sample_count`. Broad and multi-disease,
multi-modality scopes are valid. Unstated task priorities are not invented.
If omitted, sample count uses `benchmark.default_sample_count` (1000 by default); an explicitly invalid
count requires clarification. Output language must be `en` or `zh`; missing or unsupported values require
clarification. Required-field questions and re-extraction after an answer are routed automatically.
Requirement evidence quotes actual user inputs, not assistant questions, and stays outside BenchmarkInfo.
Modality scopes explicitly identify every covered modality, inclusion/exclusion and raw-value bindings;
alternatives such as CT or MRI stay in one scalar condition. Unsupported scope requires inspected evidence.
Without explicit allocation preferences, all levels use deterministic equal weights. Non-default or zero weights
must cite preferences extracted from the user's words; clinical topics alone do not authorize unequal weighting.
When no prepared dataset matches, the build run reports the missing data and waits for a separate preparation
request. The session Supervisor interprets preparation-only, build and combined requests; professional agents
receive explicit structured objectives, constraints, preferences and revision notes under stable instructions.
Artifact dependencies determine which actions can execute. A changed delegation invalidates its downstream outputs.
Each Dataset follows Plan → template review, with task defects returned to Planning. Allocation waits until
all current Plans and template sets are approved. Generation is deterministic and makes no model calls.

## Data preparation and the standard interface

Each published annotation represents one image, with `patient`, `case`, `study`, `image`, and
`provenance`. The published schema is `src/medbenchagent/schemas/annotation.v3.schema.json`;
existing v2 datasets are accepted through an explicit import adapter. Unknown provenance is not invented.
Answers come only from approved annotation fields.

`image_manifest.jsonl` lives beside `annotation.jsonl` and retains its four-field contract:

```json
{"image_id":"image-1","png_path":"sources/example/images/image-1.png","width":1024,"height":768}
```

Paths remain relative and beneath the bundle root. PNG/JPEG source bytes and image geometry are preserved.
IDs are namespaced by source/version/entity; missing patient/case/study IDs stay null. Original IDs and
source record references are retained in local provenance indexes and are not default answer fields.

Prepare each independent dataset once, then reuse its registered Card in benchmark runs:

```powershell
uv sync --extra data-processing
uv run medbenchagent
# In the conversation: Prepare BrEaST from D:/Data/BrEaST.
# Later: Build a 20-question English breast ultrasound benchmark from BrEaST.
uv run medbenchagent register --dataset-id breast-us --bundle D:/Data/Prepared/breast-us/version
uv run medbenchagent refresh-card --dataset-id breast-us
```

Raw inputs can be discovered from a preparation request. The DataProcessing source-discovery step checks
the directory, table headers and image files, then records validated sources in the run and experiment TOML.
It asks for a path when missing, inaccessible or lacking required table/image inputs. You may also preconfigure `[[dataset_sources]]` with
`dataset_id`, `name`, `description` and source entries (`id`, `root`, `kind`, `annotation`, optional `version`);
see `config/local.toml.example`. Relative roots resolve from the project root. Valid registered data
is reused by default even if its original raw sources are unavailable. Explicitly request rebuilding to
prepare a new version. Disabled or broken registered datasets require repair or explicit rebuilding.
Supervisor can page datasets and inspect readiness without modifying them.
Image and label files belonging to ONE dataset may be associated; independent datasets
must use separate IDs and preparation runs. The agent inspects evidence, saves typed operations,
normalizes images, validates the bundle, creates its Card, then updates the registry. `register` accepts
an already complete bundle with its Card. Failures do not replace the previous registered version.

Before saving a raw Mapping, `inspect_image_population` measures distinct image IDs using declared
source reads, filters and joins. `population_reference` binds that inspection and `expected_images` must
match its unique-image count. The inspection also records source image IDs and paths; operator and fallback
intermediates must match that population exactly before quarantine/deduplication. Emit must provide explicit IDs. Mapping also requires inspection references and source-document quotations
or resolved user decisions supporting the population choice. Explicit Harmonization decisions require
normalized-field inspections; merges, recoding, unit conversion and exclusions also require semantic evidence.
Free-text preparation `evidence` is now named `reason`; actual citations use evidence references.
References are checked against their dataset/source and input versions. `list_evidence` and `read_evidence`
recall past inspections, documents, RAG passages and user decisions without retaining their full payloads in prompts.
Source `read_text` freezes cited text/PDF documents; `freeze_documentation` selects additional documents for future
planning. Text PDFs support page references; scanned/encrypted documents require supported text input, not OCR.
Prepared v2 binds cache and asset indexes. Finalization checks the actual referenced output image bytes; changing
a database, image or frozen document requires a new prepared version. Quotation and count checks establish
provenance, not semantic correctness. Local/Docker backend selection is unchanged.

Prepared versions live beneath `paths.prepared_datasets_root` (default `datasets/Prepared`). Catalog entries
contain dataset_id, name, description, available and paths to annotation, image manifest, Profile and Card.
Existing normalized datasets need preparation before selection; build does not prepare the whole catalog.
Already prepared bundles with a missing or v1 Card can use `refresh-card` without reprocessing data.
Cards describe only the dataset, independent of Blueprint. Source descriptions come from verified document
quotations; the benchmark goal is never passed to Card generation. The preparation agent must explicitly
save metadata or record an empty reference list after checking for reliable documentation before publication.
Missing metadata is not silently filled during finish. A refresh creates a new Card revision, updates the
registry only after validation, and preserves old Cards, images, annotations and Profile/cache.
Selection saves all candidate summaries/digests and their order (including unselected datasets), exact
name-resolution inputs and exclusion reasons in `dataset_candidates.json` and `dataset_selection.json`.
Resume reuses the frozen decision context; external Card updates do not rewrite historical decisions.
Explicit dataset selections are honored; otherwise Card summaries guide high-recall selection.
Explicit dataset names in a request are resolved by the session Supervisor. Datasets found only in raw sources
require a separate preparation run. Unknown, disabled or broken
explicit datasets produce a diagnostic instead of being silently replaced.
Selection is a shortlist for deeper planning, not proof that a dataset supports a particular task.

Planning grounds tasks in real annotation fields and values; Cards only guide inspection. The final review
receives full answer mappings, exact capacities and quoted semantic evidence. Opaque codes or ambiguous
abbreviations require dataset documentation, retrieved references or user clarification; medical convention
does not establish a dataset's encoding. Semantic evidence and grounding for every raw-value-to-option mapping are saved with each Leaf. Direct
mappings preserve self-describing strings; codes and semantic reinterpretations need dataset-specific
evidence. Validation checks coverage and quotations; the Planning review judges whether they support the mapping. Zero-capacity leaves
cannot be published. Unsupported decisions require current field/Profile/document inspection references;
review checks their reasons against the cited inspections. Completed empty dataset plans remain valid.
Generation still derives answers deterministically from annotations, and template models never see per-image answers.
Explicit sample restrictions are stored separately from evaluation topics and resolved into dataset-local
scalar `in`/`not_in` conditions. Missing values fail selection. Planning capacity, generation and final validation
share these conditions; unsupported restrictions cannot be satisfied by prose alone. The Planner discovers
local documentation through existing source inventories before asking about ambiguous codes.
Cross-agent answers retain their issue ID and return to the originating agent as semantic evidence.
Only explicit requirement changes invalidate the brief and its downstream artifacts.
Template sets are reviewed after Planning and before Allocation. Wording defects are revised in the template
stage; task defects reopen Planning. Saved approval results are reused for unchanged specifications and
configuration. Both template attempts and Planning reworks retain their budgets across recovery.
Rejected templates and their specification accompany revision feedback. Unchanged specifications are rejected
before another review or Plan publication; repeating the same feedback does not charge another Planning rework.

Finalization starts only after every positive-allocation Generation completes. The FinalizationAgent can
inspect datasets and anomalies, assemble candidates, balance answer positions, validate the package and
publish. Invalid items are quarantined with source file/line/content/reasons; original generation files remain
unchanged. Duplicate final identities retain one equivalent item or quarantine all conflicting items. Letter
positions are balanced per final `dataset::leaf` task without changing correct-answer text or options.

Deliveries live in `delivery/` and contain `benchmark.jsonl`, `benchmark_manifest.json` (v4), `audit_report.json`,
`quarantine.jsonl` and `complete.json`. A complete marker is written before an atomic same-filesystem directory
rename; interrupted publication cannot expose a partial delivery. Published packages are never overwritten.
The global `raw_source_paths` map and manifest reference each unchanged Dataset generation; there is no merged
`benchmark.raw.jsonl`. Image paths remain references to prepared assets, not copied portable image bundles.

The report gives requested/allocated/generated/accepted/final counts and distinguishes planning capacity,
generation exhaustion, audit rejection and assembly removal. Shortfalls do not require confirmation and do
not trigger automatic refill: nonempty valid packages are delivered with `complete` and an explicit shortfall.
Zero final samples or invalid packages are blocked. Recovery reuses the published package, never appends or
balances again. Runtime contract changes require a new run; old checkpoints are not migrated.

Choose the execution backend in `config/default.toml` or your local override:

```toml
[data_processing]
backend = "auto" # auto | docker | local
```

`auto` probes the Docker CLI, daemon, configured image and container startup before preparation. If
unavailable, it records why and runs local Python subprocesses. `docker` requires a working container;
`local` never probes Docker. To prepare the Docker image manually:

```powershell
docker build -f config/data-processing.Dockerfile -t medbench-data:local .
```

The backend is fixed for that preparation version and recorded in `execution_runtime.json`. Processing
errors never trigger a backend switch. Docker retains read-only mounts, disabled networking and resource
limits. Local execution supports both fixed operators and Agent-generated Python with current-user
permissions; it does not provide Docker filesystem/network isolation. Both use timeouts, independent
work/scratch directories and the same provenance, output and repeatability checks. DuckDB memory and
thread settings apply to both; container_memory applies only to Docker. Missing local libraries report
an actionable error rather than installing packages automatically.

Generated scripts read MEDBENCH_SOURCE_ROOT and write under MEDBENCH_WORK_DIR; optional annotations use
MEDBENCH_ANNOTATION_PATH, and scratch uses MEDBENCH_DATA_SCRATCH. Do not hardcode container paths.

Sources are not copied wholesale. Source file inventories and content digests detect changes before reuse
and publication; keep external sources stable during processing. Read-only mounts do not freeze the host
filesystem. Only referenced processed image assets are materialized, once in the bundle; duplicate
verification outputs are removed after comparison. New source contents require a new input version.

Local bad image records are quarantined with reasons. Systemic join/identity errors, missing decoders,
non-determinism and unresolved display decisions block the affected operation. Multiple findings and
missing optional fields are valid data; eligibility explains why a particular Protocol cannot use them.
Exact duplicate pixels with conflicting annotations/identity are quarantined, never silently merged.

Only 2D images are supported. Multiframe DICOM, true volumes, WSI and pyramids are unsupported; no automatic
slice selection, cropping, resizing, coordinate clipping or perceptual deduplication is performed.
DICOM metadata inspection reads selected headers without requesting pixel data.

Bundles include source manifests, source-local plans/provenance, `quality_report.json`, `quarantine.jsonl`,
`dedup.jsonl`, `harmonization.json`, and one published DuckDB/Profile cache. `ready.json` is written only
once all selected sources and final validation succeed. See [the data layer reference](docs/DATA_LAYER.md)
for operator contracts and query examples.

Profile v4 provides a compact summary and field/array indexes. Planning queries complete value vocabularies,
per-source coverage, samples, same-entity pairs and exact Protocol eligibility on demand. Capacity,
Generation and answer audit share the same resolver and published population.

## Planning, templates and audit

For each selected prepared dataset, TaskPlanningAgent incrementally builds exactly two levels: Medical Concept → executable Leaf.
Core Concepts come from `config/blueprint.json`; each leaf has a runtime-defined stable ID and an explicit
`concept_id`. IDs never determine execution; dots in an ID do not create additional tree levels. A leaf Protocol binds annotation fields, its answer contract,
execution type (`letter_classification`, `bbox_grounding`, or `free_text`), response format, metric and intent.

RAG is available while understanding Concepts, expanding Leaves, designing Protocols and discovering
extensions. It explains medical terminology; only annotations establish fields and answer values. Empty
reference selection is supported. After Core planning, coverage-gap inspection can lead to Extension
proposals. Dataset validation and model semantic review must accept each proposal before it joins the run.

Global Review identifies targeted corrections. Correction rounds consume the persistent
`harness.max_reworks` budget; redelegation and checkpoint recovery do not reset it. Unresolved issues at
budget exhaustion block Plan publication. Current Plan readiness also requires valid Protocols and exact per-Leaf eligible_count; no sample quotas are stored in Plan.
Reading the same exhausted Review again preserves its blocked status and draft reference without another
model call. Upstream requirement or dataset changes refresh the draft metadata and require revalidation,
while preserving leaf identities and spent rework budget. Accepted Extension proposals and their decisions
can be inspected separately with `read_proposal`.

Persisted Concepts, Leaves and PlanningDrafts reject unknown fields and invalid ownership. Draft validation
permits unfinished nodes; final Plan validation returns located `path`/`message` issues for structure and
readiness failures. These checks reuse artifact version gates and do not rescan the dataset on every action.

After all local plans complete, AllocationAgent supplies justified Concept importance and optional
Dataset/Leaf weights. Code allocates Concept → Dataset → Leaf with deterministic capacity clipping and
redistribution. Lower levels default to equal coverage; more Leaves do not give a dataset more budget.
Zero weights exclude members. Core Concepts are shared; Extensions and Leaf IDs remain dataset-local.
A completed unsupported plan may receive zero allocation; an execution failure cannot be silently omitted.
Independent Leaf capacities do not guarantee joint feasibility under per-image limits, so actual generation
shortfalls remain explicit. No automatic refill loop is introduced.

`generation.max_tasks_per_image` limits each response format separately. Set the optional
`generation.max_total_tasks_per_image` to cap all formats combined for each `(dataset_id, image_id)`;
by default there is no combined limit. For example, “at most one question per image in total” sets the
combined limit to 1 and overrides the configured default. Generation preserves its deterministic stage
order; shared images can therefore cause shortfalls despite sufficient independent Leaf capacities.
The final report records both limits, and final validation checks them again.

Generation uses a transactional SQLite progress ledger. Finalization reads referenced records in batches,
uses disposable DuckDB indexes for candidates and duplicates, and balances answer positions as a stream.
The new algorithm is reproducible for the same inputs and seed, but may differ from older answer permutations.

The pre-allocation template stage builds and reviews a separate `templates.json` library, controlled by
`generation.question_template_count`. The builder and reviewer interfaces accept only a strict TemplateSpecification without
answer mappings or per-image answers. Protocol-defined slots allow `{modality}` and, when a target is
explicitly bound, require `{target}`. Planning leaves never contain templates; the private GenerationTask
adds templates only when assembling GenerationRequest. Code renders every question and resolves its answer
from the full Protocol without per-item LLM calls.

Hard audit checks images, sample scope, duplicates and Protocol-bound annotation answers. Final
`total_shortfall` is the requested total minus final delivered items, without double-counting planning,
generation and assembly deficits. Final items are written to `benchmark.jsonl`, accompanied by `audit_report.json`,
`quarantine.jsonl` and `benchmark_manifest.json`. Task IDs are `dataset_id::local_leaf_id`, with explicit source
mappings in manifest v4; local image IDs are preserved and image paths are rebased. Annotations and image
manifests remain dataset-local. Repeated export does not append duplicate samples.
Evaluation and scoring are unchanged; existing evaluation loaders do not accept this v3 manifest. Concept-level evaluation
aggregation is not implemented yet.

## Outputs and execution limits

The final run directory contains:

- `blueprint.json` and immutable planning versions under `.runtime/`.
- `templates.json`, raw items and the audited benchmark JSONL.
- `benchmark_manifest.json` with task metrics and relative artifact paths.
- One `audit_report.json` with hard and semantic decisions.
- `construction_events.jsonl` and the non-sensitive `effective_config.json`.
- Internal replay receipts and preparation/batch work under `.runtime/`.

Final counts and deficits refer to audited items. Event records include LLM usage and latency;
no separate aggregate construction report is generated.
The manifest uses `benchmark-manifest.v2` with additive concepts, allocation and explicit leaf concept IDs.
Historical outputs are not migrated or deleted automatically. The new runtime contract is `agent-runtime.v4`;
start a new run instead of resuming an older checkpoint.

The `[harness]` settings limit agent actions, elapsed execution time, consecutive failures and repeated rework.
Human waiting time is excluded from the execution budget. Each agent retains bounded private tool observations.
Specialists ask local questions directly; only global goal/resource decisions go through the Supervisor.
Runtime receipts bind validated parameters and dependency versions to completed outputs. The compact
`.runtime/artifacts.json` index makes stale Plans, Generations and Audits ineligible for delivery. An exhausted budget returns a blocked status,
rather than reporting a completed benchmark. Semantic uncertainty is surfaced to the user.

## Development

Read `AGENTS.md` and `docs/ARCHITECTURE.md` before editing.
Verification uses static checks and unit tests; actual model/API and conversion runs require the configured server environment.

```powershell
uv run ruff check .
uv run pytest
```



## Model inference and evaluation

The evaluation package runs generated benchmark artifacts against local Medical
VLM checkpoints. Use the audited JSONL when available. The loader requires the
matching `benchmark_manifest.json` and the normalized dataset's
`image_manifest.jsonl`; it validates their task and image contracts and resolves every image by
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
| `breastgpt_8b` | BreastGPT-8B | `[0,1000]` |
| `qwen2_5_vl` | Backward-compatible Qwen2.5-VL key | Processed Qwen image |
| `qwen2_5_vl_7b_instruct` | Qwen2.5-VL-7B-Instruct | Processed Qwen image |
| `qwen2_5_vl_32b_instruct` | Qwen2.5-VL-32B-Instruct | Processed Qwen image |
| `internvl3_8b` | InternVL3-8B | `[0,1000]` |
| `internvl3_5_8b` | InternVL3.5-8B | `[0,1000]` |
| `qwen3_vl_8b_instruct` | Qwen3-VL-8B-Instruct | Fixed `[0,1000]` |
| `qwen3_vl_8b_thinking` | Qwen3-VL-8B-Thinking | Fixed `[0,1000]` |
| `qwen3_vl_32b_instruct` | Qwen3-VL-32B-Instruct | Fixed `[0,1000]` |
| `lingshu_7b` | Lingshu-7B | Processed Qwen image |
| `lingshu_32b` | Lingshu-32B | Processed Qwen image |
| `hulu_med_7b` | Hulu-Med-7B | Original PNG |
| `hulu_med_32b` | Hulu-Med-32B | Original PNG |
| `qoq_med_vl_7b` | QoQ-Med-VL-7B | Processed Qwen image |
| `qoq_med_vl_32b` | QoQ-Med-VL-32B | Processed Qwen image |
| `chiron_o1_8b` | Chiron-o1-8B | `[0,1000]` |
| `medgemma_1_5_4b_it` | MedGemma-1.5-4B-IT | Normalized `[0,1]` yxyx, converted to xyxy |

BreastGPT requires the local ModelScope checkpoint directory, including its bundled
`plugin/` package. Download the complete snapshot before running inference:

```python
from modelscope import snapshot_download

snapshot_download("YYangYang/BreastGPT-8B", local_dir="/path/to/BreastGPT-8B")
```

Then use `--adapter breastgpt_8b --model-path /path/to/BreastGPT-8B`. The adapter loads
the checkpoint's official `breastgpt` model and template registrations through
MS-SWIFT. Its legacy `<|box_start|>...<|box_end|>` 2D grounding output is normalized
to the evaluator's single-box contract while the complete response remains available
as `raw_prediction`.

InternVL3-8B limits dynamic preprocessing to six aspect-ratio tiles and keeps the
official thumbnail when multiple tiles are used, for at most seven `448x448`
inputs per image. Other InternVL-family adapters retain the official limit of 12.

MedGemma is a gated checkpoint. Accept Google's Health AI Developer Foundations
terms before downloading it, then point the adapter at the local model directory.
MedGemma keeps the shared task prompt for non-bbox tasks and appends a model-specific
requirement to end with `FINAL_ANSWER:`. Its adapter prefers that boundary and otherwise
recognizes the
checkpoint-observed `answer is`, `category is X.`, Markdown option, and `\boxed{X}`
letter forms. For bbox tasks it follows MedGemma's native localization protocol: one
JSON-list object with normalized `[y0,x0,y1,x1]` `box_2d` coordinates. The adapter
strictly validates `[0,1]`, converts yxyx to canonical xyxy, and then scales to the
original image. Conflicting or malformed candidates remain parse errors.

All bbox prompts state the coordinate protocol owned by their adapter family. Qwen3,
InternVL, Chiron, and BreastGPT request fixed `[0,1000]` xyxy coordinates; Qwen2.5,
Lingshu, and QoQ request absolute xyxy coordinates in the resized image processed by
the model; MedGemma requests normalized `[0,1]` yxyx coordinates; and Hulu requests
original-image pixel coordinates. BreastGPT requests its native grounding-token format
through the bundled MS-SWIFT template and reports grounding in `[0,1000]`.
QoQ uses its native medical
reasoning prompt and extracts the final balanced `\boxed{...}` payload. Qwen3
Thinking extracts the suffix after the last `</think>`. InternVL3.5 uses explicit
`<think>...</think>` reasoning and `<answer>...</answer>` final-answer sections. Chiron
preserves its checkpoint system message, appends the official step-by-step trigger,
requires the `### The final answer is:` boundary, and extracts the suffix after its
last occurrence. The extractor also accepts the checkpoint-observed form without
the leading Markdown heading, `The final answer is:`. Hulu uses its processor's
`use_think=False` decode as the answer. Full reasoning remains in `raw_prediction`; an absent
official separator falls back to the full response, while an empty extracted
answer fails strict parsing. Letter responses may be one bare valid letter, a
parenthesized letter such as `(A)`, `(B) DENSITY B`, `(B. Asymmetry)`, or `（A）`, the Lingshu-style form with one
trailing ASCII period such as `A.`, or a valid letter
and period followed by option text or an explanation such as `A. Benign` or
`A. because ...`. Multiple letters without this prefix form and other wrappers
remain parse errors. Free text must be non-empty, and bbox responses must contain exactly
one distinct box. The bbox parser accepts `bbox` or `bbox_2d` fields in JSON, Markdown,
or surrounding text, plus a bare four-number list or tuple when it is the complete answer.
Repeated identical boxes are deduplicated; zero or multiple distinct boxes remain parse
errors before the existing coordinate-space checks and rescaling.

The InternVL3.5 adapter installs a model-specific reasoning system prompt based on
InternVL's optional Thinking Mode protocol. It requires all reasoning inside
`<think>...</think>` and the task-constrained final answer inside
`<answer>...</answer>`. The complete response remains in `raw_prediction`; the adapter
extracts the last complete answer section and retains the prior `</think>` suffix as a
legacy fallback. InternVL3 remains on the shared direct-answer prompt and behavior.
Chiron and InternVL3.5 explicitly
set `pad_token_id` to the tokenizer EOS ID to avoid the open-end generation warning.

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
  --task-types letter bbox free_text \
  --output results/qwen2_5_vl_7b_instruct.json \
  --batch-size 4

PYTHONPATH=src python -m medbenchagent.evaluation.eval \
  --input results/qwen2_5_vl_7b_instruct.json \
  --output results/qwen2_5_vl_7b_instruct.report.json
```

Each process loads one complete model on its `LOCAL_RANK`, processes a
round-robin item shard, and writes a temporary rank file. Rank zero creates the
ordered inference JSON after every rank succeeds. Completion is coordinated by
per-rank files rather than a post-inference NCCL barrier, so uneven inference times
are not limited by NCCL's default 10-minute process-group timeout. Generation is
greedy, with
defaults of 512 new tokens, SDPA, and 0.5 correctness thresholds for both BLEU
and bbox IoU. Existing inference outputs and temporary parts are protected
unless `--overwrite` is supplied; interrupted jobs do not resume. `--task-types`
is required, accepts one or more unique values from `letter`, `bbox`, and `free_text`,
and filters samples before model loading. Every requested type must be present in the
benchmark. The selected order is recorded in inference metadata.
Each inference item also records the effective model `coordinate_size` used by the bbox
parser. This field is optional in the backward-compatible `medbench-inference.v1`
schema, so older files remain readable.

If an existing inference JSON contains BreastGPT letter outputs such as `（A）`,
the model does not need to be run again. Recompute the report directly from the
stored `raw_prediction` values:

```bash
PYTHONPATH=src python -m medbenchagent.evaluation.eval \
  --input results/breastgpt_8b.json \
  --output results/breastgpt_8b.reparsed.report.json \
  --reparse
```

`--reparse` only re-extracts letter answers and recalculates their scores; bbox and
free-text results continue to use their recorded predictions. The original inference
JSON is not modified, and no checkpoint, GPU, or benchmark/image manifest is needed.

To recover bbox predictions from an existing inference file, reparse its stored raw
responses with the recorded coordinate space:

```bash
PYTHONPATH=src python -m medbenchagent.evaluation.eval \
  --input results/qwen3_vl_8b_instruct.json \
  --output results/qwen3_vl_8b_instruct.reparsed.report.json \
  --reparse-bbox
```

`--reparse-bbox` prefers each item's `coordinate_size`. Legacy Qwen3-VL files may omit
that field because their fixed `1000x1000` protocol is recoverable from adapter metadata;
Qwen3 Thinking reparsing uses only the final answer after the last `</think>`. Older
files from dynamic-coordinate adapters must contain `coordinate_size` and otherwise
fail explicitly instead of guessing a protocol. Reparsing only writes the new report
and never modifies the inference JSON.

For MedGemma, bbox reparsing applies the corrected native normalized-yxyx protocol even
when an older inference item recorded the former, incorrect `896x896` coordinate space.

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

The evaluator recomputes item scores and checks the recorded result. One report file
is a response-format-keyed object whose selected `letter`, `bbox`, and `free_text`
values are complete `medbench-evaluation.v2` reports. Each section independently
contains `metadata`, `summary`, `metric_groups`, `categories`, `tasks`, and `items`.
It reports accuracy, semantic-option macro-F1, SacreBLEU, and mean IoU according to each
question type's manifest metric. For letter and free-text sections, `metric_groups` keeps
the existing two hierarchical display values. For bbox sections it instead exposes the
native `acc_at_iou` and `miou` names: each question type first contributes its threshold
accuracy and mean IoU to an equal-weight mean within its parent task, then the parent-task
values are averaged with equal weight. Optional
`--bleu-threshold` and `--bbox-iou-threshold` values recompute report
correctness without rerunning the model. The report's per-item `items` list
retains both the normalized `prediction` and the model's complete
`raw_prediction` for output auditing.

For bbox sections, summary, category, task, and metric-group records use `miou` and
`acc_at_iou`; per-item records use `iou`. Acc@IoU uses the report's configurable
`bbox_iou` threshold (0.5 by default). This single-box output contract has no confidence
score, so the evaluator does not claim standard AP/mAP.

Every letter task also records semantic-option `macro_f1`, regardless of its
primary metric. Free-text tasks record `macro_f1: null`; bbox sections omit macro-F1
entirely. Category and summary macro-F1 values average the available letter-task
macro-F1 values and are `null` when no letter task is present.

Model integrations live under `medbenchagent.evaluation.models`. Add one model
module implementing `ModelAdapter.generate`, then add its explicit name and
factory to the registry in that package. Adapters own checkpoint loading, chat
templates, batch preprocessing, decoding, and any model-specific coordinate
space; common parsing and scoring remain model-independent.

## License

MedBenchAgent is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) for copyright information.
