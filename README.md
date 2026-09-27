# MedBenchAgent

MedBenchAgent uses LangGraph to build annotation-driven Medical VLM benchmarks.
Prepare and register datasets, describe the benchmark, then let the agents plan tasks,
review templates, allocate samples and publish a validated delivery. Answers come from
annotations; generation does not call a vision model to infer labels.

## Install and start

Use Python 3.12+ and run from the repository root:

```powershell
uv sync
uv run medbenchagent
# For DICOM, NIfTI and NRRD source formats:
uv run --extra data-processing medbenchagent
```

The default installation includes CSV and Excel preparation dependencies. Keep the
`--extra data-processing` flag on `uv run` when working with medical image formats;
it installs the additional readers and decoders for that invocation.
Raw dataset preparation uses OpenHands SDK 1.49.6 for source exploration and Python
adapter development. On its first use, `uv` installs the pinned SDK and tools into a
separate Python 3.12 environment; the prepared source is copied once into the run's
CodingWorker workspace. MedBench independently validates the worker's source report
before using it as evidence. The experiment's `data_processing.coding_worker_backend`
can be set to `legacy` during migration comparisons.

Configure your language-model endpoint and model in the experiment configuration or
with the environment variables below. The defaults point to a local endpoint; no service
is installed or started automatically. Model inference uses a separate environment:
see [inference and evaluation](docs/EVALUATION.md).

The TUI home screen lists experiments. Type `/new` to create one, edit its configuration,
then `/done` to enter the conversation. Existing experiments open with Enter or `/open`.
Use `/remove` to delete the selected experiment workspace after confirmation. This keeps
external run artifacts, the Dataset Catalog, and prepared datasets.

The same workflow is available through the CLI:

```powershell
uv run medbenchagent --help
uv run medbenchagent create pilot
uv run medbenchagent run pilot "Prepare BrEaST from D:/Data/BrEaST"
uv run medbenchagent run pilot "Build a 20-question English breast ultrasound benchmark from BrEaST"
uv run medbenchagent resume pilot SESSION_RUN_ID
uv run medbenchagent resume-intake pilot
```

`run` requires an existing experiment and prints the session Run ID used by `resume`.
Preparation and construction are separate runs; a combined request queues preparation
before its dependent build. Independent queued work runs serially.

## Configuration

| Source | Use and precedence |
| --- | --- |
| `config/default.toml` | Shared defaults; do not put personal paths or secrets here |
| Root `--config PATH` | Optional template merged into defaults when creating an experiment |
| `experiments/<name>/config.toml` | Saved experiment settings, used when each queued run starts |
| Supported nonempty field environment variables | Override TOML for a new run; empty values are ignored |
| `<run>/effective_config.json` | Frozen settings used on resume, unaffected by subsequent TOML edits |

Runs live under the project root at `.runtime/<UTC timestamp>_<short UUID>/`, with
checkpoints at `.runtime/checkpoints.sqlite`. Prepared dataset versions use the same
directory naming beneath `datasets/Prepared/<dataset_id>/`. The full UUID remains the
run's `run_id`.

Copy `config/local.toml.example` to `config/local.toml` and use
`uv run medbenchagent --config config/local.toml` to seed new experiments.
Reopening an experiment uses its saved TOML. Relative data/material/output paths resolve
from the repository root, not from the TOML's directory. The explicit CLI template path
resolves from the current working directory.

The optional root `.env` is loaded without replacing existing process variables.
`MEDBENCH_CONFIG` selects a TOML override for direct configuration consumers such as
`register` and `refresh-card`; it does not replace an experiment's saved configuration
or the CLI's explicit creation template.

Supported field overrides are `MEDBENCH_LLM_MODEL`, `MEDBENCH_LLM_BASE_URL`,
`MEDBENCH_LLM_TIMEOUT_SECONDS`, `MEDBENCH_LLM_MAX_RETRIES`, and `RAG_MIN_RERANK_SCORE`.
An existing experiment's `llm.trace_name` is ignored; new configurations omit it.

Keep the three independent credentials in `MEDBENCH_LLM_API_KEY`,
`MEDBENCH_EMBEDDING_API_KEY`, and `MEDBENCH_RERANKER_API_KEY`. Each defaults to
`local-vllm` when unset; keys are excluded from configuration snapshots and run artifacts.
CLI/TUI entrypoints disable optional tracing, model downloads and telemetry; configured
model APIs are still called.

At run initialization, each configured service queries `/v1/models` with its own key.
One model is selected automatically; endpoints with multiple models need an explicit
`model`, `embedding_model`, or `reranker_model` override. The resolved identities,
URLs and runtime policy are frozen in `effective_config.json`; resume verifies each
available frozen model using current credentials, so keys may rotate. Chat discovery
failure blocks the run. RAG remains optional: unavailable embedding or reranker services
are recorded as unavailable for that run, and planning continues without retrieval.
Start a new run to use a RAG service that becomes available later.

Raw-source CodingWorker tasks require Docker on Linux or Windows. Build the pinned worker
image before preparing a raw dataset:

```powershell
docker build -f config/coding-worker.Dockerfile -t medbench-openhands:local .
```

This requirement is independent of `data_processing.backend`, which selects the later
deterministic execution backend.

By default, the original source is mounted read-only and the Worker writes candidates
under `.runtime/<run>/workers/openhands/<dataset>/<source>/outputs/`. Docker or the configured
image being unavailable blocks the Worker. Set `data_processing.source_workspace_mode = "copy"`
only when an explicit source copy is desired; its snapshot is still mounted
read-only. MedBench validates candidates against the original source. A validator repair
starts a fresh OpenHands conversation in the same workspace, while an interrupted attempt
reuses its conversation. Loopback model URLs are routed to the Docker host and checked
against the run's frozen Chat model before the Worker starts.

## TUI controls and recovery

- Home: search experiments, `/new`, `/open`, `/remove`, `/quit`.
- Configuration: choose a field, edit and apply; `/done` continues, `/quit` returns.
  Changes affect runs that have not started. Dataset sources are managed by preparation.
- Conversation: `/details` toggles the sidebar; drag its left edge to change width.
  The progress panel follows the current run and filters activity by Agent. `/config` edits future settings;
  `/resume` continues unfinished work; `/queue <request>` queues independent work.
- Preparation questions: answer one field at a time or `/cancel` the pending question batch.
  Linux absolute paths are accepted as answers. Ordinary input answers a pending run
  question; `/queue` explicitly submits new work.
- `/quit` returns from a conversation to the experiment list, stopping active work at a
  saved boundary first. Ctrl+C copies selected text; Escape exits through the same stopping
  mechanism, including when the configuration screen is open. LangGraph drains at a saved superstep boundary;
  an in-flight model call or tool finishes or times out before stopping.

Replies enter history only after validation. Failed requests
remain visible but do not enter the Supervisor's accepted conversation history.
The run list shows status only; frozen settings and asset paths remain in the run directory.
Model/service and execution-environment failures preserve the checkpoint rather than consuming
agent correction budgets. Resolve the reported failure before `/resume`; do not repeatedly
resume an unchanged fault. CLI `resume` also continues queued work after the resumed run ends.
Malformed model output receives at most two contract-repair attempts; missing user facts remain
clarification questions. Invalid preparation answers keep the current question open.


`session.json` stores conversation and queue state; `checkpoints.sqlite` under the run
root stores graph recovery state. Keep these and the referenced prepared assets intact.
Resume requires unchanged code, runtime contract and frozen inputs. After changing code,
start a new run; old checkpoints are not migrated. Current contract: `agent-runtime.v16`.
Changing `pyproject.toml` or `uv.lock`, or changing the pinned local preparation packages,
also invalidates an existing preparation checkpoint. Create a new run after installing
missing preparation dependencies for an already-started run.

## Data preparation and delivery

Sources for one logical dataset may be associated; independent datasets require separate
IDs. Supply a raw directory in the preparation request or configure `[[dataset_sources]]`
in the template. Missing information is collected before preparation. Valid registered
data is reused by default; explicitly request rebuilding to create a new version.

`data_processing.backend` is `auto`, `docker`, or `local`. Auto probes Docker and may use
local execution for built-in operations. Generated Python requires Docker or explicit
`local` in a new run. Local execution uses the current user's permissions. The selected
backend remains fixed for the run; dependencies and Docker images are not installed automatically.

```powershell
# Optional Docker backend image:
docker build -f config/data-processing.Dockerfile -t medbench-data:local .
# Register an already complete bundle, or refresh its Card:
uv run medbenchagent register --dataset-id breast-us --bundle D:/Data/Prepared/breast-us/version
uv run medbenchagent refresh-card --dataset-id breast-us
```

Prepared bundles live under `datasets/Prepared/<dataset_id>/<version>` by default.
The configured Catalog initializes to `{"datasets":[]}` when missing; malformed existing
content is reported without replacement. Do not create an empty, non-JSON catalog file.
A Card refresh preserves underlying data; older bundles missing required integrity
bindings need explicit preparation again.

A benchmark request needs a scope and output language (`en` or `zh`). Omitted sample
count uses `benchmark.default_sample_count` (1000); invalid explicit counts are clarified.
`generation.max_tasks_per_image` applies per response format; optional
`max_total_tasks_per_image` limits all formats combined. Insufficient capacity can produce
fewer questions, with the shortfall reported; zero final questions block delivery.

A completed run publishes `delivery/benchmark.jsonl`, `benchmark_manifest.json`,
`audit_report.json`, `quarantine.jsonl`, and `complete.json`. Current contracts are
`benchmark-manifest.v4` and `audit-report.v3`. Raw outputs remain dataset-local; runtime
state, input snapshots and events stay in the run directory. A delivery references
prepared images rather than copying them, so preserve those assets when moving results.
Only a validated delivery with its completion marker is a finished benchmark.

The existing evaluation loader does not yet accept this multi-dataset construction
manifest. Its supported inputs and model instructions are in [EVALUATION.md](docs/EVALUATION.md).

## Development and troubleshooting

Read [architecture](docs/ARCHITECTURE.md) and the local `AGENTS.md` before changes.
[Data contracts](docs/DATA_LAYER.md) describe preparation and query interfaces;
[preparation validation](docs/PREPARATION_VALIDATION.md) records their acceptance boundaries.

```powershell
uv run ruff check .
uv run pytest
```

Tests cover project behavior and core contracts, without personal skills, real model
services or GPU checkpoints. CSV/Excel dependencies are installed by default;
medical image readers need `--extra data-processing`. Some environment-specific cases may be skipped.
Do not treat offline tests as proof of real Docker, model or large-dataset performance.

For configuration errors, check the named key and the experiment's saved TOML. For
missing data, submit a preparation request before building. For interrupted work, use
`resume` or `resume-intake`; after code/input changes, create a new run. Personal datasets,
materials, experiments, runtime output and local scripts are intentionally ignored by Git.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright information.
