# MedBenchAgent

MedBenchAgent uses LangGraph to build annotation-driven Medical VLM benchmarks.
Prepare and register datasets, describe the benchmark, then let the agents plan tasks,
review templates, allocate samples and publish a validated delivery. Answers come from
annotations; generation does not call a vision model to infer labels.

## Install and start

Use Python 3.11+ and run from the repository root:

```powershell
uv sync
uv run medbenchagent
# Optional local data preparation dependencies:
uv sync --extra data-processing
```

Configure your language-model endpoint and model in the experiment configuration or
with the environment variables below. The defaults point to a local endpoint; no service
is installed or started automatically. Model inference uses a separate environment:
see [inference and evaluation](docs/EVALUATION.md).

The TUI home screen lists experiments. Type `/new` to create one, edit its configuration,
then `/done` to enter the conversation. Existing experiments open with Enter or `/open`.

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
Legacy `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_TIMEOUT_SECONDS` remain supported;
the corresponding nonempty `MEDBENCH_*` names take precedence.

Keep secrets in `MEDBENCH_LLM_API_KEY` (legacy `DEEPSEEK_API_KEY`),
`MEDBENCH_EMBEDDING_API_KEY`, and `MEDBENCH_RERANKER_API_KEY`. Keys are optional for
unauthenticated local services and are excluded from configuration snapshots.
CLI/TUI entrypoints disable optional tracing, model downloads and telemetry; configured
model APIs are still called.

RAG is optional. Configure embedding and reranker endpoints/models in `[rag]` and put
PDF references under `paths.materials_root`. Without these, planning continues without
retrieval. Configured service failures preserve recovery state rather than silently
changing the result. Use a new run or model identifier when service weights change.

## TUI controls and recovery

- Home: search experiments, `/new`, `/open`, `/quit`.
- Configuration: choose a field, edit and apply; `/done` continues, `/quit` returns.
  Changes affect runs that have not started. Dataset sources are managed by preparation.
- Conversation: `/details` toggles the existing sidebar; `/config` edits future settings;
  `/resume` continues unfinished work; `/queue <request>` queues independent work.
- Preparation questions: answer one field at a time or `/cancel` the pending question batch.
  Ordinary input answers a pending run question; `/queue` explicitly submits new work.
- `/quit` returns from a conversation to the experiment list, stopping active work at a
  saved boundary first. Ctrl+C exits the application through the same stopping mechanism,
  including when the configuration screen is open. A model call already in flight must finish.

Replies stream during interpretation and enter history after validation. Failed requests
remain visible but do not enter the Supervisor's accepted conversation history.
Select a run in the sidebar to inspect its identifiers, frozen settings and asset paths.

`session.json` stores conversation and queue state; `checkpoints.sqlite` under the run
root stores graph recovery state. Keep these and the referenced prepared assets intact.
Resume requires unchanged code, runtime contract and frozen inputs. After changing code,
start a new run; old checkpoints are not migrated. Current contract: `agent-runtime.v15`.

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
services or GPU checkpoints. Local preparation dependencies are installed with
`uv sync --extra data-processing`; some environment-specific cases may be skipped.
Do not treat offline tests as proof of real Docker, model or large-dataset performance.

For configuration errors, check the named key and the experiment's saved TOML. For
missing data, submit a preparation request before building. For interrupted work, use
`resume` or `resume-intake`; after code/input changes, create a new run. Personal datasets,
materials, experiments, runtime output and local scripts are intentionally ignored by Git.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright information.
