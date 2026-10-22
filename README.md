# MedD2B — Medical Data to Benchmark

## Independent Compilation

The `refactor` branch provides an independent LangGraph coordination core with named
agent nodes, explicit work handoffs, a Case-based file reader and SQLite checkpoints.
The Supervisor supports `scripted` (default, offline) and `llm` modes. Professional
Coding and Planning agents still use **scripted synthetic implementations**; neither
mode uses OpenHands or a medical dataset. The output labels Supervisor and professional
agent modes separately. This is not a complete autonomous benchmark compiler.

```powershell
uv sync --locked --python 3.12
uv run --locked python -m medbenchagent.compilation run --directory .runtime/compilation-demo --compilation-id demo --pause-for-input
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/compilation-demo --compilation-id demo --answer "Continue"
uv run --locked pytest tests/test_compilation_graph.py -q
```

Use a separate run directory and add `--initial-evidence` to start with an existing
overview. In scripted mode this selects a different delegation order in the same graph.
In model mode the Supervisor chooses the next action from registered capabilities and evidence.

For model mode, configure a reachable Chat service in a TOML file (the existing
`[llm]` keys) and provide any credential through `MEDBENCH_LLM_API_KEY`:

```powershell
uv run --locked python -m medbenchagent.compilation run --directory .runtime/model-supervisor --compilation-id live --supervisor llm --config config/local.toml
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/model-supervisor --compilation-id live
uv run --locked pytest tests/test_compilation_supervisor.py tests/test_llm.py -q
```

`--goal` supplies a goal for a new run; its default requests the synthetic Case-linked task
and derived reference data. `--pause-for-input` is a scripted demonstration option;
the model decides for itself whether to ask a question. Add `--answer` on resume only
when a question is pending.

Only `llm` and `supervisor` configuration sections are loaded for model mode. There is no
RAG or Preparation preflight. Limits `supervisor.context_chars`, `artifact_chars` and
`max_artifacts` bound the full request, each preview and preview count. Truncation and omitted
previews are explicit; an oversized full request fails with its size instead of silently
dropping required context. Invalid model choices use the existing bounded repair path.

The run directory holds `run.json` (mode, goal, options and non-secret model configuration),
`checkpoints.sqlite` and request-owned files under `artifacts/`. Reusing `run` on an existing
directory is rejected. `resume` uses the saved mode/model without reading a new configuration
or rediscovering a model; credentials can still be supplied through the environment.
Model failures preserve checkpoints and never fall back to Scripted execution.
No compatibility reader is provided for earlier demonstration runs.

See the [architecture](docs/ARCHITECTURE.md), [agent boundaries](docs/agents/README.md),
[Case contracts](docs/DATA_LAYER.md), and [implementation status](docs/PLAN.md).
The existing `medd2b` CLI/TUI below remains a migration reference; the core does not call
its Workflow, Preparation Graph, Resolver, BIRStore or harness.

## Existing capabilities awaiting migration

MedD2B compiles one annotated medical dataset into a traceable Medical VLM benchmark. A real Supervisor chooses expert work, inspects coverage, requests scoped revisions, and publishes only after semantic review and deterministic audit.

Provide a dataset or raw source and optional requirements. With no clinical goal, Task Planning designs tasks from the available evidence. Explicit clinical goals constrain discovery and screening. One dataset may contain multiple sources, modalities, images and linked studies.

## Start

```powershell
uv sync
uv run medd2b
uv run medd2b create pilot
uv run medd2b run pilot "Compile this dataset" --dataset breast-us
uv run medd2b run pilot "Prepare BrEaST from D:/Data/BrEaST and compile 20 English questions"
uv run medd2b resume pilot RUN_ID
```

The no-command entry opens the Textual interface. Plain text starts a new compilation unless a question is active; then it answers that question. Use `/queue <request>` to create a separate compilation, `/resume` for recovery, and `/details` for Agent decisions and diagnostics.

A preparation-only request stops after accepted data registration. The initial startup installs the configured preparation runtime when necessary. Model endpoints and data paths come from configuration; credentials remain in environment variables.

## Independent batches and delivery

```powershell
uv run medd2b batch pilot breast-us chest-xray --request "Compile 100 English questions"
uv run medd2b export PATH_TO_PUBLISHED_DELIVERY PATH_TO_NEW_EXPORT
```

Each batch entry has independent requirements, BIR, quota and output. There is no cross-dataset allocation or automatic merge. Execution is serial; an unresolved question pauses the queue.

Deliveries contain benchmark items, a manifest, audit report, quarantine and a readable compilation report. Default packages reference verified local images. Explicit export copies model-input images and uses package-relative paths. A completion marker binds all delivered files.

Items use the strict canonical fields `item_id`, `dataset_id` and `leaf_task_id`; legacy aliases are rejected. Dataset access receipts bind real profile counts and file content. Generation reads source pages using `generation.fetch_batch_size`.

```python
from medbenchagent.evaluation.loader import load_benchmark_package
package = load_benchmark_package("/path/to/published-or-exported-package")
items = package["items"]  # ordered input_images resolve to verified absolute paths
```

Model inference remains a separate deployment. The package reader requires no construction checkpoint or Prepared cache. Existing independent inference adapters need to consume this canonical interface; old benchmark formats are not supported by this reader.

## Ownership and configuration

- `experiments/<session>/config.toml`: defaults for future compilations.
- `experiments/<session>/compilations/<id>/config.toml`: configuration copied when submitted.
- `experiments/<session>/compilations/<id>/BIR/`: one compilation's canonical business documents.
- The compilation's `outputs/`: audited candidates and published deliveries.
- `.runtime/<run>/`: execution checkpoint, events, operations and Agent-private state.
- `datasets/Prepared/`: reusable prepared dataset revisions.

Each BIR has one `dataset_binding.json`, `task_plan.json`, `generation.json` and `planning_extensions.json`. Planning requires that BIR and its frozen Registry. Batch processing queues independent compilations; compilation state does not maintain dataset-indexed Plan or generation maps. Historical files are preserved, but old internal layouts and checkpoints are not migrated or read through compatibility branches.

The Python import package remains `medbenchagent`; the product and console command are MedD2B / `medd2b`. Shared defaults live in `config/default.toml`. Language defaults to English and the sample budget to 1000; both are configurable. Missing clinical scope is valid.

Preparation execution limits come from `data_processing.preparation_stage_budgets`. Repair allowances count consecutive attempts without Host-verified progress, independently of clarification and cumulative execution limits. `planning.context_chars` covers the full serialized planning/review request, including repair turns.

The local PDF RAG component retains embedding and reranking with content-bound caches. It is not uniformly available to all semantic model calls; shared registration, autonomous retrieval loops and OpenHands cross-process access remain deferred.

## Development

```powershell
uv run ruff check .
uv run pytest
```

See [architecture](docs/ARCHITECTURE.md), [Agent responsibilities](docs/agents/README.md), [data contracts](docs/DATA_LAYER.md), and [preparation validation](docs/PREPARATION_VALIDATION.md). Offline tests use real small image fixtures and scripted model decisions. They do not establish live-model quality, Docker reliability or large-dataset throughput.

The [real-runtime checklist](docs/PREPARATION_VALIDATION.md#real-runtime-validation-checklist--pending) records those outstanding checks and reproducible commands separately.

[Apache License 2.0](LICENSE).
