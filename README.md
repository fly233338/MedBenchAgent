# MedD2B — Medical Data to Benchmark

## Independent Compilation

The independent Compilation entrypoint uses named LangGraph nodes and formal WorkRequest /
WorkResult handoffs. Supervisor and Task Planner support `scripted` or `llm`; the model
implementations use LangChain `create_agent`, native ToolStrategy, private persistent state,
summarization and invocation limits. Coding remains an optional OpenHands adapter with its own
execution loop. The default Scripted agents are synthetic development fixtures.

The native Planner reads a canonical Case dataset, records Axis and Scenario judgments, revises
semantic tasks, requests and accepts derived data, and publishes immutable Task Plans.
It does **not** generate or deliver benchmark items. GenerationSpec currently supplies the first
closed-VQA execution contract and validation only; new Generation, Review and Delivery execution
remain future work. The old `medd2b` CLI/TUI below still uses the old workflow.

### Offline synthetic demonstration

```powershell
uv sync --locked --python 3.12
uv run --locked python -m medbenchagent.compilation run --directory .runtime/compilation-demo --compilation-id demo --pause-for-input
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/compilation-demo --compilation-id demo --answer "Continue"
```

Use a new directory for each compilation. `--initial-evidence` adds a synthetic overview and
demonstrates a different delegation path. `--pause-for-input` belongs to the Scripted Supervisor.
The model Supervisor decides when a question is needed.

### Real Dataset input and native planning

Prepare a small reproducible BrEaST snapshot directly from the source workbook and PNG files.
The script defaults to the first 20 source records ordered by numeric CaseID. Source files are
read-only; the output must be a new directory. Each Case represents one source lesion record,
with unknown patient/study grouping preserved. Annotations and masks remain reference-only.

```powershell
uv run --locked python scripts/prepare_breast_example.py --workbook datasets/US/BrEaST/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx --image-root datasets/US/BrEaST --output .runtime/breast-input
uv run --locked python -m medbenchagent.compilation run --directory .runtime/breast-planning --compilation-id breast-planning --supervisor llm --planner llm --dataset-manifest .runtime/breast-input/manifest.json --config config/local.toml
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/breast-planning --compilation-id breast-planning
```

Copy `config/local.toml.example` to `config/local.toml` and configure the common `[llm]`
endpoint/model. Credentials use `MEDBENCH_LLM_API_KEY`. The endpoint must support tool calling;
actual model compatibility and semantic quality are evaluated by the operator.
The implementation's offline tests do not call a real model.

`--dataset-manifest` accepts only the current canonical DatasetManifest, including content
members. The CLI copies verified records, Mapping, assets and referenced source files into
the run. Real input requires `--supervisor llm --planner llm` and never creates synthetic data
or registers synthetic Coding. The default real-input goal is to publish an evidence-grounded
Task Plan; use `--goal` for explicit requirements. Data that needs processing produces a formal
requirement; enabling existing Coding is optional, and the new Data Processing role is not yet implemented.

### Optional OpenHands Coding

Use `--coding openhands` to register the existing local Coding capability. Its independent
Python environment must already be provisioned using `config/coding-worker.requirements.txt`.
`[coding.llm]` overrides common model fields, and `MEDBENCH_CODING_LLM_API_KEY` can provide a
separate credential. Missing values inherit the common configuration.

```powershell
uv run --locked python -m medbenchagent.compilation run --directory .runtime/coding-demo --compilation-id coding-demo --supervisor llm --coding openhands --config config/local.toml
```

Each Workstream owns its SDK Conversation and workspace. Request-local candidates are copied into
formal handoffs. No automatic worker installation, generic session manager or exactly-once shell
execution is provided. This iteration does not move Coding inside a Data Processing agent.

### Ownership, limits and recovery

The run directory contains:
- `run.json`: selected modes, goal, initial source location, non-secret model/limit snapshots and frozen prompts.
- `checkpoints.sqlite`: coordination records and native private role namespaces.
- `planner/drafts/`: immutable content-addressed domain drafts referenced by Planner state.
- `artifacts/<request_id>/`: submitted Task Plans, requirements and frozen Dataset contents.
- `coding/<workstream_id>/`: SDK state and workspace, when enabled.
- `events.jsonl`: correlated model calls, tool results and formal handoffs; not recovery state.

Resume uses the saved configuration/prompts and checkpoint-selected artifacts; it does not reopen
the original Dataset or discover a new model. Credentials may rotate through the environment.
Changed source prompts affect new runs only. Historical run layouts are not migrated.

`[supervisor]` retains bounded directory/history/preview settings. `[planner]` controls
`context_chars`, `page_size`, `preview_chars` and `max_scenarios`.
Each role's `[...loop]` controls model/tool calls per invocation, context tokens, output reservation
and summary trigger/retention. These are invocation limits, not cumulative thread quotas.
Actual requests are checked including tools, current facts and history. Summaries are working context;
task and evidence facts are re-read from the current immutable products.

Task Plan publication validates Scenario coverage, task feasibility and evidence links. Accepting
derived data is a recorded Planner judgment plus deterministic identity/scope checks. Model completion
alone cannot replace publication, satisfy a requirement, or establish benchmark delivery.
New requests clear prior final responses and submitted IDs while retaining private history.
Limit exhaustion returns failure, preserving valid publications. Unexpected runtime errors retain the checkpoint.

Dataset members cover records, optional metadata, assets and referenced source files. Changed bytes are
rejected rather than silently assigned a new accepted digest. The current FileDatasetReader performs
eager content verification and is intended for small local inputs, not large-dataset throughput.

See [architecture](docs/ARCHITECTURE.md), [agent boundaries](docs/agents/README.md),
[Case contracts](docs/DATA_LAYER.md), [task contracts](docs/TASK_DISCOVERY.md), and
[implementation status](docs/PLAN.md).

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
