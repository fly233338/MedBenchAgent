# MedD2B — Agentic Benchmark Compiler

MedBenchAgent uses five native LangChain agents inside one LangGraph Compilation:
Supervisor, Data Processing, Task Planning, Generation and Review. Data Processing owns
OpenHands Coding. Allocation, Audit and Delivery are deterministic operations.

The default collaboration is source → Profile → Scenario/backend requirements → TaskData
→ accepted Leaf definitions → generated candidate → Audit/Review → portable Delivery.
Supervisor chooses each delegation and handles local repair; the graph does not impose a global
business phase. Existing valid products can be reused. A stopped run is not automatically a
successful benchmark: the Host checks its explicit goal and delivery evidence.

## Install and configure

```powershell
uv sync --locked --python 3.12
Copy-Item config/local.toml.example config/local.toml
```

The default development group includes pytest, Ruff, Studio tooling and openpyxl for the BrEaST
example. Use `--no-dev` for the core environment. Standalone PDF retrieval and optional medical
readers have explicit extras:

```powershell
uv sync --locked --extra rag --extra data-processing
```

These extras belong to the main environment; Coding has its own separately provisioned dependencies.

Set model endpoints/identities in `config/local.toml` and credentials through `.env` or environment
variables (`.env.example`). Each new run freezes model settings, repair budgets and composed
prompts. Restore uses that snapshot; incompatible code/tool changes require a new run.

Provision the separate Coding environment before using Data Processing:

```powershell
uv venv .venv-coding --python 3.12
uv pip install --python .venv-coding/Scripts/python.exe -r config/coding-worker.requirements.txt
```

Data Processing invokes `python -m medbenchagent.agents.coding_worker` in that environment.
Supervisor and Coding implementations live under `agents`; Compilation owns coordination and assembly.

Use `config/sources.example.json` as the SourceManifest template. It contains a dataset identity
and named source directories; relative roots resolve against the manifest. Alternatively supply
an existing current canonical DatasetManifest. Production never creates synthetic input implicitly.

## Run, inspect and resume

```powershell
uv run --locked medd2b run --directory .runtime/example --compilation-id example --source-manifest .runtime/sources.json --goal-kind benchmark --sample-count 20 --config config/local.toml --goal "Compile a traceable benchmark and report actual coverage."
uv run --locked medd2b inspect --directory .runtime/example
uv run --locked medd2b resume --directory .runtime/example
# Only when the saved run is waiting for a user fact:
uv run --locked medd2b resume --directory .runtime/example --answer "The two studies are explicitly paired in the source mapping."
```

`python -m medbenchagent.compilation` exposes the same CLI. Use `--goal-kind profile` for source
understanding, or `--goal-kind plan --plan-level scenarios|executable_leaves` for planning.
Limited nonempty delivery is permitted by default and reports `partial`; `--no-limited-delivery`
requires full delivery. The original requested quantity is never reduced to hide a shortfall.
`goal_result` is succeeded, partial, blocked or failed, separately from running/waiting/stopped.

```powershell
uv run --locked medd2b create research --config config/local.toml
uv run --locked medd2b run research --source-manifest .runtime/sources.json --sample-count 20
uv run --locked medd2b tui
uv run --locked medd2b register --dataset-id example --manifest .runtime/sources.json --kind source
uv run --locked medd2b batch research example --sample-count 20
uv run --locked medd2b export PATH_TO_DELIVERY NEW_DESTINATION
```

The TUI retains experiment/run lists, editable configuration for new runs, goal/input submission,
progress, saved questions/resume and artifact paths. Stop waits for a graph boundary; it does not
pretend that arbitrary external Coding commands can be rolled back. Experiments index runs and UI
messages; the Compilation checkpoint remains the execution authority.

For Studio, freeze a dedicated run without executing its agents, then point LangGraph at it:

```powershell
uv run --locked medd2b run --prepare-only --directory .runtime/studio --compilation-id studio --source-manifest .runtime/sources.json --config config/local.toml --sample-count 20
$env:MEDBENCH_STUDIO_RUN = (Resolve-Path .runtime/studio).Path
uv run --locked langgraph dev
```

Submit an empty graph input in one Studio thread. Studio owns its checkpoints; use that thread for
Studio resume. A run directory is bound to one Studio thread. Do not interchange Studio persistence
with the CLI SQLite database. Both use the same graph, agents, policies and formal artifacts.

## Products and responsibilities

- **Profile:** bounded source observations, coverage, representative evidence and unknowns; samples may be empty.
- **Scenario TaskPlan:** Axis judgments, meaningful scenarios, selected output backends and concrete DataRequirements.
- **TaskData:** immutable prepared Cases/Assets with scenario membership, ordered fields, visibility and exact requirement bindings.
- **Executable TaskPlan:** Planner's acceptance plus Leaf templates, field/reference bindings, options and output definition.
- **GenerationSpec:** exact plan/Leaf/TaskData bindings and execution parameters; no duplicate task design.
- **Candidate:** actual items, failures, quarantine and unchanged requested counts.
- **Review/Audit/Delivery:** exact content bindings, semantic inspection coverage, deterministic checks and an independently readable package.

VQA (closed/open), report generation, grounding and detection assemble prepared fields deterministically.
There is no global single-image or PNG/JPEG input restriction. Raw-format conversion belongs to
Data Processing/Coding; multiple assets and text/value inputs retain their order. Source relationships,
clinical labels and missing answers cannot be invented by Generation.

## Small development examples

```powershell
# Explicit synthetic raw source, no model calls:
uv run --locked medd2b demo --directory .runtime/demo-source
# Reproducible canonical BrEaST subset; defaults to the first 20 source CaseIDs:
uv run --locked python scripts/prepare_breast_example.py --workbook datasets/US/BrEaST/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx --image-root datasets/US/BrEaST --output .runtime/breast-input
```

Keep real data and run outputs in ignored directories. BrEaST Case granularity is a lesion record;
unknown patient/study relationships stay unknown. Preparing an example does not establish medical
quality or autonomous model performance.

## Implementation and validation

The old Workflow/BIR/Resolver/Harness/Preparation execution chain and its private interfaces have
been removed. Useful asset readers/converters, geometry, source utilities and standalone retrieval
remain. Historical internal schemas and checkpoints are not supported through compatibility branches.

Development verification uses tiny local files, Fake Model calls through real `create_agent`, and a
persistent SDK substitute. It checks connected handoffs, local repair, checkpoint resume and package
reading. Install the extras explicitly when running the retained domain tests:

```powershell
uv run --locked --extra rag --extra data-processing pytest -q
uv run --locked ruff check .
```

Offline tests do not establish live-model autonomy, real Coding task success or complete medical
dataset acceptance. See [implementation record](docs/PLAN.md), [architecture](docs/ARCHITECTURE.md),
[agent operations](docs/agents/README.md), [data](docs/DATA_LAYER.md) and [planning](docs/TASK_DISCOVERY.md).
The package reader and external evaluation boundary are documented in [EVALUATION](docs/EVALUATION.md).
