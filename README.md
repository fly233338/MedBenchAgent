# MedD2B — Agentic Benchmark Compiler

MedBenchAgent provides five native LangChain agents inside one LangGraph Compilation:
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

For the main user interface, run `uv run --locked medd2b tui` (or just `uv run --locked medd2b`).
The single-column chat workbench creates a default experiment automatically. Describe the task,
select local source directories, confirm the goal/count, and start. It creates SourceManifest and Run
files for you. Quoted Windows paths with spaces are supported. Key files must belong to a selected source.
An existing canonical Dataset remains an advanced input option.

Use **设置** to discover/select models, check the Coding environment and select skills. Non-sensitive
settings apply to new runs; password inputs last only for the current process and never enter the
experiment configuration, chat history or activity events. Existing environment credentials continue
to work. Hand-written TOML and manifests are only needed for advanced CLI/library use.
The wheel bundles the same default configuration and Coding dependency pins. In an installed-wheel
session, experiment/output paths and `.env` are relative to the launch directory; source checkouts retain the project root.

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

Data Processing invokes `python -m medbenchagent.agents.data_processing.coding.worker` in that environment.
Supervisor and Coding implementations live under `agents`; Compilation owns coordination and assembly.
`[coding.skills].enabled` selects packaged `tabular-inspection`, `medical-image` and `mask-geometry`
skills (all enabled by default; `[]` disables them). Runs freeze their contents and resource digests;
OpenHands selects and invokes skills natively. Public/user/project skill auto-loading stays disabled.

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

`python -m medbenchagent.compilation` exposes the same CLI. Use `--goal-kind profile` with a SourceManifest for source
understanding, or `--goal-kind plan --plan-level scenarios|executable_leaves` for planning.
Only roles needed by the selected goal/input are configured. Canonical Scenario planning does not initialize
Coding; Profile and Plan runs do not initialize Generation, Review or Delivery.
Limited nonempty delivery is permitted by default and reports `partial`; `--no-limited-delivery`
requires full delivery. The original requested quantity is never reduced to hide a shortfall.
`goal_result` is succeeded, partial, blocked or failed, separately from running/paused/waiting/stopped.

```powershell
uv run --locked medd2b create research --config config/local.toml
uv run --locked medd2b run research --source-manifest .runtime/sources.json --sample-count 20
uv run --locked medd2b tui
uv run --locked medd2b register --dataset-id example --manifest .runtime/sources.json --kind source
uv run --locked medd2b batch research example --sample-count 20
uv run --locked medd2b export PATH_TO_DELIVERY NEW_DESTINATION
```

The TUI keeps a compact Agent activity table above the chat. Coding appears under Data Processing.
Select a row or **活动** to inspect recent tool arguments, commands and bounded output, filtered by role.
Activities update while work is still running; no reasoning transcript or estimated completion percentage
is shown. **历史** switches runs, **产物** shows readable summaries, verified paths and portable export.
Enter sends; Shift+Enter/Ctrl+J inserts a newline. Reading older messages does not force scrolling to the end.

**暂停** requests a pause after the current parent graph node: Coding may still be executing until then.
**继续** resumes the saved pending node. Questions appear in the chat and are answered in the same input.
Closing during execution requests the same safe pause. A run with an unfinished checkpoint and no execution
lock is shown as interrupted/recoverable, not falsely running. Prepared runs offer **开始** without recreating inputs;
runs already bound to Studio remain managed by their Studio thread. Experiment files contain UI messages/drafts
and run references; Checkpoint remains the execution authority. Finished runs cannot be resumed.

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
Primary images have distinct Evidence IDs. Case labels reference the image set; a Mask in a multi-image
Case stays unbound unless the source supplies an explicit association. Missing labels stay missing.

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

The BrEaST-shaped offline workflow test generates four tiny source Cases, retains a missing label and an
unknown multi-image Mask association, repairs a rejected Leaf, resumes a checkpoint and delivers three
of four requested items as `partial`. The Coding SDK contract test runs in `.venv-coding` when provisioned;
it exercises native Skill parsing and invocation without calling a model.

Standalone retrieval remains `LocalPdfRag(...).retrieve(query, limit=...)` under the `rag` extra.
It returns PDF/page/chunk locations, excerpts and retrieval scores. No Agent currently calls it.
Future Planner/Review integration must additionally bind document digests, retrieval configuration and
evidence identities to the Run; literature cannot replace dataset annotations. No generic Retriever layer is introduced.
