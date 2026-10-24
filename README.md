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
The TUI opens a searchable Run list. **New** creates one Run for one Dataset and one Compilation;
**Clone** reuses configuration and source selection without copying execution state. Open a Run,
use **Task** to select source directories/key files and set the goal, count and language, then **Start**.
Quoted Windows paths with spaces and Chinese input are supported. No hand-written manifest is required.
**Dataset** registers reusable source references or selects an existing Dataset; **Unlink** removes only
the index entry. A Dataset can have multiple source directories. Changing its source definition creates
a new revision; unchanged selections preserve the identity. Revision tracks the registered definition,
not a recursive scan of changing source data.

**Config** selects models and the local Coding environment. Global defaults/local configuration are
copied into the Run, then frozen on first execution. Started Runs are read-only in configuration views;
clone to change the task, dataset or model. API keys stay in process memory/environment and never enter
Run configuration or diagnostic events. Keep sensitive input out of the goal if it need not be retained.
The wheel bundles defaults, dependency pins, prompts, skills and TUI styles. Installed-wheel paths are
relative to the launch directory; source checkouts use the project root. Configure both roots in [paths].

```text
runs/<name>/                   .runtime/runs/<compilation_id>/
  config.toml                    checkpoints.sqlite
  run.json                       events.jsonl
  inputs/                        locks/ + session.json
  artifacts/                     planner/ + resources/
                                 coding/<workstream_id>/
```

Run definitions and formal data live under runs/. Checkpoints, UI records, drafts, Coding workspaces
and SDK persistence live under .runtime/. Recovery needs the runtime directory; published derived
files do not. Original source datasets remain external references. Existing old runs are left untouched
and are not migrated or resumed through compatibility branches.

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
uv run --locked medd2b run example --source-dir datasets/example --goal-kind benchmark --sample-count 20 --config config/local.toml --goal "Compile a traceable benchmark and report actual coverage."
uv run --locked medd2b inspect --directory runs/example
uv run --locked medd2b resume --directory runs/example
# Only when the saved run is waiting for a user fact:
uv run --locked medd2b resume --directory runs/example --answer "The two studies are explicitly paired in the source mapping."
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
uv run --locked medd2b batch example --sample-count 20
uv run --locked medd2b export PATH_TO_DELIVERY NEW_DESTINATION
```

The TUI uses concise English labels with dark/light themes and keyboard-first navigation. Use
**Trace** for correlated model/tool/Coding activity and actual token usage (or an explicit unavailable
marker), and **Artifacts** for verified product details and export. Wide terminals show an activity
pane; compact terminals stack it. Rendering is designed for UTF-8 terminals at 80x24 or larger.
Below that size the workbench collapses and asks for resize. NO_COLOR is respected; TERM=dumb or
non-TTY CLI startup prints help. RTL shaping depends on the terminal; screen-reader support is limited.
The CLI inspect command provides structured output without the TUI.

Keys: Ctrl+N new, Ctrl+R runs, Ctrl+P commands, Tab completion, Enter send, Shift+Enter/Ctrl+J newline,
Esc close, Ctrl+C safe exit. Slash commands: /new, /runs, /dataset, /config, /pause, /resume, /events,
/artifacts, /export, /help. Reading history preserves scroll position. Switching Runs leaves active work
running; only one executor is started by each TUI instance.

**Pause** and quit wait for the current parent graph node to save its checkpoint; this can include a
Coding call. **Resume** continues the saved node, and pending questions use the same input. An unfinished
checkpoint without an execution lock appears as interrupted. Finished Runs cannot resume; use Clone.
Events are diagnostic, never a completion ledger. Local Coding runs with the user's filesystem
permissions: read-only source instructions and Host path/digest checks are not an OS sandbox.
No Docker backend, detached executor or monitoring service is introduced.

For Studio, freeze a dedicated run without executing its agents, then point LangGraph at it:

```powershell
uv run --locked medd2b run --prepare-only --directory runs/studio --source-manifest .runtime/sources.json --config config/local.toml --sample-count 20
$env:MEDBENCH_STUDIO_RUN = (Resolve-Path runs/studio).Path
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

TUI tests render home/workbench views at 80x24, 100x35 and 140x45 in dark/light themes into
.runtime/tui-review/. Inspect rendered SVGs as well as Pilot assertions after layout changes.
