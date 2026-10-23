# MedD2B — Medical Data to Benchmark

## Independent Compilation

The new Compilation exposes five native LangChain agents: **Supervisor, Data Processing,
Task Planning, Generation and Review**. OpenHands Coding is an execution capability inside Data
Processing. Allocation, Audit and Delivery are deterministic domain operations.

This iteration makes those capabilities callable and connects their contracts. The outer LangGraph
still only coordinates named nodes, handoffs and checkpoints. Full cross-agent workflow prompts,
legacy workflow removal and the old CLI/TUI migration are deferred to the next iteration.
Available capabilities do not establish live-model coordination or medical-quality validation.

### Run the complete capability set

Configure the common `[llm]` in `config/local.toml`, using `config/local.toml.example` as a template.
Keep credentials in `MEDBENCH_LLM_API_KEY`; Coding may override fields in `[coding.llm]` and use
`MEDBENCH_CODING_LLM_API_KEY`. Its independent SDK environment must already be provisioned from
`config/coding-worker.requirements.txt`.

Copy `config/sources.example.json` to an ignored location such as `.runtime/sources.json`,
set the Dataset identity and actual raw directories, then run:

```powershell
uv sync --locked --python 3.12
uv run --locked python -m medbenchagent.compilation run --directory .runtime/full-capabilities --compilation-id full-capabilities --all-agents --source-manifest .runtime/sources.json --sample-count 20 --config config/local.toml
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/full-capabilities --compilation-id full-capabilities
```

Source roots resolve relative to the source manifest. Each compilation needs a new run directory.
An existing canonical input can instead use `--dataset-manifest PATH`.
`--goal` supplies the actual user request; quantity defaults to the configured
`benchmark.default_sample_count` unless `--sample-count` is supplied, and is frozen as a
`benchmark_request` artifact. Pass `--answer` on resume only for a pending user question.

`--all-agents` enables Supervisor/Planner/Data Processing/Generation/Review model implementations,
OpenHands and deterministic Delivery. Individual controls remain available:
`--supervisor`, `--planner`, `--coding`, `--data-processing llm`, `--generation llm`,
`--review llm`, and `--delivery`. New professional roles default to disabled.
When Data Processing is enabled, Coding is internal and is not independently delegated by Supervisor.

### Products and capabilities

- **Profile:** inspected source observations, coverage and unknowns; full canonical Cases are not required.
- **TaskPlan:** Axis/Scenario judgments, backend choices and data requirements can be published before Leaves.
- **TaskData:** backend-scoped Manifest, `cases.jsonl` and asset references; multiple Scenarios may share Cases.
- **Leaf:** semantic task plus ordered input fields, question templates, reference bindings and output definition.
- **Generation:** VQA (closed/open), report, grounding and detection assembly from prepared Cases.
- **Review/Audit:** independent semantic findings and deterministic checks tied to exact target content.
- **Delivery:** portable questions/resources, provenance documents, reports, counts and a completion manifest.

Generation does not impose a global single-image or PNG/JPEG restriction. Data Processing prepares
the needed representation; backends assemble declared fields and references without interpreting
raw medical directories. `BenchmarkItem.inputs` is an ordered collection of asset/text/value entries.
The former `input_images` field is no longer accepted; direct producers, audit and the independent
loader now use the single current contract.

A generated candidate is not a delivered benchmark. Empty candidates and blocking or stale reports
cannot be delivered. A nonempty limited delivery requires `allow_limited_delivery` in the Delivery
WorkRequest scope and an explanation in its objective/constraints; actual shortfalls remain visible.

### Minimal offline development

The existing Scripted demonstration remains available and uses synthetic data only:

```powershell
uv run --locked python -m medbenchagent.compilation run --directory .runtime/compilation-demo --compilation-id demo --pause-for-input
uv run --locked python -m medbenchagent.compilation resume --directory .runtime/compilation-demo --compilation-id demo --answer "Continue"
uv run --locked pytest tests/test_agent_capabilities.py -q
```

Only two capability tests were added for this iteration: one connects Profile, TaskData, Leaf and
generation operations (including multiple inputs), and one checks Review/Delivery binding and
independent package loading. Tests use scripted model/SDK stand-ins, not real services.

The existing BrEaST sample converter is still available when a canonical input is useful:

```powershell
uv run --locked python scripts/prepare_breast_example.py --workbook datasets/US/BrEaST/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx --image-root datasets/US/BrEaST --output .runtime/breast-input
```

### Settings, state and recovery

Only enabled agents' settings are loaded. New role limits use `[data_processing_agent]`,
`[generation_agent]` and `[review_agent]`, with invocation budgets in each `[...loop]`.
The existing Supervisor, Planner and Coding configuration sections retain their roles.
No automatic SDK installation, RAG preflight or generic session manager is added.

`run.json` freezes enabled modes, effective non-secret settings, quantity and actual prompt texts.
`checkpoints.sqlite` owns coordination and private native role state. Planner domain drafts and
formal products are immutable files; SDK histories/workspaces stay with OpenHands. Logs are diagnostic.
Resume uses frozen settings and bound products, not current source prompts or a rediscovered model.

Profile observations bind actual source-file content. TaskData is published as a self-contained
snapshot with verified member hashes and real coverage counts. Native tools expose explicit
pagination/truncation and keep model input fields separate from reference-only data.

See [architecture](docs/ARCHITECTURE.md), [agent capabilities](docs/agents/README.md),
[data contracts](docs/DATA_LAYER.md), [task contracts](docs/TASK_DISCOVERY.md), and
[implementation/validation status](docs/PLAN.md).

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
items = package["items"]  # ordered inputs resolve to verified absolute paths
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
