# MedD2B — Medical Data to Benchmark

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

The Python import package remains `medbenchagent`; the product and console command are MedD2B / `medd2b`. Shared defaults live in `config/default.toml`. Language defaults to English and the sample budget to 1000; both are configurable. Missing clinical scope is valid.

## Development

```powershell
uv run ruff check .
uv run pytest
```

See [architecture](docs/ARCHITECTURE.md), [Agent responsibilities](docs/agents/README.md), [data contracts](docs/DATA_LAYER.md), and [preparation validation](docs/PREPARATION_VALIDATION.md). Offline tests use real small image fixtures and scripted model decisions. They do not establish live-model quality, Docker reliability or large-dataset throughput.

[Apache License 2.0](LICENSE).
