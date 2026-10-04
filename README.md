# MedBenchAgent

MedBenchAgent uses LangGraph to prepare medical datasets and build annotation-driven Medical VLM benchmarks. Submit one natural-language request; the Supervisor chooses the required agents and keeps the run resumable.

## Start

```powershell
uv sync
uv run medbenchagent
```

For DICOM, NIfTI or NRRD preparation:

```powershell
uv run --extra data-processing medbenchagent
```

Create an experiment and submit a request through the CLI:

```powershell
uv run medbenchagent create pilot
uv run medbenchagent run pilot "Prepare BrEaST from D:/Data/BrEaST and build a 20-question English benchmark"
uv run medbenchagent resume pilot RUN_ID
```

The TUI opens when no subcommand is supplied. Create an experiment, edit its configuration, then enter the conversation. `/details` shows runs and Agent activity; `/queue <request>` adds an independent request; `/resume` continues an interrupted run.

## Configuration

| Source | Purpose |
| --- | --- |
| `config/default.toml` | Shared defaults |
| `--config PATH` | Template merged when creating an experiment |
| `experiments/<name>/config.toml` | Configuration for future runs |
| `.runtime/<run>/control/effective_config.json` | Frozen settings for a run |

New experiments use `experiments/<name>/BIR/` as the canonical Benchmark Intermediate Representation. Runs keep only BIR references and derived views under `.runtime/`.

Relative paths resolve from the repository root. Keep credentials in environment variables; credentials are not written to configuration snapshots. Chat model discovery is required. Retrieval services are optional and their availability is recorded per run.

## Run layout

Each run contains:

```text
<run>/
  control/     frozen config, contract, artifact index, events and operation ledger
  inputs/      frozen Catalog and dataset Cards
  agents/      Agent state and semantic outputs
  workers/     OpenHands preparation workspaces
```

Prepared data remains under `datasets/Prepared`. Deliveries reference prepared images instead of copying them.

## Data preparation and construction

The Supervisor can inspect registered datasets, prepare configured raw sources, reuse valid prepared versions, plan and review tasks independently for each selected dataset, allocate one global sample budget, generate dataset-local JSONL items and publish one audited delivery. Missing facts become a focused `ask_user` question inside the same run. Requests outside dataset preparation or benchmark construction end without creating artifacts.

Raw-source preparation uses an OpenHands coding session. The source is read-only; candidates are validated deterministically before registration. Registered datasets are reused unless the request explicitly asks for a rebuild.

## Development

```powershell
uv run ruff check .
uv run pytest
```

The tests use local fixtures and model doubles. Real Docker, model services and large datasets require separate integration validation.

See [architecture](docs/ARCHITECTURE.md), [data contracts](docs/DATA_LAYER.md), [preparation validation](docs/PREPARATION_VALIDATION.md) and [evaluation](docs/EVALUATION.md).

## License

[Apache License 2.0](LICENSE).

