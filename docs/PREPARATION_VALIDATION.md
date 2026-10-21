# Preparation contracts

Understand produces mapping.json (image rules, optional exact overrides, record_sources and annotation semantics) and typed image_conversion_plan.json. Image rules are expanded by code from the frozen source inventory, with series directories derived from file paths; source roots are not rescanned and models do not enumerate images. Worker checks read the same inventory, mounted read-only in Docker.

Text writes final locators into Cases. Ordinary images reference original absolute source files without copies. Converted locators use Prepared/<dataset>/images/<source-policy-slice identity>.png. The shared resolver separates mounted read paths from host locators. Text writes runtime work/accounting.json; Host independently counts declared records and checks seen/consumed/excluded/skipped/emitted totals, allowing joins and splits.

Image is skipped when the validated cases.jsonl references no image locator with a convert rule, even if unused conversion rules exist. Otherwise it emits only required converted PNG candidates and image_manifest.jsonl. Package validates candidates and atomically installs fixed images without overwriting conflicting content. Immutable revisions contain only cases.jsonl, metadata.json and image_manifest.jsonl; metadata binds direct source contents, while the manifest binds converted contents. No images are copied into revisions. Source files must remain available and unchanged.

Workers and Host use the same read-only stage checks. Package performs final consistency checks; Publisher commits unchanged validated metadata. Runtime owns scripts, accounting, logs and semantic plans. Existing runs are not migrated.

Workers generate candidate files and finish. No success submission receipt is required. The runtime checks required outputs; Host independently validates contents and advances only on acceptance, retrying invalid candidates. `validate` is an optional read-only development check, not a completion action. Only exceptional outcomes use `work/control.json` with `StageControl`: `needs_input` contains scoped questions and `blocked` contains diagnostics. CLI `signal --outcome needs_input --questions-file work/questions.json` or `signal --outcome blocked --summary ...` writes this control message. Explicit control takes precedence over complete files; malformed control and missing outputs fail closed. Runtime errors and cancellation remain failures, not implicit success. Understand defaults to 80 iterations with a 1800-second timeout; Text and Image retain 80.

Worker prompts require only construction checks and prohibit CLI `validate`; workers finish as soon as required outputs are ready. The coding-process adapter collects files and reads exceptional control using the standard library, without importing the main application. Host separately applies the complete StageControl schema and authoritative stage validation.

## Sessions, feedback and snapshots

Each stage and mapping-revision epoch owns a stable `sessions/<stage>-<epoch>/` workspace and OpenHands conversation. Ordinary implementation repair and user clarification append a host message to that conversation. Host-turn markers are checked against SDK-persisted events to avoid resending after a crash.

Each round snapshots candidate files and accepted inputs under `attempts/<stage>/<attempt>/`; conversation and log histories are not copied. The host validates those snapshots, not mutable workspace files. Previous-reference digest checks cover semantic inputs and candidate outputs. A semantic mapping revision creates a new session epoch and invalidates dependent Text/Image work.

Failure counters are independent of round counters. The first rejected candidate establishes a baseline; subsequent changed candidates count as progress only when Host-passed checks advance or remaining diagnostics shrink without regression. Progress resets consecutive no-progress failures. Rewrites alone and returning to an already achieved validation result do not reset them. Clarification does not consume a failure retry. Environment failures and explicit blocked controls stop; mapping revision has a separate finite budget. Stage iteration and active execution-time use is persisted and cumulative across rounds, including mapping revisions; waiting for user answers is excluded. OpenHands `LLMSummarizingCondenser` is configured from stage budgets, with model context/output reserves honored when supplied. Accepted facts and user decisions remain explicit in host prompts/files instead of relying on summaries alone.

See [Data Preparation Agent](agents/data-preparation.md) for ownership and acceptance coverage.

## Real-runtime validation checklist — pending

Offline architecture cleanup verification on 2026-10-21, against `ba510e4`: `uv run ruff check .` passed; `uv run pytest -q` passed 315 tests. Obsolete interface tests were removed and current BIR, preparation-only resume, cache-integrity and extension-dependency coverage was updated. The four existing warnings concern LangSmith/nrrd deprecations and two Pydantic fixture serialization warnings.

The cleanup is accepted through offline Ruff/pytest checks. Live model behavior, Docker execution and large-dataset performance have **not** been validated by those checks. Record each run separately with commit, frozen configuration, model IDs, dataset revision, run/compilation IDs, commands, result and unresolved issues. Keep source data, credentials and generated packages out of Git.

1. **Real model, small data:** copy `config/local.toml.example`, configure an available Chat endpoint and a registered small dataset (or a raw source), then run the commands below. Check preparation-only completion separately from full compilation; inspect independent review, a scoped repair, publication and limitations. Interrupt a running compilation, resume its `RUN_ID`, and confirm accepted work is reused. Export its delivery, move the export, and load it with the existing package reader.

   ```powershell
   uv run medd2b --config config/local.toml create cleanup-model
   uv run medd2b run cleanup-model "Compile 10 English questions" --dataset DATASET_ID
   uv run medd2b resume cleanup-model RUN_ID
   uv run medd2b export PUBLISHED_DELIVERY NEW_EXPORT_DIRECTORY
   ```

2. **Docker:** with a Linux Docker daemon available, build `docker build -f config/coding-worker.Dockerfile -t medbench-openhands:local .`. Set `data_processing.backend = "docker"` in a fresh configuration and create a new experiment. Repeat the raw-source preparation and resume checks. Verify read-only source mounts, container-to-model connectivity, candidate/host path translation and enforced stage budgets. Current cleanup environment has no Docker command on PATH; this check remains pending.
3. **Large data and independent batches:** use existing authorized datasets and a new experiment. Run `uv run medd2b batch cleanup-large DATASET_A DATASET_B --request "Compile 1000 English questions"`. Record source size, wall time, peak memory, model usage, output counts and separate BIR/delivery paths. Verify each dataset has its own quota; test interruption/resume. Pagination tests do not establish bounded total memory or production throughput.
4. **Real RAG services:** configure embedding and reranker endpoints and use `LocalPdfRag` on a small selected PDF set. Retrieve, replace a selected PDF at the same path, and retrieve again with the same reader. Confirm a new cache/index and current passages; changing an unselected PDF must leave the selected cache intact. This validates the standalone component, not a shared retrieval loop across models.
