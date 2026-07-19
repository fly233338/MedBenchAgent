# MedBenchAgent

MedBenchAgent plans medical VLM benchmark generation from a natural-language brief,
image-level `annotation.jsonl` data, and PDF medical references. The first release
creates closed VQA benchmark items after an approved Question Design. It writes an immutable,
unshuffled `benchmark_items.raw.jsonl`, then by default writes a separately balanced
`benchmark_items.jsonl` copy. Both files contain exactly the requested `sample_count`.

## Dataset input

The workflow accepts only an `annotation.jsonl` file. Each line represents one image
and must contain `image_id`, `image_path`, `patient_id`, `case_id`, `study_id`, and an
object-valued `annotation`. `image_path` is stored as opaque metadata: dataset
exploration does not resolve it or check whether an image file exists. Benchmark items
are identified by `image_id`; mapping IDs to image files is the responsibility of an
external module. The versioned input contract is at
`src/medbenchagent/schemas/annotation.v1.schema.json`.

Validated annotation text is cached by JSONL content hash and cache-format version
under `benchmarks/dataset_cache/` as a DuckDB database with one `images` row per JSONL
line and a deterministic `task-capability-profile.v1`. The profile describes annotation
structures and every scalar leaf field, but exposes raw values only for low-cardinality
categorical fields; identifiers, DICOM metadata, numeric values, and coordinates are
not included as examples. CSV and XLSX inputs are not supported.

## Planning review

The workflow computes the complete Task Plan in one model call, then presents it for
review. Confirm it with `{"action": "confirm"}`, or revise exactly one task with
`{"action": "revise", "task_id": "...", "feedback": "..."}`. Revisions regenerate
only the selected task. The working Blueprint is updated only after confirmation;
the resulting `task_plan_report.json` retains decisions for both retained and dropped
tasks.

## Development

Copy `.env.example` to `.env`. The default configuration targets the local vLLM
deployment at `http://127.0.0.1:8005/v1` with model `deepseek-v4-flash`, plus the
Qwen3 embedding and reranker models under `/public/home/rxa-jj-wzk/models/Qwen-RAG`.
`DEEPSEEK_API_KEY` may remain empty when vLLM authentication is disabled.

For Web VS Code, start the development server without browser launch and forward port
8123 through the IDE:

```bash
uv run langgraph dev --host 127.0.0.1 --port 8123 --no-browser --allow-blocking
```

The graph entry point is `medbenchagent.workflow.graph:compile_benchmark_graph`.

## Benchmark outputs

Only retained `close_vqa` tasks are generated. Their fixed option lists and answers are
resolved from annotations; DeepSeek generates and independently reviews only question wording.
The raw file keeps task/source-row order and the original option order. The final copy balances
the correct option positions inside each task while retaining all other item content.

Balance an existing raw file without modifying it:

```bash
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```
