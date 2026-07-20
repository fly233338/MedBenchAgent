# MedBenchAgent

MedBenchAgent builds medical VLM benchmarks from a natural-language brief,
image-level `annotation.jsonl` data, and PDF medical references. The workflow screens
supported tasks, designs them with PDF RAG, and writes closed VQA, detection, finding
description, and report items into one mixed JSONL. It first writes an immutable,
unshuffled `benchmark_items.raw.jsonl`, then by default writes a separately finalized
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
review. Respond in natural language to confirm it or describe one task to revise;
DeepSeek maps every response to a validated confirm/revise action. BenchmarkInfo review
uses the same natural-language interpretation. Revisions regenerate only the selected
task. The working Blueprint is updated only after confirmation; the resulting
`task_plan_report.json` retains decisions for both retained and dropped tasks.

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

## Question design and benchmark outputs

Every retained task receives a PDF-grounded Question Design before generation. Closed
VQA designs define fixed options and an annotation-to-answer mapping; DeepSeek only
generates and independently reviews question wording. Detection designs bind a target
description and an `xyxy` box from the same annotation object, then instantiate an
approved template. Finding-description and report tasks use paired expert text directly
as the reference answer and never ask the model to synthesize ground truth.

The raw file keeps task/source-row order and original closed-task option order. Every
item records `task_type` and `response_format`; `answer` is a choice letter, reference
text, or an `xyxy` box. The final copy balances correct option positions inside each
closed task while detection and free-text items pass through unchanged.

Balance an existing raw file without modifying it:

```powershell
uv run medbenchagent balance-answers --input RAW --output COPY --seed 0
```
