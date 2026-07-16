# MedBenchAgent

MedBenchAgent plans medical VLM benchmark generation from a natural-language brief,
image-level `annotation.jsonl` data, and PDF medical references. The first release
stops after an approved `GenerationRequest`; it does not generate benchmark items.

## Dataset input

The workflow accepts only an `annotation.jsonl` file. Each line represents one image
and must contain `image_id`, `image_path`, `patient_id`, `case_id`, `study_id`, and an
object-valued `annotation`. Image paths are resolved relative to the JSONL file.
The versioned output contract is at
`src/medbenchagent/schemas/annotation.v1.schema.json`.

Validated data is cached by JSONL content hash under `benchmarks/dataset_cache/` as a
DuckDB database with one `images` row per JSONL line and a deterministic profile. CSV
and XLSX inputs are not supported.

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
