# MedBenchAgent

MedBenchAgent plans medical VLM benchmark generation from a natural-language brief,
tabular datasets, image assets, and PDF medical references.  The first release stops
after an approved `GenerationRequest`; it does not generate benchmark items.

## Development

Copy `.env.example` to `.env` and configure the OpenAI-compatible DeepSeek endpoint.
The local BGE embedding and reranker model paths must be configured before PDF RAG can
run.  Start the development server with:

```powershell
uv run langgraph dev
```

The graph entry point is `medbenchagent.workflow.graph:compile_benchmark_graph`.
