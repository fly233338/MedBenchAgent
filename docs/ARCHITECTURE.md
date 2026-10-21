# MedD2B architecture

MedD2B uses LangGraph and OpenHands to compile one medical dataset into an annotation-grounded benchmark. Models interpret semantics and choose work; code validates contracts, resolves references, executes data operations and enforces publication gates.

## Ownership

| Owner | Canonical responsibility |
| --- | --- |
| Dataset | Accepted prepared revisions, source/content bindings |
| Compilation | One dataset binding, optional user constraints, Registry, Task Plan, allocation, review, candidate and delivery |
| Run and Agent checkpoints | Execution position, private context, pending questions, operation replay |
| Session | Messages and references to independent compilation runs |

A submission creates `experiments/<session>/compilations/<id>/`, with a configuration snapshot and BIR. A retry resumes that same compilation. Sessions do not own or overwrite a shared BIR. Batch submission queues independent compilations; source joining and multi-image tasks remain features within a dataset.

Inside each BIR, `dataset_binding.json`, `task_plan.json`, `generation.json` and `planning_extensions.json` have fixed locations alongside allocation, review, candidate and delivery documents. Compilation tools bind one dataset; planning and generation read that identity from BIR. Allocation counts and capacities are keyed directly by Leaf ID. Dataset candidate lists and the session batch queue remain collections.

Each BIR business document owns its atomic `_commit`, revision, content digest and dependency digests. The manifest fixes compilation and Registry identity. Reads reject changed dependencies. Checkpoints retain execution state, dependency-bound evidence caches and references, without parallel Plan, allocation or access documents. UI summaries are read from BIR. Internal schemas have stable, unversioned identifiers; old internal layouts and checkpoints have no migration or compatibility reader.

Dataset identity is fixed independently of the benchmark brief. Revising requirements invalidates planning and downstream work, while retaining the same binding. BIR initialization distinguishes stale derivatives that need rebuilding from corrupted content, which is rejected. Selection questions and their dependency digests are validated before committing a binding or extension choice.

## Supervision

Supervisor remains a model-driven tool loop. It can inspect current evidence/coverage, prepare sources, bind a dataset, delegate planning, request independent review, reopen specific Scenarios, allocate, generate, audit and publish.

Tool availability describes valid dependencies, not one compulsory sequence. A complete plan allows review and allocation; review can also diagnose a discovered draft. A current reviewed candidate allows publication. Review findings can return work to planning or, for explicit source concerns, preparation. Rework is bounded and does not relax user constraints.

A delegated agent's outcome is separate from the success of the delegation call. The top-level `agent_outcome` carries `complete`, `progress`, `needs_input`, `failed` or `blocked`, diagnostics, questions and artifact references. A successful tool call does not imply a completed child or compilation. Missing facts become scoped LangGraph interrupts; answers return to their owner. Preparation-only requests never enter benchmark planning.

## Boundaries

- `workflow/`: orchestration, business clarification/evidence adapters, scoped revision and user-facing business projections.
- `runtime/`: generic action execution, model transport, checkpoint/interrupt integration, operation logs, process adapters and presentation of events.
- `preparation/`: source freezing, stage sessions, candidate snapshots, deterministic acceptance and publication.
- `agents/`: evidence-grounded task design and independent semantic review.
- `planning/`: Registry, Resolver, BIR, dataset/extension binding and dataset-local allocation.
- `data/`: canonical annotations, assets, index, source semantics and evidence sampling.
- `generation/` and `audit/`: deterministic generation and candidate audit, followed by publication and portable export. These stages have no Agent-private sessions or forwarding Agent modules.
- `evaluation/` in the source package: read-only package loading and evaluation configuration; model inference is independent.

The generic harness receives domain callbacks; it does not import BIR, interpret Leaf evidence or recognize dataset-selection states.

## Preparation and planning

Preparation keeps Understand → Text → conditional Image → Package → Publish. Ordinary repair and clarification continue the same stage's OpenHands session. Candidate snapshots are immutable per round; a semantic mapping revision creates a new session epoch. Only host-accepted artifacts cross stage boundaries.

Host validation reports passed checks. A first failed candidate establishes the repair baseline; a changed candidate must advance those checks or resolve diagnostics without regression to reset consecutive no-progress failures. Editing bytes alone does not count. Clarification does not consume this failure allowance. Stage iteration and active execution-time budgets are cumulative across rounds and restored from checkpoints; user waiting time is excluded.

Task Planning keeps Axis → Scenario → Leaf. There is one canonical pending Scenario queue. Deterministic transitions directly invoke the dedicated semantic Axis/screening calls; Leaf exploration retains a tool loop. Context construction belongs to Planning, prioritizes required/query evidence, reports omissions and supports exact paged reads. Design and review use the same source-evidence interface in separate calls.

Planning requires the compilation BIR and its frozen Registry. Pure validators receive Registry explicitly. Prompts have one source in Registry; imports do not load a repository-default Registry. Tests construct real temporary BIR fixtures rather than enabling a second local-file planning mode.

Derived dataset facts are cached on disk by the verified binding digest, outside checkpoints. Planning read caches also bind requirements, extensions, Scenario and rework round. A blocked planning attempt records input and clarification references so changed inputs permit another attempt. Completed Scenario lists, copied review reports and generated Leaf prompt contexts are not parallel state.

`planning.context_chars` bounds the complete serialized model request, including system instructions, output schema, tool definitions, evidence, tool results and repair turns. Direct planning/review calls and the planning tool loop share this final transport check. Every retry is checked again. Required content is never silently truncated; an oversized request reports its size and configured limit so the query can be narrowed or the budget increased.

Allocation is dataset-local. Generation and Audit share `resolve_task_input`, frozen ordering and deterministic Resolver answers. Reference-only evidence stays out of model input. `eligible_count` counts SamplingUnits; structural Scenario capacity counts Cases.

`DatasetAccessReceipt` is a flat data-layer contract produced by Prepared indexing and consumed unchanged by binding, planning, generation and audit. It verifies dataset identity, file digests and actual profile counts. Benchmark items have one strict contract: `item_id`, `dataset_id`, `leaf_task_id`, ordered `input_images` and optional sampling metadata. Generation validates before writing; audit rejects old aliases or malformed records with explicit quarantine reasons. `generation.fetch_batch_size` controls source pagination.

## Local PDF retrieval

The PDF retrieval component retains embedding, Chroma retrieval and reranking. Cache identity binds selected document byte digests, chunking strategy/parameters and embedding endpoint/model. Content is checked before reuse, including repeated retrieval on one reader; obsolete cache layouts are not read. Paragraph chunking always consumes text even when overlap exceeds the preceding short paragraph.

This component is not exposed uniformly to all semantic model calls. Shared tool registration, autonomous retrieval loops for every model and OpenHands cross-process access remain deferred; no MCP service or additional communication layer is introduced.

## Publication and recovery

Audit writes an unpublished candidate and quarantine; it never writes a delivery completion marker. Publication requires a current approved semantic review and a valid candidate. Shortfalls/rejections require a limitation explanation. It writes the compilation report and a hash-bound completion marker before recording delivery in BIR.

Audit enforces the total per-image limit from the canonical brief; intake resolves its default from the frozen configuration. Publication recovery verifies and reuses a completed package without rewriting its files, then records any missing BIR receipt.

Default delivery references Prepared/source images. Export verifies those image digests, copies only model inputs and produces relative paths. The standalone package reader validates package files and images after relocation.

LangGraph resumes execution from checkpoints. BIR dependency invalidation drives necessary downstream work after a revision; accepted datasets and previous delivery directories remain intact. OpenHands persisted user events deduplicate host feedback across process restarts.

Details and verification responsibilities live in [Agent documents](agents/README.md).
