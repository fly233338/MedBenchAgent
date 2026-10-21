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

Each BIR business document owns its atomic `_commit`, revision, content digest and dependency digests. The manifest fixes compilation and Registry identity. Reads reject changed dependencies. Checkpoint and UI copies never override BIR. Internal schemas have stable, unversioned identifiers.

## Supervision

Supervisor remains a model-driven tool loop. It can inspect current evidence/coverage, prepare sources, bind a dataset, delegate planning, request independent review, reopen specific Scenarios, allocate, generate, audit and publish.

Tool availability describes valid dependencies, not one compulsory sequence. A complete plan allows review and allocation; review can also diagnose a discovered draft. A current reviewed candidate allows publication. Review findings can return work to planning or, for explicit source concerns, preparation. Rework is bounded and does not relax user constraints.

A delegated agent's outcome is separate from the success of the delegation call. Supervisor receives the child's actual status and diagnostics. Missing facts become scoped LangGraph interrupts; answers return to their owner. Preparation-only requests never enter benchmark planning.

## Boundaries

- `workflow/`: orchestration, business clarification/evidence adapters, scoped revision and user-facing business projections.
- `runtime/`: generic action execution, model transport, checkpoint/interrupt integration, operation logs, process adapters and presentation of events.
- `preparation/`: source freezing, stage sessions, candidate snapshots, deterministic acceptance and publication.
- `agents/` and `planning/`: evidence-grounded task design and independent review; Registry, Resolver and BIR contracts.
- `data/`: canonical annotations, assets, index, source semantics and evidence sampling.
- `generation/` and `audit/`: deterministic materialization, candidate audit, publication and portable export.
- `evaluation/` in the source package: read-only package loading and evaluation configuration; model inference is independent.

The generic harness receives domain callbacks; it does not import BIR, interpret Leaf evidence or recognize dataset-selection states.

## Preparation and planning

Preparation keeps Understand → Text → conditional Image → Package → Publish. Ordinary repair and clarification continue the same stage's OpenHands session. Candidate snapshots are immutable per round; a semantic mapping revision creates a new session epoch. Only host-accepted artifacts cross stage boundaries.

Task Planning keeps Axis → Scenario → Leaf. There is one canonical pending Scenario queue. Deterministic transitions directly invoke the dedicated semantic Axis/screening calls; Leaf exploration retains a tool loop. Context construction belongs to Planning, prioritizes required/query evidence, reports omissions and supports exact paged reads. Design and review use the same source-evidence interface in separate calls.

Allocation is dataset-local. Generation and Audit share `resolve_task_input`, frozen ordering and deterministic Resolver answers. Reference-only evidence stays out of model input. `eligible_count` counts SamplingUnits; structural Scenario capacity counts Cases.

## Publication and recovery

Audit writes an unpublished candidate and quarantine; it never writes a delivery completion marker. Publication requires a current approved semantic review and a valid candidate. Shortfalls/rejections require a limitation explanation. It writes the compilation report and a hash-bound completion marker before recording delivery in BIR.

Default delivery references Prepared/source images. Export verifies those image digests, copies only model inputs and produces relative paths. The standalone package reader validates package files and images after relocation.

LangGraph resumes execution from checkpoints. BIR dependency invalidation drives necessary downstream work after a revision; accepted datasets and previous delivery directories remain intact. OpenHands persisted user events deduplicate host feedback across process restarts.

Details and verification responsibilities live in [Agent documents](agents/README.md).
