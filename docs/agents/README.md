# Agent and stage documentation

| Responsibility | Document |
| --- | --- |
| Supervisor | [Supervisor](supervisor.md) |
| Data Preparation | [Data Preparation](data-preparation.md) |
| Task Planning | [Task Planning](task-planning.md) |
| Independent semantic review | [Review](review.md) |
| Allocation | [Allocation](allocation.md) |
| Generation | [Generation](generation.md) |
| Candidate audit, publication and export | [Delivery](delivery.md) |

Each document records inputs, outputs, private state, model-visible evidence, feedback, recovery and acceptance. Schemas and registered tool signatures remain the authority for fields. Update the owning document when changing a contract or handoff; keep architecture-level boundaries in the architecture overview rather than duplicating every field.

Deterministic stages are tools, not additional autonomous agents. The existing Python package name is an implementation namespace; product terminology is MedD2B.
