# Generation

**Input:** canonical Task Plan and dataset-local allocation, frozen sampling/presentation manifests and seed.

**Output:** benchmark JSONL and a BIR output receipt binding file content, plan and allocation.

Items use only `item_id`, `dataset_id` and `leaf_task_id` for their respective identities. The strict shared model includes ordered `input_images` and sampling metadata; legacy aliases and undeclared fields are rejected at output/audit boundaries. Dataset binding receipts validate source identity and content before generation. `generation.fetch_batch_size` is the source pagination setting.

**Execution:** a deterministic tool. It resolves ordered model inputs and references with the shared Resolver, chooses equivalent templates using the seed, and records input image paths/digests, role bindings, options, answer and reference IDs.

**Private state:** no semantic conversation. Large source data remains in the dataset index; runtime logs describe progress.

**Feedback:** shortages are returned for audit/supervision. Missing or changed images are ineligible. Generation does not reinterpret labels or invent mappings.

**Recovery:** input dependencies and output digest are checked. Completed effects are replayable; a plan/allocation change requires regenerated output.

**Acceptance:** input ordering, answer reconstruction, reference-only separation, deterministic templates and canonical dependency invalidation. Relevant tests cover generation, task inputs, masks and end-to-end compilation.
