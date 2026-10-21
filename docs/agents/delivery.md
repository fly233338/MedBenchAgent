# Candidate audit, publication and export

**Input:** generated items and their canonical allocation, Task Plan, access/manifests and independent semantic review.

**Audit output:** an unpublished candidate containing benchmark JSONL, benchmark manifest, audit report and quarantine, plus a BIR candidate receipt. No completion marker is created.

**Audit execution:** re-resolve item inputs, answers, template and references; enforce allocation and per-image limits; reject duplicates/invalid records; balance closed-label option positions. An empty candidate cannot become a delivery.

The audit entry lives in `audit.task_items` and processes the one bound dataset. The manifest contains its dataset ID, task map, allocation and content dependencies; it does not assemble multiple datasets or carry empty evaluation-protocol placeholders. `generation.delivery` owns publication and export.

**Publication:** require an approved current review and verified candidate dependencies. Shortfalls/rejections require a limitation reason. Add a readable compilation report and a completion marker binding every file, then record delivery in BIR. Previous deliveries live in separate content-identity directories.

**Export:** verify a published package and every model-input image, copy assets to a new destination, rewrite model-input paths as relative paths, and preserve benchmark identity. Source/provenance metadata remains explanatory; loading the exported package requires no checkpoint, source tree or Prepared index.

**Consumer:** `medbenchagent.evaluation.loader.load_benchmark_package` verifies package files, item counts, task references, ordered input-image digests and portable path containment.

**Recovery and acceptance:** no publication on stale dependencies or blocking findings; interrupted output is not complete until its marker exists. Tests relocate exports, remove construction/data directories and load successfully, and reject image mutation.
