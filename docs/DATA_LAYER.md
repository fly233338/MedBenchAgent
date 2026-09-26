# Data layer contracts

The runtime separates source interpretation from deterministic execution. User-facing orchestration remains
in LangGraph; the data modules do not implement another workflow scheduler.

## Preparation

A run prepares all sources of ONE logical dataset. Sources retain `id`, nullable declared `version`,
`kind` (`raw` or `annotation`), `root`, and `annotation`. Artifact identity and used-file contents are distinct
from the user-declared source version. No checkpoint migration is performed (`agent-runtime.v15`).

The source-scoped chain is:

`Source → SourceFingerprint → SemanticPreparationSpec → ExecutionPlan → CandidateExtraction →
PopulationDecision / PreparationValidation → Annotation → ImageInventory / ImagePlan → NormalizedSource`.

- `save_source_fingerprint(inspection_references)` freezes selected real inspections, coverage and
  denominators, source inventory and the current used-file index. Subsequent unrelated reads do not invalidate it.
- `save_semantic_spec(spec)` saves `semantic-preparation.v1`. Required decisions cover population,
  all four identities, missingness, finding pairing and spatial conventions. Relationships, field mappings
  and scalar entity invariants are explicit. `expected_images` is optional source evidence, not a script count.
- Each decision has `meaning`, `inspection_references`, `interpretation` (preserve/interpreted), and
  `semantic_evidence` (source=document/user, reference, exact quote). Interpreted decisions require evidence.
  Population decisions for raw sources always require interpretation evidence. Facts must be in the fingerprint.
- Each field decision adds `target`, `source_path`, `source_field`, `transformation`
  (preserve/value_mapping/custom), optional scalar `mapping`, and `unmapped` (error/preserve).
  Targets use canonical dotted paths, with `[]` for array objects, e.g. `image.annotation.findings[].label`.
  Source fields are table column names or JSON Pointers; source_sheet/source_delimiter bind table reader options.
  Direct bindings are independently compared;
  custom transformations require evidence and are explicitly outside that semantic proof.
- Raw specs must declare image ID/path mappings. Every non-null emitted field needs a decision;
  absent patient/case/study IDs remain null. Namespace assignment belongs to trusted normalization.
- `save_dataset_adapter(code)` saves Python and returns its artifact reference. It requires a Spec,
  not an unsupported-operator reason. `save_execution_plan(plan)` binds the current Spec, Fingerprint,
  implementation and runtime. Example built-in request:

```json
{
  "executor": "built_in",
  "steps": [
    {"op": "read", "path": "labels.csv", "output": "labels"},
    {"op": "emit", "input": "labels", "fields": {
      "image.image_id": "image_id",
      "image.image_path": "filename",
      "image.annotation.diagnosis": "diagnosis"
    }}
  ]
}
```

Generated requests use `{"executor":"generated_python","adapter_reference":"<current artifact id>"}`
and no steps. `PreparationPlan` is a low-level operator-call type, not an Agent planning or publication interface.

### Dataset Adapter I/O

Scripts read `MEDBENCH_SOURCE_ROOT`, `MEDBENCH_SPEC_PATH`, `MEDBENCH_INPUTS_PATH`, and write only candidates
under `MEDBENCH_WORK_DIR`. Input instructions include the provenance JSON schema. Docker mounts spec,
inputs, source and code read-only. Local execution follows the same contract without filesystem isolation.
Do not assign final provenance, namespace IDs, normalize images, update Catalog or write completion markers.

`annotation.jsonl` contains `{record, record_ref}`. `provenance.jsonl` contains:

```json
{
  "schema_version": "extraction-provenance.v1",
  "id": "record:1",
  "step": "extract",
  "locators": [{"kind": "table", "path": "labels.csv", "row": 1}],
  "findings": []
}
```

Each locator uses `source-locator.v1` (default):

| Kind | Address |
| --- | --- |
| table | path, one-based logical data row, optional sheet and delimiter; CSV text line numbers are not row numbers |
| json | path and JSON Pointer; root Pointer is empty, object keys use `~0`/`~1` escaping |
| jsonl | path, one-based record number, optional Pointer within the record |
| mask | path and foreground integer labels; optional instance must exist and belong to foreground |

Every annotation array object also needs a `findings` entry with its concrete output JSON Pointer and
source locators. `group_findings` supplies these automatically. Masks must be two-dimensional and agree
with the referenced image geometry; empty foreground results are allowed without an asserted instance.
Parsers use the selected backend and disk-backed row indexes. JSON documents over 64 MiB must be exported
to JSONL; JSONL scans are streaming. Unknown binary annotation formats require a trusted reader first.

### Population and admission

`inspect_image_population` supports relational selection/join; `inspect_json_structure` inspects object
shape and `inspect_json_population` enumerates an explicit collection with direct ID/path fields and optional
scalar equality filtering. Both produce full image ID/path manifests before extraction. Preview counts do
not substitute for the population. The framework reconciles every candidate identity before quarantine.

`inspect_population_capability` recognizes aggregate_filter, nested_array_join and derived_identity as
outside the fixed population inspectors. A real structure inspection and a source-grounded Spec must
justify needing that operation. The gap cannot be substituted for an execution failure or missing evidence.
These custom selections use generated Python and `population_mode=human_review`; the script must also
emit full `population.jsonl` rows with source_id/image_id/image_path. `read_candidate_population` pages it.

`execute_extraction` runs twice in independent output directories and publishes CandidateExtraction only.
`validate_preparation` verifies source versions, output content, locator existence, direct mappings, finding
associations and complete identity correspondence. Human-review candidates pass technical checks before
an issue is created. LangGraph routes the natural-language answer back to the preparation agent;
approve/reject/clarify parsing binds to Source, Fingerprint, Spec, ExecutionPlan and CandidateExtraction.
Rejection requires revision, clarification requires a clearer answer, and changed inputs invalidate approval.

Only accepted candidates become Annotation. Assurance is `independently_reconciled`,
`input_annotation_only`, or `human_reviewed_selection_completeness_unproven`. These do not assert that
arbitrary Python transformations or medical interpretations are independently proven. Canonical/image QA
and entity invariant checks remain in trusted normalization/publication.

`kind=annotation` executes a fixed identity import automatically in `select_source`, generating its
structural Spec, candidates and validation using this same chain. Existing image manifests still locate
images. This proves completeness only against the supplied annotation. Semantic recoding uses a raw source.

## Normalization, quality and identity

Published v3 records add provenance with source ID/version, original IDs/path, record reference and plan
reference. ID namespaces encode the source/version/entity/raw-ID tuple; null entity IDs remain null.
Annotation arrays retain their entity boundaries. Provenance is not an answer field.

Only referenced images are processed. Raster/DICOM/NIfTI/ITK readers expose metadata separately from
pixel reads. PNG/JPEG bytes are copied; other 2D inputs use lossless PNG representation or a documented
display window. Missing decoder dependencies block execution. Unsupported dimensionality is reported
as an unusable record; no slice/region is chosen implicitly.

Quarantine records include code, stage, source, locator, image ID, severity, action and message. Invalid
findings quarantine the whole image record; dropping one finding could falsely create a unique answer.
Multiple valid findings and missing optional fields are retained. Repeated entity values are reported;
only fields explicitly declared invariant block on inconsistency.

Exact file and pixel hashes are retained. Pixel hashes include geometry, dtype, channel/mode and palette
information before display windowing. Duplicate pixels do not establish patient identity. Conflicting
annotation/identity groups are quarantined; equivalent records use a stable representative and retain all
provenance in `dedup.jsonl`. No perceptual or cross-library identity inference is performed.

## Harmonization and publication

A `harmonization.v2` plan contains source-specific field decisions with `field`, `action`, `target`,
optional scalar `mapping`, optional numeric `scale`/`offset`, `reason`, `inspection_references` and `semantic_evidence`. Decisions are unique per
source field. `merge` requires a target with the same entity scope and array depth. `keep_separate` and
`exclude` do not accept value transformations. Missing fields or vocabulary entries are rejected.

With multiple sources, unaligned fields remain under `<entity>.annotation.sources.<source_id>`.
An empty plan on one source is identity. Missing/not-applicable/unknown/negative labels are never assumed
equivalent; mappings must explicitly establish the desired distinction.

Normalized folders live beneath the bundle root. The merged manifest references their images directly.
Only accepted representative records enter the published annotation and cache. Reports account for
input = accepted + quarantined + duplicates_removed. Explicit upstream filtering is separately recorded
in operator reports. `ready.json` is written after final indexing and input verification.

Sources are treated as read-only inputs, not frozen snapshots; Docker enforces read-only mounts, local code does not. File listings detect population/stat changes and
the disk-backed used-file index checks content digests. Source changes reject reuse. Completed plans and
outputs are immutable; new plans create new artifacts. Receipts pin the backend: Docker records image ID; local Python records interpreter and related library versions.
Extractors execute twice with independent outputs; standard fixed PNG/JPEG imports also compare two
normalizations. Repeat directories are removed only after comparison.


New bundles archive preparation artifacts under `preparation/<source>/<operation>/`: Fingerprint, Spec,
execution plan, adapter when used, complete population, upstream provenance, validation, runtime, frozen
inspection/user receipts and source inventory. `preparation/index.json` binds their contents and assurance;
ready/access includes preparation_path/preparation_digest and admission/reuse verifies every bound file.
Final plan_ref is bundle-relative and points to execution_plan.json, which links spec.json. Older bundles
are not patched and retain legacy_unknown preparation assurance. Canonical annotation, image manifest,
Profile/Card and Catalog interfaces are unchanged.

Execution receipts and validation receipts are separate. Reuse verifies bound outputs; an incomplete
attempt directory is retained and replaced by a clean attempt. Runtime errors never downgrade the backend.
Quantity accounting is candidate images = retained + whole-image quarantine + equivalent duplicates removed;
upstream selection is separate and human review never upgrades its independent completeness guarantee.

## Queries and eligibility

`DatasetQuery(access_manifest)` binds every read to one cache revision. The database stores canonical
records, scalar occurrences with concrete ancestor-array indices, and array lengths. Large record sets
are scanned in batches; Python does not build a complete records list to compute the profile.

| Query | Result |
| --- | --- |
| `values(field, offset, limit, source_id)` | Exact values/counts, total and next offset; preview is never the vocabulary |
| `by_source(field)` | Coverage and distinct values per selected source, including zero coverage |
| `sample(offset, limit, source_id)` | Bounded canonical records without raw provenance |
| `pair(left, right)` | Same-image coverage versus same-entity pairs |
| `array(path)` | Array occurrence length distribution, zero/one/many counts |
| `capacity(protocol)` | Exact eligible count, exclusion reasons and per-source capacity |

The planner tools expose these operations. `resolve_eligibility` wraps the shared deterministic answer
resolver; `resolve_answer` remains the Generation/Audit entry point. Protocol validation verifies mappings
against observed values in the same record stream. Co-occurrence statistics are evidence for planning,
not a second implementation of answer eligibility.

Profile v4 carries image/entity/source summary and field/structure indexes. Coverage denominators are
explicit; unknown IDs do not collapse into a single entity. The current generation contract still requires
one unique answer witness per task/image. Per-leaf capacity does not solve global cross-leaf allocation.

## Verification boundary

Offline tests exercise typed operations, provenance, multi-source publication, quarantine, duplicates,
array witnesses, complete vocabulary, capacity, source changes and replay. Docker command construction
is tested with a subprocess double. No real dataset run, model call, container execution or large-scale
throughput validation is implied by these tests.

## Backend, Cards and registry

`data_processing.backend` is auto/docker/local. Auto selects once at preparation initialization, after bounded
Docker CLI/daemon/image/startup probes. Business errors never switch backend. Local runs fixed operations and
generated Python in a subprocess with timeout and process-tree cleanup; it has current-user permissions.
Generated Python requires Docker or explicit local selection; auto-to-local may run fixed operations only. The runtime
records both requested_backend and resolved backend. Switching requires a new run.
Install local dependencies explicitly using `uv sync --extra data-processing`. Both backends use environment
paths MEDBENCH_SOURCE_ROOT, MEDBENCH_WORK_DIR, MEDBENCH_ANNOTATION_PATH and MEDBENCH_DATA_SCRATCH.

A bundle version lives under prepared_datasets_root/<dataset_id>/<version>. Card v2 copies deterministic
Profile facts and adds dataset-intrinsic descriptions, domain and field roles with validated path references.
It binds annotation, Profile and image manifest content, never Blueprint. Preparation metadata is separate:
verified verbatim source quotations drive extraction of dataset_name/dataset_description/source_description,
saved with their references in dataset_metadata.json. Card generation sees only that metadata and Profile;
benchmark requirements never enter either metadata extraction or Card generation.

`refresh-card --dataset-id <id>` creates a new cards/<uuid>.json revision and then updates the catalog pointer.
Data, cache/Profile and old Cards are unchanged. v1 migration discards all previous semantic descriptions and
capability hints; absent reliable source metadata, it uses Profile only. Register requires v2 and validates
facts before changing the catalog. Entries hold dataset_id/name/description/available and
annotation_path/image_manifest_path/profile_path/dataset_card_path.

Each Selection operation freezes its name-resolution input, requirement, candidate order, every model-visible
Card summary/digest, and excluded-candidate reasons. candidate_cards/ retains full immutable Card copies;
dataset_candidates.json is published as DatasetCandidates and is a direct dependency of DatasetSelection.
dataset_selection.json includes the frozen decision inputs as well as selection method and output. Retries
reuse these inputs; external changes to rejected Cards do not invalidate an already running selection.

Selection and subsequent task planning reuse this cache. Core Concepts are shared across datasets; Leaf and
Extension identities are local. Planning publishes eligible_count without quotas. Global Allocation computes
Concept/Dataset/Leaf counts independently of the plans. Only final benchmark samples are merged; source
annotations and image manifests remain independent. Finalization performs dataset-local checks and publishes
one global audit report with the delivery.
