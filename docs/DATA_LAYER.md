# Data layer contracts

The runtime separates source interpretation from deterministic execution. User-facing orchestration remains
in LangGraph; the data modules do not implement another workflow scheduler.

## Preparation

`prepare_data(sources, dataset_id, feedback)` selects all sources of ONE logical dataset at once. Independent
datasets must be prepared separately. The CLI exposes prepare --dataset-id and register --dataset-id --bundle. A Source has `id`, nullable
`version`, `kind` (`raw` or `annotation`), `root`, and optional `annotation` (default `annotation.jsonl`).
Source IDs are unique within a selection. Each source has scoped artifact dependencies, so processing one
source cannot invalidate another by overwriting a global Mapping pointer.

The agent selects one source, inspects evidence, saves a PreparationPlan and executes it. A minimal plan:

```json
{
  "schema_version": "preparation.v1",
  "mapping": {
    "authoritative_population": "one row per image in labels.csv",
    "expected_images": 100,
    "entity_meanings": {"image": "one acquisition image"},
    "evidence": ["README: annotation table definition"],
    "invariant_fields": []
  },
  "steps": [
    {"op":"read","path":"labels.csv","output":"labels"},
    {"op":"emit","input":"labels","fields":{
      "image.image_id":"image_id",
      "image.image_path":"filename",
      "image.annotation.diagnosis":"diagnosis"
    }}
  ]
}
```

The mapping's expected count is the number of authoritative image records emitted before image quarantine.
Read/select/filter/rename/cast/value_mapping/missing_normalization/join/group_findings/namespace_ids/bbox/
mask_to_bbox/emit are the supported operations. Each nonterminal step has a unique output and references
only earlier inputs. Arbitrary SQL and executable expressions are not part of the contract.

- CSV/TSV preserve strings by default; explicitly cast non-ID columns.
- Join keys, cardinality, null keys and unmatched behavior are mandatory decisions. Overlapping columns
  require rename; accidental many-to-many expansion is rejected.
- `group_findings` groups by image keys and builds objects from same-row fields. It retains source row
  locators, including joined/grouped rows, in stable order.
- `bbox` supports xyxy, xywh, cxcywh and normalized xyxy. Normalized coordinates require dimension columns.
- `mask_to_bbox` requires foreground labels, `single` or `by_value` instance semantics, and dimension
  columns. Mask/image sizes must agree. Empty masks produce no box; no resize is performed.
- `emit.fields` maps canonical paths to columns. Optional IDs/modality/anatomy default to null. Missing
  image IDs use stable source locators. Do not use this mechanism to invent patient/case/study identities.

Every emitted intermediate row contains `{record, record_ref}`. `provenance.jsonl` contains
`{id, locators, step}`; each locator is `relative-file:positive-row` or
`relative-file#sheet=sheet-name:positive-row`. IDs link emitted records to this index. Fallback plans have
`fallback_reason` and no operator steps; their Python must satisfy the identical contract. Missing/reused
references and nonexistent source files are rejected. A file's semantic meaning still requires evidence.

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

A `harmonization.v1` plan contains source-specific field decisions with `field`, `action`, `target`,
optional scalar `mapping`, optional numeric `scale`/`offset`, and `evidence`. Decisions are unique per
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
Raw operations execute twice with independent outputs; standard fixed PNG/JPEG imports also compare two
normalizations. Repeat directories are removed only after comparison.

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
generated fallback Python in a subprocess with timeout and process-tree cleanup; it has current-user permissions.
Install local dependencies explicitly using `uv sync --extra data-processing`. Both backends use environment
paths MEDBENCH_SOURCE_ROOT, MEDBENCH_WORK_DIR, MEDBENCH_ANNOTATION_PATH and MEDBENCH_DATA_SCRATCH.

A bundle version lives under prepared_datasets_root/<dataset_id>/<version>. Card v1 copies deterministic
Profile facts and adds model-produced semantics with validated path references. It binds annotation, Profile,
image manifest and Blueprint content. Register validates these facts before replacing a catalog entry.
Catalog holds dataset_id/name/description/available and annotation_path/image_manifest_path/profile_path/
dataset_card_path. Missing Card/Profile assets require prepare, never implicit whole-catalog preparation.

Selection and subsequent task planning reuse this cache. Core Concepts are shared across datasets; Leaf and
Extension identities are local. Planning publishes eligible_count without quotas. Global Allocation computes
Concept/Dataset/Leaf counts independently of the plans. Only final benchmark samples are merged; source
annotations, image manifests and local audits remain independent.
