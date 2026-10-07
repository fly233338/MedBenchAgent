# Preparation validation

## Stage boundaries

The sole production graph is Understand → Prepare Text → Prepare Image → Package → Publish.
Each coding stage owns one request.json, prompt.md, result.json and work directory.
Frozen inputs, current outputs, logs and same-attempt conversation persistence live under work/.
LangGraph checkpoint owns stage progress. Each stage has two attempts, with one semantic mapping
revision cycle. Technical failures retry their stage; revised mapping invalidates accepted Text/Image.

Understand validates typed mapping.json and image_conversion_plan.json plus deterministic population
and source observations. Host replays inspect_source.py using authorized frozen source roots.
These measured observations are runtime work products, not a predicted image manifest.

Text validates Case schema/identity, source dispositions and image requests without requiring converted
images. PopulationOutcome uses case_ids and record_refs exclusively. Every source unit has exactly
one terminal disposition; retained references and final Cases reconcile in both directions.
Image consumes immutable Text inputs. Requests freeze source, asset, view, policy and target identity.
Shared target_path determines paths before execution. Case sharing of an identical image is allowed;
conflicting identities and paths are rejected. Image checks real files, source identities, digests,
mask-feature bindings and complete presentation coverage. Native untransformed JPEG/PNG bytes must
be preserved. Medical sources and derived displays must be real PNG, with explicit window/slice policy.

## Read-only checks and deterministic assembly

Worker and Host call the same stage validators. Validators never restore or overwrite candidates.
CLI reports are written separately from candidates. Host authorization, inspector replay and publication
remain Host responsibilities. Missing accepted files and changed digests are failures, not implicit repairs.

Package copies only canonical domain outputs and attaches measured mask features to the corresponding
Evidence. Reference masks retain source digest/geometry and do not enter model input. DatasetPublisher
filters rejected Cases and dependent assets/presentation/sampling/split records in staging, updates source
dispositions, and validates that final set before generating Profile/Card and advancing current atomically.
No retained source or Case reference may dangle. Unidentifiable relationships prevent publication.

## Publication and recovery

Published data consists of Cases, provenance, accepted mapping/conversion policy, Asset/Presentation
manifests, required sampling/split manifests, ordinary image files and discovery/commit metadata.
Scripts, population working files, prompts and detailed diagnostics remain in runtime.
Catalog and DatasetAccess retain verified immutable revision paths. Root projections are browse-only.
A failure before commit cannot replace current; repeated publication of the same inputs is idempotent.
Consumers never infer image order or medical semantics from filenames, numeric IDs or missing labels.

Tests cover orphan and plural relations, quarantine closure, worker graph retries/mapping revision,
projection tampering, real JPEG/PNG preservation, DICOM/NIfTI/NRRD/Series conversion, and the deterministic
prepare→catalog→planning→allocation→generation→finalization pipeline. External LLM/OpenHands services
and real Docker execution require separately available service configuration.
