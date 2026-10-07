# Preparation: semantic decisions, data, and runtime state

## One owner per boundary

The graph runs Understand → Prepare Text → Prepare Image → Package → Publish.
Understand accepts only mapping.json and image_conversion_plan.json. `mapping.images` maps final
relative locators to source_id/source_path/source_format/policy_id; source references and policies
are checked. There is no host-generated target-path convention overriding these decisions.
Inspection scripts, measurements and population notes are optional runtime work, not handoff dependencies.

Text produces cases.jsonl. Asset Evidence.value is the final ordinary-image locator. Source mask
features are computed with mask_geometry and attached directly to reference-only Case Evidence.
Text checks schema, unique Case IDs, required mapped coverage, locators and mask source digests.
Detailed annotation accounting belongs in runtime and is not a published contract.
Optional sampling/split semantics are supplied in metadata.json; their existing typed contracts remain.

Image consumes accepted mapping, conversion plan and Cases. It writes images/ and image_manifest.jsonl.
Each ImageEntry contains the final path, measured format/dimensions/digest, source identity/digest and
necessary geometry metadata. Native untransformed JPEG/PNG bytes are preserved. Medical displays
require real PNG. Stage checks reject missing, extra, changed or inconsistent outputs without repairing them.

Package copies only referenced image files, Cases and their single manifest, projects needed field
semantics and sampling/split data into metadata.json, then performs one final cross-file check.
Publisher only verifies the validated content digest, copies to the destination filesystem and atomically
commits the directory. It performs no quarantine, Case rewrite, manifest repair, indexing or profiling.
Invalid candidates must be repaired by their producing stage; they are never silently changed during publish.

## Persisted file inventory

Published revision:

```
cases.jsonl
image_manifest.jsonl
metadata.json
images/<actual ordinary images>
```

Metadata is dataset content: dataset identity, field meanings/missingness needed for planning and optional
sampling/split semantics. Original mapping and conversion policy remain in preparation runtime.

Outside revisions, current.json and .commits/ record publication; .derived/ holds rebuildable DuckDB,
Profile, Card and access projections. No full current projection or second image copy is generated.
Deleting .derived/ cannot lose dataset content; catalog loading reconstructs it from immutable inputs.

An attempt has request.json (including context), prompt.md, result.json and work/. The only copied inputs
are accepted mapping, conversion plan and, for Image, Cases plus optional metadata. No context packet,
per-file schema bundle, source-report receipt, inspection script or population handoff is materialized.
work/submission.json carries worker control/questions and is not a dataset output. Checkpoint owns graph
progress. CLI schema inspection prints the canonical Python schemas without creating per-attempt files.

## Verification

Tests cover the actual graph with deterministic workers, image-stage retry, mapping revision, semantic
question resume, native image byte preservation, real medical decoding, immutable/idempotent publication,
minimal revision file sets, stale/corrupt file rejection, cache reconstruction and benchmark delivery.
External model/OpenHands services and Docker execution are verified separately when configured.
