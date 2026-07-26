---
name: medical-dataset-preparation
description: Prepare local medical imaging datasets for MedBenchAgent by producing canonical annotation.jsonl, deterministic 8-bit PNGs, exact image_manifest.jsonl mappings, and reproducible preparation artifacts. Use when raw annotations and source images must be normalized for benchmark generation.
---

# Medical Dataset Preparation

Keep source data read-only and local. Keep annotation facts separate from prepared pixels, and never infer medical facts from filenames or image appearance.

## Inputs

Obtain explicit paths for the annotation sources, source-image root, artifact directory, and processed-image directory. Keep the processed images inside the artifact directory so every manifest path is relative and non-escaping.

Use the project contract at `src/medbenchagent/schemas/annotation.v1.schema.json`.

## Workflow

### 1. Plan the annotation mapping

Inventory the supplied tables, documents, and image references. Determine the authoritative image population, record granularity, identifiers, joins, cardinalities, conflicts, missing rows, and stable ordering.

Write concise `source_inventory.json` and `conversion_plan.json` files containing observed counts and mapping decisions. Stop for clarification when identity, joins, grouping, or conflicts remain materially ambiguous.

### 2. Normalize annotations

Write `<artifact_dir>/scripts/prepare_annotations.py` with all dataset-specific logic. Require explicit input and output arguments, preserve source-supported values and row grouping, emit one canonical record per image, and replace `annotation.jsonl` atomically.

Run it twice with unchanged inputs and compare hashes. Write a concise `annotation_report.json` with source hashes, output hash and count, reconciliation results, and unresolved issues.

Keep `annotation.image_path` relative to the source-image root. Do not place PNG paths or conversion metadata in `annotation.jsonl`.

### 3. Prepare images

Run this skill's bundled `scripts/prepare_images.py` with the source-image root, artifact directory, and processed-image directory. Do not rewrite the converter or request resize, crop, rotation, flip, or another spatial transform.

```text
uv run python .codex/skills/medical-dataset-preparation/scripts/prepare_images.py --source-image-root <source> --artifact-dir <artifact> --processed-image-dir <artifact/images>
```

The orchestrator runs the locked converter, builds an exact source-path mapping, validates every PNG against its manifest entry, and writes:

```text
<artifact_dir>/
├── scripts/prepare_annotations.py
├── annotation.jsonl
├── source_inventory.json
├── conversion_plan.json
├── annotation_report.json
├── image_manifest.jsonl
├── image_preparation_report.json
└── <processed_images>/
```

`image_manifest.jsonl` contains one row per annotation with `image_id`, `source_path`, relative `png_path`, `width`, `height`, `mode`, `sha256`, conversion status, and spatial-transform metadata. It preserves annotation order and joins only exact normalized source-relative paths.

## Completion

Finish only when annotations satisfy the canonical contract, every annotation maps to one existing verified `L` or `RGB` PNG, no spatial transform occurred, all paths remain beneath the artifact directory, and unchanged reruns produce identical annotations, PNGs, and manifests.
