---
name: medical-dataset-preparation
description: Prepare a medical imaging dataset as separate normalized annotations and deterministic 8-bit PNG assets linked by image_id. Use when a dataset needs annotation.jsonl, converted images, image_manifest.jsonl, and reproducible per-dataset annotation/image preparation scripts.
---

# Medical Dataset Preparation

Keep annotation facts and prepared pixels separate. Treat sources as read-only, keep data local, and do not infer medical facts from image appearance.

## Required inputs and artifact layout

Before inspecting or writing data, obtain all of the following explicit paths:

1. Annotation source location.
2. Source-image root.
3. Artifact output directory.
4. Processed-image root.

The artifact output directory must be new, or initially contain only the two generated scripts. The processed-image root must be absent or empty and be inside the artifact output directory, or otherwise be reachable from it by a non-escaping relative path. This is required because `image_manifest.jsonl` stores a relative, resolvable `png_path`. If the two locations are on separate machines or mounts without that relationship, stop and ask for a compatible output layout; never write absolute paths or `..` traversal into the manifest.

Create this layout before writing any annotation output:

```text
<artifact_output_dir>/
├── scripts/
│   ├── prepare_annotations.py
│   └── prepare_images.py
├── annotation.jsonl
├── source_inventory.json
├── conversion_plan.json
├── conversion_report.json
└── image_manifest.jsonl
```

`prepare_annotations.py` and `prepare_images.py` are required deliverables, not temporary code. They must be deterministic, runnable from a clean checkout, accept the relevant explicit paths as command-line arguments, and write only to the declared output locations. Keep all dataset-specific transformation logic in these two files.

## 1. Normalize annotations

Use `$normalize-medical-image-dataset` as the authoritative annotation workflow. Read its `schema.json` completely before inspecting the source.

1. Inventory every supplied annotation table, sheet, JSON file, image index, and relevant dataset document. Identify candidate image, patient, case, and study keys; measure duplicate, missing, orphaned, conflicting, and one-to-many relationships.
2. Determine the authoritative image population and write `source_inventory.json`.
3. Design the mapping before writing conversion code. Record source inputs, join keys and measured cardinalities, image-ID and image-path derivation, stable ordering, field mappings, grouping rules, included and excluded source fields, expected counts, and unresolved decisions in `conversion_plan.json`.
4. Stop for clarification when image identity, joins, row granularity, conflicts, or grouping remain materially ambiguous. Do not resolve them by table order or medical inference.
5. Write `<artifact_output_dir>/scripts/prepare_annotations.py`. It must accept the annotation source, source-image root, and artifact output directory; validate that its own location is `<artifact_output_dir>/scripts/`; emit exactly one `annotation.jsonl` object per image; group repeated source rows without separating fields that belong together; preserve source-supported values and types; and write atomically.
6. Run the generated annotation script, then validate every record against the annotation schema, reconcile source and output counts, check unique non-empty image IDs and relative image paths, rerun the script, and compare output hashes. Preserve the normalization report and workflow log in the artifact directory.

During mapping design, define each `image_path` relative to the source-image root that will later be passed to the image converter. Keep it as a source-supported logical reference; do not replace it with a PNG path or add pixel-conversion metadata to the annotation.

## 2. Prepare images

1. Write `<artifact_output_dir>/scripts/prepare_images.py`. It must accept the source-image root, artifact output directory, and processed-image root.
2. The script must be a thin deterministic orchestrator. It must invoke this skill's vendored converter at `.codex/skills/medical-dataset-preparation/scripts/convert_images.py` with `uv run --script` through `subprocess.run(..., check=True)`; it must not copy, rewrite, or reimplement pixel conversion. This preserves the converter's declared image-decoding dependencies.
3. Resolve and verify the converter path before running. Allow an optional `--converter-script` override for environments where the repository is installed elsewhere, but default to this skill's repository-relative copy. Record the resolved converter path and SHA-256 in the image-stage report.
4. Invoke it as `--input <source-image-root> --output <processed-image-root>`. Do not request resize or apply crop, flip, rotation, or another spatial transform.
5. Preserve the converter-generated `images/`, `image_conversion_manifest.jsonl`, mapping CSV, and conversion report under the processed-image root. Treat any recognized-image conversion failure as a failed preparation run.
6. After successful conversion, have `prepare_images.py` invoke this skill's `scripts/build_image_manifest.py` using `<artifact_output_dir>/annotation.jsonl`, the converter manifest, and `--png-root <processed-image-root>`. Write `<artifact_output_dir>/image_manifest.jsonl` atomically and verify every `png_path` resolves beneath the artifact output directory.

## 3. Build the image mapping

`prepare_images.py` runs this skill's bundled `scripts/build_image_manifest.py` after conversion. Inspect both stage reports and verify the final mapping before declaring the dataset prepared.

The mapping script joins `annotation.image_path` to `source_relative_path` exactly after normalizing path separators. It rejects missing, duplicate, or ambiguous mappings and preserves annotation order. If the two stages cannot share one unambiguous source-relative path convention, stop and resolve the mapping instead of adding fuzzy matching.

## Output boundaries

- Leave `annotation.jsonl` unchanged after normalization. Do not add PNG paths, hashes, dimensions, display transforms, or conversion status to it.
- Write one `image_manifest.jsonl` row for every annotation record. Each row contains `image_id`, `source_path`, `png_path`, `width`, `height`, PNG `sha256`, `conversion_status`, and an explicit spatial-transform record.
- Resolve `png_path` relative to the image manifest's directory and ensure it does not traverse out of the artifact directory. Downstream image consumers look up the PNG by `image_id`; they do not use `annotation.image_path` as the prepared-image location.
- Keep unreferenced converted files in the conversion artifacts, but do not add them to `image_manifest.jsonl`.

## Completion criteria

Finish only when every annotation `image_id` maps to exactly one successfully converted 8-bit PNG, all manifest paths are relative and resolvable, dimensions and hashes are present, no spatial transform occurred, and rerunning unchanged inputs produces identical annotations, PNGs, and manifests.
