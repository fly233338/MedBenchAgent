---
name: medical-dataset-preparation
description: Prepare a medical imaging dataset as separate normalized annotations and deterministic 8-bit PNG assets linked by image_id. Use when a dataset needs annotation.jsonl, converted images, and image_manifest.jsonl without adding pixel-conversion metadata to the annotations.
---

# Medical Dataset Preparation

Keep annotation facts and prepared pixels as separate outputs. Treat source files as read-only, keep data local, and do not infer medical facts from image appearance.

## 1. Normalize annotations

Use `$normalize-medical-image-dataset` as the authoritative annotation workflow. Read its `schema.json` completely before inspecting the source.

1. Inventory every supplied annotation table, sheet, JSON file, image index, and relevant dataset document. Identify candidate image, patient, case, and study keys; measure duplicate, missing, orphaned, conflicting, and one-to-many relationships.
2. Determine the authoritative image population and write `source_inventory.json`.
3. Design the mapping before writing conversion code. Record source inputs, join keys and measured cardinalities, image-ID and image-path derivation, stable ordering, field mappings, grouping rules, included and excluded source fields, expected counts, and unresolved decisions in `conversion_plan.json`.
4. Stop for clarification when image identity, joins, row granularity, conflicts, or grouping remain materially ambiguous. Do not resolve them by table order or medical inference.
5. Implement the confirmed plan with dataset-specific deterministic code. Emit exactly one `annotation.jsonl` object per image, group repeated source rows without separating fields that belong together, preserve source-supported values and types, and write atomically.
6. Validate every record against the annotation schema, reconcile source and output counts, check unique non-empty image IDs and relative image paths, rerun the converter, and compare output hashes. Preserve the normalization report and workflow log.

During mapping design, define each `image_path` relative to the source-image root that will later be passed to the image converter. Keep it as a source-supported logical reference; do not replace it with a PNG path or add pixel-conversion metadata to the annotation.

## 2. Prepare images

1. Use `$convert-medical-images-to-png` to convert the source-image root.
2. Preserve the generated `images/`, `image_conversion_manifest.jsonl`, mapping CSV, and conversion report.
3. Do not request resize or apply crop, flip, rotation, or another spatial transform.
4. Treat any recognized-image conversion failure as a failed preparation run.

## 3. Build the image mapping

Run this skill's bundled `scripts/build_image_manifest.py` with the normalized annotation file, the conversion manifest, and the desired `image_manifest.jsonl` output path. Then inspect both stage reports and verify the final mapping before declaring the dataset prepared.

The mapping script joins `annotation.image_path` to `source_relative_path` exactly after normalizing path separators. It rejects missing, duplicate, or ambiguous mappings and preserves annotation order. If the two stages cannot share one unambiguous source-relative path convention, stop and resolve the mapping instead of adding fuzzy matching.

## Output boundaries

- Leave `annotation.jsonl` unchanged after normalization. Do not add PNG paths, hashes, dimensions, display transforms, or conversion status to it.
- Write one `image_manifest.jsonl` row for every annotation record. Each row contains `image_id`, `source_path`, `png_path`, `width`, `height`, PNG `sha256`, `conversion_status`, and an explicit spatial-transform record.
- Resolve `png_path` relative to the image manifest's directory. Downstream image consumers look up the PNG by `image_id`; they do not use `annotation.image_path` as the prepared-image location.
- Keep unreferenced converted files in the conversion artifacts, but do not add them to `image_manifest.jsonl`.

## Completion criteria

Finish only when every annotation `image_id` maps to exactly one successfully converted 8-bit PNG, all manifest paths are relative and resolvable, dimensions and hashes are present, no spatial transform occurred, and rerunning unchanged inputs produces identical annotations, PNGs, and manifests.
