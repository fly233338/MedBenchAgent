---
name: prepare-medical-annotations
description: Normalize raw medical imaging annotations into MedBenchAgent's canonical image-centric annotation.jsonl with a reproducible dataset-specific script and concise mapping artifacts. Use when source tables, JSON files, or metadata must be reconciled before image conversion.
---

# Prepare Medical Annotations

Keep sources read-only and local. Preserve source-supported facts without inferring labels or relationships from filenames, medical knowledge, or image appearance.

## Inputs

Obtain the annotation sources, source-image root, and a new artifact directory.

Read `annotation.v1.schema.json` in this skill directory completely before inspecting the source data. Treat it as the authoritative output contract. Resolve bundled resources relative to this `SKILL.md`; do not assume a repository layout or skill installation path.

## Workflow

### 1. Inventory the sources

Inspect every supplied annotation table, spreadsheet sheet, JSON file, image manifest, and relevant dataset document before defining the mapping.

Record in `source_inventory.json`:

- File and sheet names, row counts, columns, inferred types, null counts, and unique-value counts.
- Candidate image, patient, case, and study identifiers and candidate image paths.
- Duplicate keys, candidate join keys, and measured join cardinalities.
- The observed granularity of each source row, such as image, finding, region, report, or study.
- Exact, missing, and ambiguous matches between annotation references and source-image paths.

Use only concise, non-sensitive representative values. Do not copy complete tables or unnecessary source records into the inventory.

### 2. Design the mapping

Choose the authoritative image population and reconcile every annotation source against it. Write `conversion_plan.json` before implementing conversion.

Record:

- Source inputs and the authoritative image population.
- Stable output ordering and deterministic `image_id` and `image_path` derivation.
- Join keys, measured cardinalities, and source-row-to-annotation mappings.
- The provenance of modality, anatomy, and other included annotation fields.
- Included fields, excluded fields with reasons, and preservation of source value types.
- Missing, orphaned, duplicate, and conflicting records.
- Expected source and output counts and every unresolved decision.

Do not flatten one-to-many annotations into duplicate image records, silently discard unmatched annotations, resolve conflicts by table order, or combine fields from separate source rows without source support. Stop when identity, joins, grouping, conflicts, or output meaning remain materially ambiguous.

### 3. Implement deterministic conversion

Write `<artifact_dir>/scripts/prepare_annotations.py` with all dataset-specific logic documented in the conversion plan.

Require explicit input and output paths. Read sources without modifying them, group all annotations belonging to an image, and emit exactly one canonical record per authoritative image. Preserve source values, JSON-compatible types, and fields originating from the same source row. Use stable record and array-item ordering.

Write to a temporary file and replace `annotation.jsonl` only after complete success. Keep every `image_path` relative to the source-image root. Do not decode images or add PNG metadata during this stage.

### 4. Validate and report

Run the generated converter twice with unchanged inputs and compare SHA-256 hashes. Validate the result with this skill's `scripts/validate_annotations.py` and its local `annotation.v1.schema.json`.

Verify:

- Every line is valid JSON and contains the canonical fields.
- Every `image_id` and `image_path` is non-empty and unique.
- Every `image_path` is relative, remains beneath the source-image root, and resolves to a file.
- The output count matches the authoritative image population.
- Source rows, missing records, orphaned records, duplicates, and conflicts are reconciled.
- Actual join cardinalities match the conversion plan.
- One-to-many annotations remain grouped under one image record.
- Source-row field grouping and source value types are preserved as planned.
- Both converter runs produce byte-identical output.

Write `annotation_report.json` with input and converter hashes, output hash and count, reconciliation counts, validation results, determinism results, observable mapping decisions, and unresolved issues. Do not include complete source rows, unnecessary protected information, or hidden reasoning.

## Output

```text
<artifact_dir>/
├── scripts/prepare_annotations.py
├── annotation.jsonl
├── source_inventory.json
├── conversion_plan.json
└── annotation_report.json
```

Finish only when no source file was modified, every source-supported annotation was included or accounted for, canonical records are valid, identifiers and source-relative paths are unique, referenced files resolve beneath the source root, counts reconcile, and unchanged reruns are byte-identical.
