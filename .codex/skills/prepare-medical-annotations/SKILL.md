---
name: prepare-medical-annotations
description: Normalize raw medical imaging annotations into MedBenchAgent's canonical image-centric annotation.jsonl with a reproducible dataset-specific script and concise mapping artifacts. Use when source tables, JSON files, or metadata must be reconciled before image conversion.
---

# Prepare Medical Annotations

Keep sources read-only and local. Preserve source-supported facts without inferring labels or relationships from filenames, medical knowledge, or image appearance.

## Inputs

Obtain the annotation sources, source-image root, and a new artifact directory. Read the canonical contract at `src/medbenchagent/schemas/annotation.v1.schema.json`.

## Workflow

1. Inventory the supplied annotation files and image references. Determine the authoritative image population, record granularity, identifiers, joins, conflicts, missing rows, and stable ordering.
2. Write concise `source_inventory.json` and `conversion_plan.json` files with observed counts and mapping decisions. Stop when identity, joins, grouping, or conflicts remain materially ambiguous.
3. Write `<artifact_dir>/scripts/prepare_annotations.py` with all dataset-specific logic. Require explicit paths, preserve source values and row grouping, emit one canonical record per image, and replace `annotation.jsonl` atomically.
4. Run the generated script twice and compare hashes. Validate the result with this skill's `scripts/validate_annotations.py`, then write `annotation_report.json` with source hashes, output hash and count, reconciliation results, and unresolved issues.

Keep every `image_path` relative to the source-image root. Do not decode images or add PNG metadata during this stage.

```text
uv run python .codex/skills/prepare-medical-annotations/scripts/validate_annotations.py --annotation <artifact/annotation.jsonl> --source-image-root <source>
```

## Output

```text
<artifact_dir>/
├── scripts/prepare_annotations.py
├── annotation.jsonl
├── source_inventory.json
├── conversion_plan.json
└── annotation_report.json
```

Finish when the canonical records are valid, identifiers and source-relative paths are unique, referenced files resolve beneath the source root, counts reconcile, and unchanged reruns are byte-identical.
