---
name: normalize-medical-image-dataset
description: Explore a local medical imaging dataset's tabular annotations and deterministically normalize all source-supported image annotations into an image-centric annotation.jsonl file.
---

# Normalize Medical Image Dataset

Use this skill when given a medical imaging dataset with tabular annotations and asked to create an image-centric `annotation.jsonl`. Read `schema.json` in this skill directory completely before inspecting source data. The schema is the output contract.

## Purpose and boundaries

The workflow starts with local annotation tables and produces one normalized JSONL file. It is dataset-agnostic: do not assume a disease, modality, anatomy, source filename, directory layout, annotation vocabulary, or fixed set of labels.

- Treat source files as read-only and write outputs to a separate destination.
- Keep source data local; do not upload images, rows, identifiers, or annotations to external services.
- Preserve only source-supported information. Do not infer diagnoses, findings, labels, relations, or image facts from medical knowledge or image appearance.
- Use code for record conversion, never a model row by row.
- Stop and request clarification if image identity, join keys, row granularity, conflicts, or target structure remain materially ambiguous after exploration.

## Output contract

Write `annotation.jsonl` with exactly one JSON object per image. Every record has these six top-level fields:

```json
{
  "image_id": "image-001",
  "image_path": "images/image-001.png",
  "patient_id": null,
  "case_id": null,
  "study_id": "study-001",
  "annotation": {
    "modality": "Mammography",
    "anatomy": "Breast",
    "finding": "example source value"
  }
}
```

Apply these rules:

- Require every top-level field defined by `schema.json`; write unavailable patient, case, or study identifiers as `null`.
- Put all dataset-defined content, including `modality` and `anatomy`, directly inside `annotation`. Do not use an `annotation.metadata` wrapper or add dataset-specific fields at the record top level.
- Populate `annotation.modality` and `annotation.anatomy` from source annotations when available; otherwise use documented dataset information and record that decision.
- Keep the annotation structure shallow. Put image-level values directly under `annotation`; use arrays only when repeated source rows genuinely belong to the same image, keeping fields from one source row together in one array item.
- Preserve source values and JSON-compatible types. Do not translate labels, collapse categories, or normalize terminology unless the source documentation defines that transformation.
- Use a source image identifier when available. Otherwise derive `image_id` deterministically from a unique source path or filename and document the rule.
- Make `image_path` a non-empty relative, source-supported logical reference. It may refer to an image not present locally; document whether local existence was verified.

## Workflow

### 1. Read and inventory the source tables

Explore every supplied table, sheet, JSON annotation file, image manifest, and dataset document before deciding a mapping. For each table or sheet, record row count, columns, inferred types, null counts, unique counts, representative non-sensitive values, candidate image/patient/case/study identifiers, candidate image paths, duplicate keys, and candidate join cardinalities.

Determine whether repeated rows represent images, findings, regions, reports, or another source-supported entity. If image assets are available, index their relative paths, filenames, and stems and measure exact, missing, and ambiguous matches. Otherwise inventory source-supported image names or hierarchy fields and mark output paths as logical references.

Write the observed results to `source_inventory.json`; do not copy full annotation tables or unnecessary sensitive values into it.

### 2. Design and confirm the mapping

Choose the authoritative image population, reconcile every annotation table to it, and write `conversion_plan.json` before converting. Include the source inputs, stable output ordering, image-path derivation, join keys and measured cardinalities, source-row-to-annotation mappings, modality/anatomy provenance, included and excluded columns, missing/orphaned/duplicate/conflicting records, expected counts, and unresolved decisions.

Do not silently flatten a one-to-many relationship into duplicate image records, discard unmatched source annotations, or resolve conflicts by table order.

### 3. Implement a deterministic converter

Write a dataset-specific conversion script that implements only the approved plan. It must group all annotations for an image before emitting one record, preserve stable record and array ordering, parse source values deterministically, write through a temporary file, and replace the final JSONL only after success. Re-running unchanged inputs must produce identical bytes.

### 4. Validate and record the result

Validate each JSONL line against `schema.json`. Check unique non-empty image IDs, non-empty relative image paths, output count against the image population, source-row reconciliation, intended join cardinalities, and preservation of row-level field grouping. Run a second conversion and compare output SHA-256 hashes.

Write `conversion_report.json` and `workflow_log.md` beside the output. Record observable mapping decisions, count reconciliation, validation results, commands, errors/fixes, and source/converter/output hashes. Do not include chain-of-thought or unnecessary protected health information.

## Completion criteria

Finish only when the source data has not been modified, every source-supported annotation is accounted for, `annotation.jsonl` satisfies `schema.json`, output ordering is deterministic, and the mapping artifacts are sufficient for another engineer to reproduce the conversion.
