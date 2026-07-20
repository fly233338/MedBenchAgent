---
name: normalize-medical-image-dataset
description: Inspect heterogeneous medical imaging datasets and convert their local tabular annotations and image references into the project's image-centric annotation.jsonl contract. Use when Codex is asked to normalize CSV, XLSX, JSON, DICOM, or image-folder datasets; consume an image_conversion_manifest.jsonl from $convert-medical-images-to-png; resolve image IDs and multi-table joins; write a dataset-specific deterministic converter; validate the result; or document a reproducible medical dataset onboarding workflow.
---

# Normalize Medical Image Dataset

Convert one local medical imaging dataset into a validated, image-centric `annotation.jsonl`. Treat `schema.json` in this skill directory as the output contract and read it completely before inspecting the dataset.

## Boundaries

- Keep the benchmark application outside this workflow. Produce normalized data only.
- Treat every source file as read-only. Write into a separate output directory.
- Keep medical data local. Do not upload rows, images, identifiers, or annotations to external services.
- Do not infer diagnoses, findings, labels, or relationships from medical knowledge or from the images. Preserve only source-supported information.
- Use the model to inspect structure and write conversion code. Never ask a model to convert records row by row.
- Stop and ask the user when image identity, join keys, row granularity, conflicting values, or target nesting remain materially ambiguous after inspection.

## Output Contract

Write exactly one JSON object per image and name the file `annotation.jsonl`:

```json
{
  "image_id": "image-001",
  "image_path": "images/image-001.png",
  "patient_id": null,
  "case_id": null,
  "study_id": "study-001",
  "annotation": {
    "modality": "X-ray",
    "labels": ["example-label"],
    "regions": [{"label": "example-region", "bbox": [10, 20, 30, 40]}]
  }
}
```

Apply these rules:

- Require all six top-level fields from `schema.json` on every record.
- Write unavailable `patient_id`, `case_id`, and `study_id` values as explicit `null`.
- Keep dataset-defined content inside `annotation`; avoid additional top-level dataset fields.
- Preserve source values and JSON-compatible types. Do not translate labels, collapse clinical categories, or normalize terminology unless the source documentation defines the transformation.
- Store image-level scalar or object annotations directly under `annotation`.
- Group repeated rows for one image into a semantic array such as `findings`, `regions`, `labels`, or `reports`. Keep fields from the same source row in the same array element.
- Prefer general medical entity names over source filenames or sheet names. Do not force a fixed medical ontology.
- Use an explicit source image identifier when available. Otherwise derive `image_id` deterministically from a unique image path or filename and document the rule.
- Make `image_path` a relative, source-supported logical reference to the dataset image. It may name an image file not present locally, or a DICOM multi-level hierarchy such as `study_id/series_id/image_id.dcm`.
- Derive the reference only from source image names, paths, identifiers, or documented DICOM hierarchy. Document the derivation rule and do not claim that the target exists locally unless image assets are actually present and checked.

### Converted-image manifest

When `image_conversion_manifest.jsonl` is supplied, join annotations to rows whose `status` is `converted` and write the matched `png_relative_path` into `image_path`.

- Match normalized `source_relative_path` first.
- Fall back to `source_filename`, then source stem, only when the candidate is unique.
- Stop on missing or ambiguous matches. Never select the first candidate.
- Require one normalized annotation record per converted PNG used by the dataset.
- Treat `image_path` as relative to the conversion output root, which contains the `images/` directory.

## Workflow

### 1. Ground in the repository

Read the applicable `AGENTS.md`, repository status, the requested source directory, and any project-side validator before writing files. Preserve unrelated and user-owned changes.

### 2. Inventory the source

Recursively inventory tables, sheets, JSON files, image assets, reports, and dataset documentation. For every table or sheet, record:

- row count, columns, inferred types, null counts, unique counts, and a few representative values;
- candidate image, patient, case, and study identifiers;
- candidate image path or filename fields;
- duplicate-key counts and candidate join cardinalities;
- whether repeated rows represent images, findings, regions, reports, or another source-supported entity.

When image assets are available, build a disk image index using relative path, filename, and stem. Measure exact, missing, and ambiguous matches rather than selecting the first match. When they are unavailable, inventory source-supported image names or DICOM hierarchy fields and record that `image_path` is a logical reference with no local-existence check.

Write `source_inventory.json` with the observable results. Avoid copying full tables or unnecessary patient-level values into this artifact.

### 3. Design the mapping

Choose the authoritative image population and reconcile every annotation source against it. Write `conversion_plan.json` before writing the converter. Include:

- source files and sheets used;
- the image inclusion rule and stable record ordering;
- image-path derivation and hierarchy-ID mappings, including whether paths are logical references or verified local assets;
- join keys with measured `1:1`, `1:N`, or `N:M` cardinality;
- each source row entity and its target path under `annotation`;
- included and excluded columns with reasons;
- missing, orphaned, duplicate, or conflicting source records;
- expected input and output counts;
- unresolved decisions requiring user confirmation.

Do not silently flatten a `1:N` relation into duplicate image records. Do not silently discard unmatched annotations or resolve conflicting values by table order.

### 4. Implement a deterministic converter

Write a dataset-specific conversion script next to the other run artifacts. The script must:

- encode only the approved conversion plan;
- parse source values deterministically and preserve Unicode;
- group all annotations for an image before emitting its single record;
- preserve a stable order for records and nested arrays;
- use parameterized or dataframe joins rather than string-built data expressions;
- write to a temporary file and replace the final output only after success;
- produce identical bytes when rerun against unchanged inputs.

Prefer existing repository dependencies. Add a dependency only when the source format cannot be handled safely with the available environment.

### 5. Validate and reconcile

Validate every line against `schema.json` and any repository validator. Also verify:

- unique, non-empty `image_id` values;
- every `image_path` is non-empty, relative, and deterministically derived from source-supported image naming or DICOM hierarchy; when local image assets are present, also verify that resolved paths exist and are unambiguous;
- output count matches the approved image population;
- every used source row maps to an output record;
- every excluded or orphaned row is counted with a reason;
- joins do not create unintended row multiplication;
- nested records retain the fields that belonged together in the source;
- a second conversion run produces the same SHA-256 output hash.

Manually compare representative output records with their source rows, including images with no optional annotations, multiple repeated annotations, missing hierarchy IDs, and conflicting or boundary cases.

Run the bundled validator after the dataset-specific converter:

```powershell
uv run python .codex\skills\normalize-medical-image-dataset\scripts\validate_annotation.py `
  --annotation <output\annotation.jsonl> `
  [--image-manifest <conversion\image_conversion_manifest.jsonl>] `
  [--image-root <conversion-output-root>]
```

The validator checks that the skill schema matches the project schema, enforces the JSONL contract and relative paths, verifies final PNG manifest references, and optionally checks files below `--image-root`.

### 6. Record the workflow

Write these artifacts beside `annotation.jsonl`:

- `source_inventory.json`
- `conversion_plan.json`
- the dataset-specific converter
- `conversion_report.json`
- `workflow_log.md`

Record commands, exit status, source-backed mapping decisions, user confirmations, errors and fixes, count reconciliation, validation results, and input/converter/output hashes. Record observable evidence and decisions, not hidden chain-of-thought. Do not include unnecessary protected health information in logs or examples.

## Completion Criteria

Finish only when the source remains unchanged, the JSONL passes all structural and logical-image-reference checks, all source annotations are accounted for, the conversion is deterministic, and the workflow artifacts are sufficient for another engineer to reproduce the result.
