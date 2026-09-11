---
name: normalize-medical-image-dataset
description: Prepare a local medical imaging dataset for ingestion by this project by deterministically converting source-supported tabular annotations into the project's canonical image-centric annotation.jsonl format.
---

# Normalize Medical Image Dataset

Use this skill when a local medical imaging dataset must be converted into the canonical image-centric `annotation.jsonl` input required by this project.

Read `schema.json` in this skill directory completely before inspecting the source data. The schema is the authoritative output contract.

## Purpose and boundaries

This workflow converts local annotation tables and related dataset metadata into a single normalized `annotation.jsonl` file that can be consumed directly by the project.

The workflow is dataset-agnostic. Do not assume a disease, modality, anatomy, source filename, directory layout, annotation vocabulary, or fixed label set.

- Treat all source files as read-only and write outputs to a separate destination.
- Keep source data local. Do not upload images, rows, identifiers, or annotations to external services.
- Preserve only information explicitly supported by source annotations or dataset documentation.
- Do not infer diagnoses, findings, labels, relationships, or image facts from medical knowledge, field names, or image appearance.
- Use deterministic code for record conversion. Do not use a model to convert records row by row.
- Stop and request clarification if image identity, join keys, row granularity, conflicts, or the target annotation structure remain materially ambiguous after source exploration.

## Output contract

Write `annotation.jsonl` with exactly one JSON object per image.

Every record must contain these six top-level fields:

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

Apply the following rules:

- Include every top-level field required by `schema.json`.
- Write unavailable patient, case, or study identifiers as `null`.
- Put all dataset-defined annotation content, including `modality` and `anatomy`, directly inside `annotation`.
- Do not add an `annotation.metadata` wrapper.
- Do not add dataset-specific fields at the record top level.
- Populate `annotation.modality` and `annotation.anatomy` from source annotations when available.
- When modality or anatomy is absent from the annotation tables, use documented dataset-level information only when it applies reliably to the relevant images, and record that decision in `conversion_plan.json`.
- Keep the annotation structure shallow.
- Put image-level values directly under `annotation`.
- Use arrays only when repeated source rows genuinely represent multiple annotations belonging to the same image.
- When using arrays, keep fields originating from the same source row together in the same array item.
- Preserve source values and JSON-compatible types.
- Do not translate labels, merge categories, normalize terminology, or reinterpret values unless the source documentation explicitly defines that transformation.
- Use a source image identifier as `image_id` when a reliable unique identifier is available.
- Otherwise, derive `image_id` deterministically from a unique source path or filename and document the derivation rule.
- Make `image_path` a non-empty relative logical reference supported by the source data.
- `image_path` may refer to an image that is not locally available, but the conversion artifacts must state whether local file existence was verified.

## Workflow

### 1. Read and inventory the source data

Explore every supplied annotation table, spreadsheet sheet, JSON annotation file, image manifest, and dataset document before defining the mapping.

For each table or sheet, record:

- Row count
- Column names
- Inferred data types
- Null counts
- Unique-value counts
- Representative non-sensitive values
- Candidate image identifiers
- Candidate patient, case, and study identifiers
- Candidate image paths
- Duplicate keys
- Candidate join keys and measured cardinalities

Determine what each source row represents, such as:

- An image
- A finding
- A region
- A report
- A study
- Another explicitly documented entity

Do not assume that one source row corresponds to one image.

When image assets are locally available:

- Index their relative paths, filenames, and filename stems.
- Measure exact matches between annotation references and local files.
- Identify missing matches.
- Identify ambiguous matches.

When image assets are not locally available:

- Inventory source-supported filenames, paths, or hierarchy fields.
- Treat generated `image_path` values as logical references.
- Record that local existence was not verified.

Write the observed results to `source_inventory.json`.

Do not copy complete annotation tables, full records, or unnecessary sensitive values into the inventory.

### 2. Design and confirm the mapping

Choose the authoritative image population and reconcile every annotation source against it.

Write `conversion_plan.json` before implementing the converter.

The plan must include:

- Source input files
- Authoritative image population
- Stable output ordering
- `image_id` derivation
- `image_path` derivation
- Join keys
- Measured join cardinalities
- Source-row-to-annotation mappings
- Modality and anatomy provenance
- Included source columns
- Excluded source columns and exclusion reasons
- Missing records
- Orphaned records
- Duplicate records
- Conflicting records
- Expected source and output counts
- Any unresolved decisions

Do not:

- Flatten a one-to-many relationship into duplicate image records.
- Discard unmatched source annotations silently.
- Resolve conflicts according to table order.
- Combine fields from separate source rows into one annotation item unless the source structure explicitly supports that combination.
- Proceed when unresolved ambiguity could materially change the output structure or meaning.

### 3. Implement a deterministic converter

Write a dataset-specific conversion script that implements only the documented conversion plan.

The converter must:

- Read source files without modifying them.
- Group all annotations belonging to an image before emitting its record.
- Emit exactly one record per authoritative image.
- Preserve source-row field grouping.
- Use stable record ordering.
- Use stable array-item ordering.
- Parse source values deterministically.
- Avoid environment-dependent ordering or formatting.
- Write to a temporary file first.
- Replace the final `annotation.jsonl` only after the full conversion succeeds.

Running the converter repeatedly with unchanged inputs must produce byte-identical output.

### 4. Validate and record the result

Validate every line of `annotation.jsonl` against `schema.json`.

Verify that:

- Every line is valid JSON.
- Every record satisfies the schema.
- Every `image_id` is non-empty and unique.
- Every `image_path` is non-empty and relative.
- The output count matches the authoritative image population.
- Source rows are fully reconciled.
- Missing, orphaned, duplicate, and conflicting rows are accounted for.
- Actual join cardinalities match the conversion plan.
- One-to-many annotations remain grouped under one image record.
- Fields originating from the same source row remain grouped together.
- Source values and JSON-compatible types are preserved as intended.

Run the conversion a second time and compare both outputs byte for byte.

Write `conversion_report.json` beside `annotation.jsonl`.

The report must include:

- Input file hashes
- Converter hash
- Output file hash
- Output record count
- Source-to-output count reconciliation
- Schema-validation results
- Identifier-uniqueness results
- Image-path validation results
- Join-cardinality validation results
- Determinism-check results
- Missing, orphaned, duplicate, and conflicting record counts
- Observable mapping decisions needed to interpret the output

Do not include chain-of-thought, complete source rows, or unnecessary protected health information.

## Completion criteria

The workflow is complete only when:

- No source file has been modified.
- Every source-supported annotation has been included, excluded with a documented reason, or identified as unresolved.
- `annotation.jsonl` contains exactly one valid record per authoritative image.
- Every output record satisfies `schema.json`.
- Output ordering is stable.
- Repeated conversion produces byte-identical output.
- Source and output counts are reconciled.
- `source_inventory.json`, `conversion_plan.json`, and `conversion_report.json` contain enough observable information for another engineer to reproduce and verify the conversion.
