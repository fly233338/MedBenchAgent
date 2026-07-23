# Vin-Dr Normalization Workflow Log

## Scope

Normalized `datasets/Mammography/Vin-Dr` into one image-centric JSONL file at
`datasets/Mammography/Vin-Dr-normalized/annotation.jsonl`. Source files were treated as
read-only. All run artifacts are in `datasets/Mammography/Vin-Dr-normalized`.

## Inventory

- Read repository instructions in `AGENTS.md`, the project README, the dataset loader,
  the skill `schema.json`, and Vin-Dr `info.md`.
- Source directory contains CSV annotations, documentation, license, and
  `SHA256SUMS.txt`; no local `images/` directory is present.
- `SHA256SUMS.txt` lists 20,000 logical DICOM paths in the form
  `images/<study_id>/<image_id>.dicom`.
- `metadata.csv`: 20,000 rows, 20,000 unique SOP Instance UID values. The duplicate
  `SOP Instance UID` columns are identical in all rows.
- `breast-level_annotations.csv`: 20,000 rows, 20,000 unique `image_id` values.
- `finding_annotations.csv`: 20,486 rows across the same 20,000 unique `image_id`
  values. 338 images have multiple finding rows; the maximum is 7 finding rows.

Detailed inventory is recorded in `source_inventory.json`.

## Mapping Decisions

- Authoritative image population: one output record per
  `breast-level_annotations.csv.image_id`, sorted lexicographically.
- `image_path`: the source-supported logical SHA manifest path
  `images/<study_id>/<image_id>.dicom`; no local image existence is claimed.
- `patient_id` and `case_id`: explicit `null`; no source patient or case identifier was
  available.
- `study_id`: `breast-level_annotations.csv.study_id`.
- `annotation.breast`: breast-level image fields from `breast-level_annotations.csv`.
- `annotation.dicom_metadata`: metadata fields from `metadata.csv`, with numeric and
  list-like DICOM values parsed into JSON-compatible numbers/lists where unambiguous.
- `annotation.modality` and `annotation.anatomy`: fixed `Mammography` modality and `Breast` anatomy from the
  dataset documentation because these values are not supplied as source annotation fields.
- `annotation.findings`: one object per source finding row, preserving row grouping with
  `finding_categories`, `finding_birads`, and `bbox`. Empty finding BI-RADS and empty
  bbox values are explicit `null`.

The complete plan is recorded in `conversion_plan.json`.

## Commands And Results

- `git status --short`: exit 0; pre-existing unrelated `.gitignore` modification was
  observed and not touched.
- Inventory generation command: exit 0; wrote `source_inventory.json`.
- First converter run: exit 1; failed on list-valued `Window Center`. The converter was
  updated to preserve scalar or list-valued numeric window fields.
- `uv run python datasets/Mammography/Vin-Dr-normalized/convert_vindr.py`: exit 0; wrote 20,000
  records.
- Second identical conversion command: exit 0; annotation SHA-256 remained
  `09ff392c7b4f31300f10ec0a7ad31f0b84db79839b816138e7bb45799698f41c`.
- Independent logical validation: exit 0; verified 20,000 records, 20,000 unique image
  IDs, sorted output order, non-empty relative image paths, and all 20,486 finding rows
  accounted for.
- Skill validator command:
  `uv run python .codex\skills\normalize-medical-image-dataset\scripts\validate_annotation.py --annotation datasets\Mammography\Vin-Dr-normalized\annotation.jsonl`
  exited 0; it verified 20,000 structurally valid records and schema parity.
- Repository loader/profile validation command exited 0; it verified 20,000 records and
  exposed `annotation.modality=Mammography` and `annotation.anatomy=Breast`.
- `uv run ruff check .`: exit 0.
- `uv run pytest`: exit 0; 88 passed, 1 skipped.

## Hashes

- `annotation.jsonl`:
  `09ff392c7b4f31300f10ec0a7ad31f0b84db79839b816138e7bb45799698f41c`
- `breast-level_annotations.csv`:
  `8e5369dbd5cbd52a0fa9a4567a27cd04658611a55387dd751dc01473d5723f41`
- `finding_annotations.csv`:
  `59cae3a856026b8b5822ed545e5150efce43f8f2f8aca991ce69d1fd221e4cbc`
- `metadata.csv`:
  `44ec668cb1a43b526a4c9ab341ae748edbd26b451d7ad72a139e070d13020206`
- `SHA256SUMS.txt`:
  `221bd57f7d45fbc25f642fdb676b306b3246380677ac2afebc62b4f298cc2c08`

## Residual Notes

The normalized JSONL uses source-supported logical image references. The project loader
treats `image_path` as opaque metadata, so validation does not require the Vin-Dr DICOM
tree to be present locally.
