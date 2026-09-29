# Data Preparation Coding Agent

Work in the supplied workspace and use the complete read-only source set. Inspect every source,
documentation file, table, JSON file and relevant image metadata before deciding mappings. Record
inventory in `notes/inventory.md` and a source-to-target mapping in `notes/mapping.md`.

Write a repeatable converter under `scripts/prepare.py`, test it on representative and boundary
records, run the supplied validator, repair errors, then process the complete population. Preserve
raw strings and distinguish missing, unknown, uncertain and negative values. Never guess medical
semantics. Ask one concrete question when source evidence cannot resolve a semantic choice.

Submit only after writing `outputs/annotation.jsonl`, `outputs/annotation_fields.json`,
`outputs/image_manifest.jsonl`, `outputs/provenance.jsonl`, `notes/dataset_notes.md`, and
`outputs/complete.json`. The host performs the final validation and publication.
For every image manifest row, preserve the original image in place and emit
`image_id`, `source_id`, `source_path`, `width`, `height`, and `file_digest`; do not copy
source images into the candidate workspace.
