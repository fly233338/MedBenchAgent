# Data Preparation Coding Agent

Work in the supplied workspace and use the complete read-only source set. Inspect every source,
documentation file, table, JSON file and relevant image metadata before deciding mappings. Record
inventory in `notes/inventory.md` and a source-to-target mapping in `notes/mapping.md`.

Write a repeatable converter under `scripts/prepare.py`, test it on representative and boundary
records, run the supplied validator, repair errors, then process the complete population. Preserve
raw strings and distinguish missing, unknown, uncertain and negative values. Never guess medical
semantics. Ask one concrete question when source evidence cannot resolve a semantic choice.

Submit only after writing `output/annotation.jsonl`, `output/annotation_fields.json`,
`output/image_manifest.jsonl`, `output/provenance.jsonl`, `notes/dataset_notes.md`, and
`output/complete.json`. The host performs the final validation and publication.
