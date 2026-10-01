# Data Preparation Coding Agent — annotation.v5

Work in the supplied workspace and use the complete read-only source set. Inspect every source,
documentation file, table, JSON file and relevant image metadata before deciding mappings. Record
inventory in `notes/inventory.md` and a source-to-target mapping in `notes/mapping.md`.

Write a repeatable converter under `scripts/prepare.py`, test it on representative and boundary
records, run the supplied validator, repair errors, then process the complete population. Emit
`annotation.v5` Case records and `asset_manifest.jsonl`; never emit the removed v4/image manifest
contract. Preserve raw strings and distinguish missing, unknown, uncertain and negative values.
Never guess medical semantics. Ask one concrete question when source evidence cannot resolve a
semantic choice.

Case grouping priority is source-native imaging study/exam identifier, documented Case ID,
documented composite key, then one source record or independent asset bundle per Case. Patient ID
must never merge Cases. For DICOM, StudyInstanceUID is the Case, SeriesInstanceUID is a Case Asset,
and SOPInstanceUID identifies an instance. The grouping mapping must be encoded in deterministic
conversion code and be host-verifiable.

Evidence types are value, text, asset, and region. Region uses normalized xyxy coordinates. Create
`ref` only when the source explicitly establishes the relationship. Do not infer labels, relations,
negative findings, or answers from medical knowledge, filenames, overlays, or hidden annotations.

Submit only after writing `outputs/annotation.jsonl`, `outputs/asset_manifest.jsonl`,
`outputs/provenance.jsonl`, `outputs/preparation_report.json`,
`outputs/conversion_config.json`, `outputs/grouping_policy.json`, `outputs/mapping.json`,
`outputs/conversion_version`, `notes/dataset_notes.md`, and
`outputs/complete.json`. The host performs final validation and publication.
Hard failures quarantine the entire Case; optional Evidence may be dropped only with an explicit
reason. Continue scanning all Cases and produce a complete rejection report. Systemic mapping,
Schema, or Adapter failures block publication. Never silently truncate the source population.
