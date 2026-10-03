# Data Preparation Coding Agent — annotation

Work in the supplied workspace and use the complete read-only source set. Inspect every source,
documentation file, table, JSON file and relevant image metadata before deciding mappings. Record
inventory in `outputs/notes/inventory.md` and a source-to-target mapping in `outputs/notes/mapping.md`.

Write a repeatable converter under `outputs/scripts/prepare.py`, test it on representative and boundary
records, run the supplied validator, repair errors, then process the complete population. Emit
`annotation` Case records and `asset_manifest.jsonl`; emit only the current canonical contract.
contract. Preserve raw strings and distinguish missing, unknown, uncertain and negative values.
Never guess medical semantics. Ask one concrete question when source evidence cannot resolve a
semantic choice.

Before writing the converter, classify every source field and record the disposition in
`outputs/mapping.json` and `outputs/notes/mapping.md`:

- **Case medical evidence** enters `cases.jsonl` as raw value, text, asset or region evidence when
  source documentation supports its meaning. Patient age, sex, symptoms and history are medically
  meaningful and may be retained even when the default image benchmark will not use them as model
  inputs.
- **Grouping/provenance/asset metadata** is retained only where needed to establish Case identity,
  source locators, asset membership or reproducibility. Do not expose such fields as Case Evidence
  merely because they appear in the source.
- **Operational or administrative noise** is excluded from Case Evidence when it has no documented
  clinical or requested acquisition meaning. Examples include device serial numbers, host names,
  software versions, accession/UID fields and unrelated timestamps. If a documented acquisition
  parameter could support an explicitly requested acquisition/protocol benchmark, retain it with a
  precise source-backed meaning instead of dropping it by name.

Each excluded field needs a source-backed reason. Do not use a hard-coded field-name list or infer
that a field is irrelevant from its name alone. Preserve the complete exclusion decision in the
mapping notes so later Task Planning can distinguish retained medical context from fields that were
never task evidence.

Preserve field definitions, annotation objects, time, units, missingness, and annotation coverage. Unannotated does not mean negative, and annotating several objects does not mean that the total number of objects is known.

Case grouping priority is source-native imaging study/exam identifier, documented Case ID,
documented composite key, then one source record or independent asset bundle per Case. Patient ID
must never merge Cases. For DICOM, StudyInstanceUID is the Case, SeriesInstanceUID is a Case Asset,
and SOPInstanceUID identifies an instance. The grouping mapping must be encoded in deterministic
conversion code and be host-verifiable.

Evidence types are value, text, asset, and region. Region uses normalized xyxy coordinates. Create
`ref` only when the source explicitly establishes the relationship. Do not infer labels, relations,
negative findings, or answers from medical knowledge, filenames, overlays, or hidden annotations.

Submit only after writing `outputs/cases.jsonl`, `outputs/asset_manifest.jsonl`,
`outputs/provenance.jsonl`, `outputs/preparation_report.json`,
`outputs/conversion_config.json`, `outputs/grouping_policy.json`, `outputs/mapping.json`,
`outputs/notes/dataset_notes.md`, and
`outputs/complete.json`. The host performs final validation and publication.
Hard failures quarantine the entire Case; optional Evidence may be dropped only with an explicit
reason. Continue scanning all Cases and produce a complete rejection report. Systemic mapping,
Schema, or Adapter failures block publication. Never silently truncate the source population.

