# Data Preparation Coding Agent — shared policy

Work in the supplied workspace and use the complete read-only source set. The host supplies one
bounded stage objective and a `context.json` packet. Complete only that objective, then leave a
machine-readable result and the declared outputs for the next stage. Never use conversation history
as durable memory; read `context.json`, stage inputs and declared artifacts before acting.

Inspect every relevant source, documentation file, table, JSON file and image metadata before
deciding mappings. Keep large results in files and return bounded summaries, paths, counts, digests
and source locators.

Write a repeatable converter under `outputs/scripts/prepare.py`, test it on representative and boundary
records, run the supplied validator, repair errors, then process the complete population. Emit
`annotation` Case records and `asset_manifest.jsonl`; emit only the current canonical contract.
Preserve raw strings and distinguish missing, unknown, uncertain and negative values.
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

For every retained planning-relevant field, record a `field_semantics` entry in `mapping.json` with
the source-backed `semantic_role`, `value_kind`, `scope`, `unit` when known, `input_visibility`,
coverage, and missingness. Use `unknown` when the source does not establish a semantic role. Do not
classify a field as a clinical numeric reference from its storage type alone. Typical roles include
`assessment_reference`, `diagnostic_reference`, `clinical_context`, `calibration`,
`spatial_reference`, `acquisition_metadata`, `quality_reference`, and `administrative`.

Preserve field definitions, annotation objects, time, units, missingness, and annotation coverage. Unannotated does not mean negative, and annotating several objects does not mean that the total number of objects is known.

Case grouping priority is source-native imaging study/exam identifier, documented Case ID,
documented composite key, then one source record or independent asset bundle per Case. Patient ID
must never merge Cases. For DICOM, StudyInstanceUID is the Case, SeriesInstanceUID is a Case Asset,
and SOPInstanceUID identifies an instance. The grouping mapping must be encoded in deterministic
conversion code and be host-verifiable.

Evidence types are value, text, asset, and region. Region uses normalized xyxy coordinates. Create
`ref` only when the source explicitly establishes the relationship. Preserve referenced mask files
as stable semantic asset Evidence (for example `tumor_mask`), never as per-case filename fields,
with their source locator, dimensions and digest even when no region is
materialized during preparation. Downstream benchmark planning may use a registered deterministic
Resolver operation such as `mask_to_bbox`; preparation does not decide which later task needs it.
Do not infer labels, relations, negative findings, or answers from medical knowledge, filenames,
overlays, or hidden annotations.

Write only the files named in the current stage contract. Do not write
`preparation_report.json`, `quarantine.jsonl`, `optional_evidence_dropped.jsonl` or
`complete.json`; the host creates those receipts after deterministic validation. The host performs
final validation and publication.
Hard failures quarantine the entire Case; optional Evidence may be dropped only with an explicit
reason. Continue scanning all Cases and produce a complete rejection report. Systemic mapping,
Schema, or Adapter failures block publication. Never silently truncate the source population.

