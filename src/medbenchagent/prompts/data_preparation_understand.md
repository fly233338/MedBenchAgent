Current stage: understand

Objective

Understand every configured source and produce a source-grounded,
implementable preparation mapping. Complete source inspection and semantic
interpretation within this conversation.

Required deliverables

- source_report.json
- image_population.jsonl
- mapping.json
- semantic_questions.json
- inspect_source.py
- submission.json

Use the corresponding schemas in inputs/contracts. Write deliverables under
./outputs.

Procedure

1. Establish source coverage. Inspect the configured source set, file
   structures, tables, nested records, image/volume formats, masks, and
   available documentation. Identify the authoritative population and stable
   source-unit identities. Record coverage limits explicitly.
2. Build reproducible observations. Write and execute inspect_source.py using
   the input/output interface declared in the context. Record measured counts,
   schemas, missing values, duplicate keys, joins, unmatched records, and
   relevant boundary cases. Store complete results in files.
3. Explain how records relate. Determine Case identity, asset membership, field
   scope, annotation scope, and documented relationships. Do not merge
   independent examinations merely because they share a patient identifier.
   Apply the declared canonical grouping rules. Record unresolved identity or
   relationship choices.
4. Classify source fields. For each field, record whether it is retained as
   medical evidence, retained as grouping/provenance/asset metadata, or
   excluded. Include a source-backed reason and a stable mapping rule ID. Do
   not use a field-name blacklist as the basis for exclusion.
5. Define transformation semantics. Specify target locations, raw-value
   preservation, value conversions, missingness, units, grouping, joins,
   asset treatment, and population accounting. Record field_semantics for
   retained planning-relevant fields. Use unknown where meaning is not
   established.
6. Resolve only what the evidence supports. Consult source documents and
   applicable user decisions. Every semantic decision must carry resolvable
   evidence references. Record contradictions and missing facts in
   semantic_questions.json.
7. Check the handoff. Ensure that another coding session could implement the
   preparation using these files without access to this conversation.
   Essential decisions must be in structured fields, not only in notes.

Questions and completion

- Write semantic_questions.json even when there are no questions.
- Mark a question as blocking when proceeding would require inventing meaning,
  choosing an unsupported population, or producing materially ambiguous data.
- If blocking questions remain, preserve the current reports, mapping draft,
  and inspection script and submit needs_input.
- If the source cannot be read or an execution dependency is missing, report
  the concrete technical blocker instead of asking the user to supply a
  medical interpretation.
- Before submitting, run the inspection script, check its outputs, validate
  the stage file structures, and report what was checked.
- Do not perform the full dataset conversion in this stage.
- Submit completed deliverables for host acceptance. Do not declare the
  mapping accepted yourself.
