Current stage: prepare

Objective

Implement, test, and execute the accepted preparation mapping over the
complete source population. Perform coding, trial execution, full execution,
inspection, and implementation repair within this conversation.

Required deliverables

- scripts/prepare.py
- cases.jsonl
- asset_manifest.jsonl
- provenance.jsonl
- conversion_config.json
- grouping_policy.json
- execution_report.json
- output_manifest.json
- submission.json

Use the corresponding schemas and rules in inputs/contracts. Write
deliverables under ./outputs. Register optional deliverables and every
required file dependency in output_manifest.json.

Procedure

1. Read the accepted handoff. Read the source report, mapping, population
   contract, applicable user decisions, and output rules. Confirm their
   identities against the context bindings.
2. Plan implementation coverage. Inspect every mapping rule that applies to
   each source/table before implementing that source/table. Track coverage by
   rule_id. A rule not printed in the context packet still applies when it is
   present in the accepted mapping.
3. Implement a reproducible converter. Write scripts/prepare.py using the
   invocation interface declared in the context. Use source-relative
   identities and locators. Keep behavior determined by the frozen inputs,
   accepted mapping, conversion configuration, and execution environment.
4. Test representative and boundary cases. Cover relevant formats, null and
   uncertain values, repeated keys, unmatched joins, multiple assets, masks,
   and other boundary cases identified in the source report. Record the actual
   checks and results. Use the supplied validation command if one is declared.
5. Execute the complete population. Run the converter over all required
   source units. Account for retained, excluded, duplicate, rejected, and
   unprocessed units according to the accepted population policy. Do not use
   a sample run as evidence of complete processing.
6. Inspect and repair implementation. Check output structures, counts, IDs,
   relationships, locators, asset digests, provenance, and population
   reconciliation. Repair code and rerun affected checks when implementation
   fails.
7. Package the submission. Register every deliverable and dependency,
   including generated assets, auxiliary scripts, and optional manifests.
   Ensure paths remain valid under the declared bundle layout.
   Include the accepted mapping and population digests in output_manifest.json
   and execution_report.json; do not silently modify the accepted mapping.

Semantic changes

- Treat the accepted mapping as read-only.
- If source evidence contradicts the mapping or a required semantic choice is
  missing, preserve the evidence and affected partial outputs. Identify the
  mapping rule and the required change.
- Report mapping_revision_required for a proposal that needs to return to the
  understand stage.
- Submit needs_input only when an actual missing user fact is required.
- Do not silently reinterpret fields, change population selection, alter Case
  grouping, or invent missing semantics.

Completion

- execution_report.json must distinguish tests, full execution, failed checks,
  limitations, population accounting, and mapping-rule coverage.
- Record the command and configuration needed to reproduce execution.
- Submit only files that exist and report their actual status.
- Do not create host validation, quarantine, readiness, or publication
  receipts.
- Submission means the candidate is ready for independent host validation; it
  does not mean the dataset has been published.
