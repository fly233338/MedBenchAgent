Current stage: prepare

Objective

Implement, test, and execute the accepted mapping over the complete frozen
source population, then submit a reproducible candidate for host validation.
The accepted mapping defines semantics for this stage.

Procedure

1. Establish the accepted inputs.
   Read the AcceptedHandoff, the bounded source report, applicable decisions,
   execution interface, and output contract index. Confirm that the referenced
   files and identities match the host context. If the accepted handoff is
   missing or inconsistent, report a contract blocker; do not reconstruct
   accepted semantics from a previous conversation. Resolved answers are
   supplied in inputs/decisions.json; they do not change the accepted mapping
   unless the host routes the work back through understand.

2. Index all mapping rules.
   Load the complete mapping programmatically and create a rule coverage
   table. Read the details of every applicable rule before implementing its
   source or rule group. Summaries and samples do not reduce the rule set.
   Track the implementation and verification reference for each rule ID.

3. Implement the converter.
   Create scripts/prepare.py with the exact input-config, output-dir, and
   sample/full mode interface in context.execution. Use the declared source
   paths and accepted population. Keep IDs, locators, grouping, joins,
   missingness, raw values, units, and asset roles consistent with the mapping.
   Register helper scripts and file dependencies. Generate grouping_policy
   through the supplied projection helper rather than defining new semantics.

4. Run representative and boundary tests.
   Select cases covering the actual rule groups and observed boundaries:
   null and uncertain values, duplicate keys, unmatched joins, multiple
   assets, encodings, masks, volumes, and any additional relevant condition.
   Run the sample command into work/sample and check results against the
   accepted rules. Record real execution references and expected outcomes.
   Repair implementation failures and rerun affected checks.

5. Execute the complete population.
   Run the full command into the declared output directory. Track it to
   completion. Give every frozen source unit exactly one final accounting
   outcome and preserve its Case relationships and provenance. A source
   unit may produce multiple Cases, and a Case may combine source units.
   Do not equate source-unit count with Case count or claim full completion
   from a successful sample.

6. Validate and package.
   Verify schemas, IDs, relationships, source locators, asset digests,
   encoding preservation, provenance, rule coverage, and full population
   reconciliation. Use the supplied report and packaging helpers, then run
   the candidate validation command and inspect `work/validation.json`.
   Register all output dependencies and enabled optional manifests. Bind
   reports to the accepted mapping and population using the exact schema
   fields supplied by the host. Do not submit unless the shared validator
   reports `accepted`.

Handling conflicts

Inspect source data only to implement or diagnose a specific rule. Do not
restart open-ended source interpretation. Repair coding, test, and execution
failures in this stage without changing the accepted mapping.

If a rule contradicts source evidence, omits an essential semantic choice,
or requires a change to population, grouping, or field meaning, preserve
the affected work and submit blocked with mapping_revision_required.
Include affected rule IDs, source evidence, observed versus expected
behavior, and the change that needs understand-stage review.

Use needs_input only when an actual missing user fact is required. Write
the current question into this stage's semantic_questions.json and reference
its issue ID. Do not overwrite or modify upstream questions or decisions.

Completion

Use submitted only after relevant tests and the full run actually succeed,
all source units are accounted for, all required mapping rules are covered,
and the declared candidate checks pass without unresolved blocking issues.
Record failed checks and limitations accurately. Do not write host-owned
acceptance, quarantine, readiness, or publication receipts.

Write the submission, ensure no command is still running, and call finish.
The host independently validates and publishes the candidate.
