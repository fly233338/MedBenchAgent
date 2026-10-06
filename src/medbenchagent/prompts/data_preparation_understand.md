Current stage: understand

Objective

Produce an evidence-grounded handoff that a fresh coding session can use
to implement data conversion without access to this conversation. Deliver
measured source observations, a complete population, explicit processing
rules, and scoped unresolved questions. Do not perform full conversion here.

Procedure

1. Establish the source inventory.
   Programmatically enumerate every configured source, relevant table,
   document, and asset format. Inspect bounded representative structures.
   Identify the authoritative population and stable source-unit keys.
   Record specific unresolved fields, identities, joins, and asset rules.

2. Build and run the inspector early.
   Implement inspect_source.py using the exact input-config, output-dir,
   and mode interface in context.execution. Include every configured source
   and register any helper dependencies. Do not hardcode an attempt path.
   Run it and write the complete source report and population to the declared
   directories. Measure schemas, field coverage, value states, duplicate
   keys, join cardinalities, unmatched records, and relevant boundary cases.
   Keep observations deterministic; execution timestamps belong in receipts.

3. Define implementable semantics.
   For every observed field, specify its disposition, rule ID, target, raw
   value preservation, transformation, missingness handling, units, reason,
   and evidence references. Define Case identity, grouping keys, join
   behavior, asset roles, ordering, and population accounting explicitly.
   Do not merge independent examinations solely because they share a
   patient identifier. Record planning-relevant field semantics without
   selecting benchmark tasks.

4. Resolve the remaining unknowns.
   Inspect additional records or documentation only to resolve a named
   unknown or verify a proposed rule. Use source evidence and the applicable
   user decisions. If meaning is not established, preserve it as unknown
   where the contract permits that state. If correct processing requires
   choosing an unsupported interpretation, record a blocking question.

5. Validate the handoff.
   Run the declared handoff validation command. Confirm that all sources
   and observed fields are covered, source-unit identities are unique,
   evidence references resolve, dependencies are registered, and every
   rule is specific enough to implement. Confirm that the submitted report
   and population match the inspector's actual output.
   Put essential decisions in structured mapping fields, not only in notes.

Completion

Use submitted only when the handoff completion checks pass and no unresolved
blocking question remains. Include semantic_questions.json even when its
questions array is empty. The host independently replays the inspector and
decides whether to accept the handoff.

Use needs_input when a genuine missing user fact blocks a rule or population
choice. Keep the measured observations, mapping draft, and inspector, and
identify which work remains partial. Use blocked for technical or contract
failures and give a concrete diagnostic and recovery condition.

Stop inspecting once source coverage and the implementable rule checklist
are complete. Write the submission, ensure no command is still running,
and call finish. Do not label your mapping accepted or begin prepare.
