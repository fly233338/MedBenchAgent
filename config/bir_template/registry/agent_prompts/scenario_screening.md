# Scenario Feasibility Screening

## 1. Purpose

Review one candidate Scenario before Leaf Task design.

Keep the Scenario when the supplied evidence supports at least one viable
task within its input structure, operation, and clinical target.

Do not require a pre-existing question or final answer field.

## 2. Review Procedure

Check the following in order:

1. Scenario alignment
   - Interpret the Scenario using the supplied Registry definitions.
   - Keep its Axis values unchanged.

2. Input feasibility
   - Check that the required primary inputs and explicit relationships exist.
   - Apply the benchmark input policy.

3. Task possibility
   - Identify whether at least one clinically meaningful question could
     require the specified operation on the specified target.
   - Do not classify a task solely by its annotation type or answer format.

4. Reference feasibility
   - Accept direct annotations or deterministic derivations using supported
     operations.
   - Check the required entities, measurement scopes, units, and relationships.
   - Do not assume missing fields, transformations, or case-specific facts.

5. Benchmark validity
   - Exclude a Scenario when the available evidence establishes that no
     viable task avoids leakage, unsupported inference, or an input-independent
     shortcut.

## 3. Semantic Constraints

1. Finite categorical annotations can support recognition or classification.
   A categorical value stored as text does not by itself support description.
2. A finding mask can support localization of that finding. It does not
   spatially localize a diagnosis or assessment category.
3. Quantification requires an explicit numeric reference or a supported
   deterministic derivation with a defined scope.
4. Comparison requires explicitly related, clinically comparable entities
   and an identifiable comparison dimension.
5. Alignment requires a correspondence or concordance task; multiple inputs
   alone are insufficient.
6. Description requires multiple supported observations and an appropriate
   reference, not a verbose restatement of one category.
7. A failure of one possible Leaf Task does not invalidate every task in
   the Scenario.
8. Do not infer a constant answer distribution from a small example set.
   Use supplied coverage and distribution evidence.

## 4. Decision

- `keep`: at least one evidence-supported task route exists.
- `exclude`: the supplied evidence does not establish any viable route,
  or establishes a blocking incompatibility.

Explain the decisive evidence or missing prerequisite in the reason.
Distinguish missing evidence from an explicit incompatibility.

Do not design Leaf Tasks, question templates, or answer options.

## 5. Output

Return JSON only, with exactly:

- `decision`: `keep` or `exclude`.
- `reason`: a concise, non-empty explanation.
