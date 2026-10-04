# Benchmark Requirement Extraction

## 1. Purpose

Extract the user's benchmark intent from the complete request and
clarification history into one structured object.

Preserve broad, comparative, multi-dataset, and unfamiliar task requests.
Do not restrict the brief to capabilities already supported by the system.

Do not select datasets, design tasks, or decide implementation feasibility.

## 2. Extraction Procedure

1. Read all numbered user inputs and relevant clarification context.
2. Apply later user corrections to earlier requirements.
3. Extract only stated requirements or unambiguous answers to clarifications.
4. Normalize descriptive fields into clear English while preserving meaning.
5. Attach exact user quotations to explicit non-empty fields.

A broad clinical scope or explicitly requested dataset population is valid.
Do not require the user to name a single disease.

## 3. Fields

Return the fields defined by the supplied schema:

1. `clinical_scope`: the stated benchmark scope, or null.
2. `diseases` and `modalities`: explicitly requested concepts, or empty lists.
3. `evaluation_focus`: requested capabilities, priorities, or exclusions.
4. `dataset_mentions`: user-facing dataset names; do not resolve catalog IDs.
5. `intent_notes`: explicit constraints not represented by other fields.
6. `language`: the requested benchmark output language, `en`, `zh`, or null.
7. `sample_count`: the requested count, or null when unspecified.
8. `sample_scope`: explicit sample inclusion and exclusion requirements.
9. `allocation_preferences`: explicit weighting or allocation preferences.
10. `max_total_tasks_per_image`: an explicit total limit across task types,
    or null.
11. `evidence`: source quotations for the extracted requirements.

## 4. Normalization and Scope Rules

1. Do not infer benchmark output language from the language of the message.
2. Normalize clear count expressions, such as `10k`, to integers.
3. Preserve an explicitly invalid sample count or its original text so it
   can be corrected. Do not replace it with a valid count or null.
4. Do not invent diseases, modalities, sample filters, or allocation weights.
5. Distinguish evaluating a disease-related capability from selecting only
   patients with that disease.
6. Include every explicit modality and population restriction in `sample_scope`.
7. Group modality alternatives into one scope per operator.
   "CT or MRI" means alternatives, not an intersection.
8. Use `all` only for an explicitly simultaneous requirement.
9. A clinical topic alone is not an allocation preference.
10. Preserve explicit input permissions, task granularity, and capability
    requests in the appropriate field or in `intent_notes`.
    Do not silently rewrite them to match a runtime limitation.

## 5. Evidence Rules

1. Copy quotations verbatim from numbered user inputs.
2. Use the zero-based `input_index` of the input containing each quotation.
3. Never quote an assistant question as user evidence.
4. Keep quotations in their original language.
5. Include evidence for every explicit non-empty field, including
   `intent_notes`, invalid explicit sample counts, and per-image limits.
6. Unspecified values require no fabricated evidence.

## 6. Output

Return one JSON object matching the supplied schema, with no wrapper.

Use null or empty lists for unspecified values as appropriate.
Always include `sample_scope` and `allocation_preferences`, even when empty.

Return JSON only.
