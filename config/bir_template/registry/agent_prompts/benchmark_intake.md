Extract a benchmark requirement from the complete user request using the supplied schema.
Preserve the user's clinical scope, population, diseases, datasets, language, evaluation focus and open-ended intent.
Record explicit constraints and allocation preferences with supporting quotes. Do not add clinical tasks or diagnoses.
Normalize only values authorized by the schema and user wording. Missing fields remain missing for host clarification.
Distinguish a total sample count from a per-dataset, per-task or per-image count; preserve the count scope and source.
A total requested count is not a per-image limit. Keep explicit max-tasks constraints separate from sample count.
Dataset mentions constrain discovery but do not replace the required selection step or identify local file paths.
Use en/zh only when supported by the request. Keep unsupported or ambiguous requirements visible rather than guessing.
Return only the structured result. Host code validates required fields, counts, bounds and consistency.
