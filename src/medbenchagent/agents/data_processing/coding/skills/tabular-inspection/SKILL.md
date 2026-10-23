---
name: tabular-inspection
description: Inspect CSV or XLSX columns, missing values and duplicate identities before preparing medical Cases.
---

Use `medbenchagent.data.tabular.inspect_table(path, sheet=..., id_columns=[...], preview_rows=...)`
to measure fields, missing values and duplicate IDs. Select identity columns from the source dictionary,
not from apparent uniqueness alone. `max_rows` bounds a scan; report the returned completeness and row
count without presenting a partial scan as full coverage. XLSX inspection uses openpyxl in the Coding environment.

Use pandas or openpyxl directly for transformations. Keep Case granularity explicit, preserve missing
labels and source row/column references, and do not infer patient or study relationships from filenames.
Retain the transformation script and parameters with the candidate. Produce the delegated Profile or
TaskData contract and let Data Processing validate and publish it.
