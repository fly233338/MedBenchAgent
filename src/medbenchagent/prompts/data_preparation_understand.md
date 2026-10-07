Understand the dataset once. Output only mapping.json and image_conversion_plan.json.
Use image_rules (source_id, pattern/path_field, source_format, action, policy_id) and rare exact source_id:path image_overrides. Do not enumerate images.
Direct ordinary JPEG/PNG/WebP images keep original source locators. Convert medical inputs to PNG; reference_only masks remain hidden.
Declare record_sources for independent runtime accounting (CSV/TSV, Excel sheet, JSONL, or JSON array; records_field selects a nested JSON array), Case identity, joins, fields and missingness.
Conversion policies use typed PresentationSpec: explicit axis/indices for volumes and required windows; no invented orientation fields.
Do not invent output facts. Inspection/scripts stay in work/. Ask for missing semantic decisions; preserve unknown versus negative.
