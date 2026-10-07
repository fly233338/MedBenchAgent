Current stage: prepare_text

Implement the accepted mapping for the complete source population. Do not reinterpret semantics.
Write cases.jsonl, canonical provenance.jsonl and image_requests.jsonl; optional sampling/split manifests preserve explicit order.
Use data.presentation.target_path(source_id, asset_id, view_id, policy, source_format) for every request.
Read policies from accepted image_conversion_plan.json. Do not convert images or invent output measurements.
Source Asset identities remain stable. Reference-only masks require requests too. Write scripts/tests in work/.
Validate with preparation.assembly.validate_stage_outputs; report missing semantic facts through questions.
Run python -m medbenchagent.preparation.cli validate-prepare-text --context work/inputs/context.json --candidate work/outputs --report work/validation.json.
