Current stage: prepare_image

Implement accepted image_conversion_plan.json for accepted image_requests.jsonl.
Do not change Case semantics, requests, paths, identities or sampling order. Read accepted Text files in inputs/accepted/.
Use data.presentation.materialize_presentation and shared Asset adapters. Preserve native JPEG/PNG bytes when no transform is needed;
medical sources, slices and montages produce real PNG. Missing window/slice semantics require clarification.
Write asset_manifest.jsonl, presentation_manifest.jsonl, images/ and mask_features.jsonl (empty if no masks).
Mask feature rows contain case_id, evidence_id and mask (MaskFeatureSet); use data.mask_geometry and bind source digest.
Reference masks never become model-visible images. Asset metadata must retain semantic_role and input_visibility.
Scripts, sample checks and execution logs belong in work/. Validate outputs before submission.
Run python -m medbenchagent.preparation.cli validate-prepare-image --context inputs/context.json --candidate outputs --report work/validation.json.
