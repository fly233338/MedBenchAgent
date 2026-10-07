Consume accepted mapping, conversion plan and cases.jsonl. Do not change Case semantics.
For each non-mask image locator referenced by Cases, prepare that exact path under work/outputs/images/.
Use data.asset_adapters.materialize_image: preserve ordinary JPEG/PNG bytes without transforms; medical displays use PNG.
Write one image_manifest.jsonl with ImageEntry records keyed by final path, measured output digest/dimensions and bound source identity/digest.
Do not emit asset/presentation/request/mask-feature sidecars. Cases already contain reference mask features.
Use explicit conversion decisions; missing window or slice semantics requires clarification, not guessing. Validate actual outputs before submission.
