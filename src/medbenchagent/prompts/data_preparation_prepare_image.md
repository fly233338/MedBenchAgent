Consume accepted mapping, typed conversion plan and cases.jsonl. Do not change Case semantics.
Only materialize referenced convert images; never copy direct source images. Resolve targets with preparation.images.resolve_image.
Write candidate PNGs under work/outputs/images/<final basename>.png using data.asset_adapters.materialize_image and accepted policy.
Write image_manifest.jsonl only for converted outputs. ImageEntry.path is the final host locator, not the candidate path; record actual digest, dimensions and source identity/digest.
Host validates and atomically installs files into fixed Prepared/<dataset>/images/. Do not write directly to final destinations.
Cases already contain reference mask features. Missing window/slice semantics requires clarification. No sidecar manifests.
