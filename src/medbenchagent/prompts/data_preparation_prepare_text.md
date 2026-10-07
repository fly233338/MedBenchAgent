Implement accepted mapping over the complete annotation population. Output cases.jsonl.
Use preparation.images.resolve_image with mounted source_roots, host locator_roots and bundle_root from context; never serialize container paths.
Asset Evidence.value is the final locator: original source for direct images, fixed Prepared/<dataset>/images/<identity>.png for conversions.
Use source path fields when rules specify path_field. The shared resolver accepts the resulting source-relative path and optional slice index.
Preserve field and missingness semantics; converted files need not exist yet. Embed digest-bound reference mask geometry directly in Evidence.mask.
Use preparation.accounting.TextAccounting to iterate records, consume/exclude/skip each one, emit Cases and write work/accounting.json. Do not put accounting in outputs.
Optional metadata.json may contain sampling/split using existing models. Perform necessary checks, then finish; Host validates the outputs.
