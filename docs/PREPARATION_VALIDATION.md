# Preparation contracts

Understand produces mapping.json (image rules, optional exact overrides, record_sources and annotation semantics) and typed image_conversion_plan.json. Image rules are expanded by code from the frozen source inventory, with series directories derived from file paths; source roots are not rescanned and models do not enumerate images. Worker checks read the same inventory, mounted read-only in Docker.

Text writes final locators into Cases. Ordinary images reference original absolute source files without copies. Converted locators use Prepared/<dataset>/images/<source-policy-slice identity>.png. The shared resolver separates mounted read paths from host locators. Text writes runtime work/accounting.json; Host independently counts declared records and checks seen/consumed/excluded/skipped/emitted totals, allowing joins and splits.

Image is skipped when the validated cases.jsonl references no image locator with a convert rule, even if unused conversion rules exist. Otherwise it emits only required converted PNG candidates and image_manifest.jsonl. Package validates candidates and atomically installs fixed images without overwriting conflicting content. Immutable revisions contain only cases.jsonl, metadata.json and image_manifest.jsonl; metadata binds direct source contents, while the manifest binds converted contents. No images are copied into revisions. Source files must remain available and unchanged.

Workers and Host use the same read-only stage checks. Package performs final consistency checks; Publisher commits unchanged validated metadata. Runtime owns scripts, accounting, logs and semantic plans. Existing runs are not migrated.
