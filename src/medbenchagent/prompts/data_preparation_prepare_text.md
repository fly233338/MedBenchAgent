Implement accepted mapping over the complete annotation population. Output cases.jsonl.
Asset Evidence.value is the final image locator from mapping.images, not a generated asset ID.
Use mapping field and missingness semantics without reinterpretation. Images need not exist yet.
Compute required reference mask geometry with data.mask_geometry and attach it directly to Evidence.mask with its source digest.
If sampling/split semantics are needed, write optional metadata.json containing sampling/split objects using existing manifest models.
Keep source accounting, scripts and debugging in work/; they are not dataset files. Validate schema, uniqueness, mapped coverage and locators.
