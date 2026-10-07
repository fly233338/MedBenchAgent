Current stage: understand

Understand source semantics once. Produce mapping.json and image_conversion_plan.json (policies keyed by policy_id, plus notes).
Determine Case identity, joins, field meanings, missingness, Evidence relations, Asset roles, image selection and conversion policy.
The shared target_path function defines stable source/asset/view target paths. Ordinary JPEG/PNG are preserved without transforms;
medical sources need explicit window/axis/slice decisions and produce PNG. Do not predict output files, digests or dimensions.

Implement inspect_source.py with --input-config, --output-dir, --mode inventory. Enumerate the complete source population;
write image_population.jsonl and derive source_report.population using write_population_projection. Host replays this script.
Population identity is (source_id, source_unit_id); related volume and mask assets can belong to one logical unit.
Missing annotation is not negative. Do not infer clinical meaning, windows or spatial units without evidence.
Record blocking questions in semantic_questions.json; otherwise write an empty questions list.
Run the supplied validate-understand command, repair contract failures, then submit. Do not publish or convert images here.
