## Prepared Evidence Constraints

- Use only registered prepared Evidence and explicitly validated derived features when judging task support or constructing Resolver references. The presence of a mask, label, region, calibration value, or numeric field does not by itself establish localization, quantification, comparison, or another capability.
- Keep model-visible image Evidence separate from reference-only masks, labels, regions, calibration, metadata, and derived reference features. Reference-only information may support ground-truth resolution but must not become evaluated-model input.
- Quantitative targets require an explicit subject, scope, unit, or documented dimensionless definition. Do not invent thresholds, unit conversions, aggregations, class mappings, or measurements.
- Multi-image, multi-view, volume-derived, and ordered-input tasks require the actual selected model inputs to have the documented relationships and ordering required by the task. Multi-view additionally requires explicit view identity and correspondence.
- Missing annotation is not negative. Unknown, missing, unannotated, and explicitly negative states remain distinct. Empty masks, tied maxima, unavailable planes, missing units, and incomplete required evidence must not be silently replaced with `0`, an empty box, or `N/A`.
- Prefer direct references or registered deterministic derivations whose coverage, scope, unit, relation, and model-input/reference role are explicit and executable.
- Every benchmark task must resolve to at least one verified model-visible ordinary image input. Two Evidence entries that refer to the same underlying source image do not establish two distinct image inputs.
- Multiple images do not by themselves establish multi-view semantics. A shared patient ID, source directory, or Case membership is insufficient without the relationship required by the active input-structure definition.
- Hidden spatial calibration cannot justify an image-only physical measurement unless the effective benchmark input policy makes the required calibration information available to the evaluated model.
