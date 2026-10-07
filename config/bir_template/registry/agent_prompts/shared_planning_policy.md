## Prepared Evidence Constraints

- Treat only explicitly registered and validated derived features as usable Resolver references. The presence of a mask, label, region, calibration field, or numeric value does not by itself establish localization, quantification, comparison, or another capability.
- Keep primary model-visible image/volume Evidence separate from reference-only mask, label, region, calibration, and derived Evidence. Never use filenames, asset IDs, patient IDs, or hidden metadata as shortcuts.
- Quantitative targets require an explicit subject, scope, unit, or documented dimensionless definition. Do not invent thresholds, conversions, aggregations, or class mappings.
- Multi-view, volume, and ordered-panel tasks require stable view identity, documented correspondence, and a frozen presentation/sampling order. An index refers only to that registered order.
- Missing annotation is not negative. Empty masks, tied maxima, missing planes, missing units, and incomplete manifests are unsupported and must be excluded rather than replaced with 0, an empty box, or `N/A`.
- Prefer direct or derived references whose coverage, scope, unit, relation, and input/reference role are explicit and executable.


Every task has at least one verified ordinary image presentation. Two Evidence IDs for one Asset do not establish two images. Multiple images do not establish multi-view semantics. Hidden spacing cannot justify an image-only physical measurement.
