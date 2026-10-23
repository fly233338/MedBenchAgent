---
name: mask-geometry
description: Measure mask geometry with explicit foreground rules and preserve image association uncertainty.
---

Use `medbenchagent.data.mask_geometry.extract_mask_features` and `geometry_digest` for deterministic
mask measurements. Inspect their signature before calling and supply the source digest, channel and
foreground values explicitly. Preserve native shape, returned bounding boxes, connected components
and indexed-plane metadata. Supply spacing and physical units only when supported by source evidence.

Geometry does not prove what the mask represents or which image it labels. Verify the image association
from source evidence before constructing grounding or detection references. Keep ambiguous masks
unbound and report a concrete data requirement. Masks and derived reference geometry must not become
model-visible inputs to the task they answer. Retain parameters and measured outputs with the candidate.
