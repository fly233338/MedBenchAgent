---
id: ultrasound-spatial-calibration
description: Hidden pixel spacing metadata cannot support image-only physical measurements.
stages:
  - scenario_screening
  - leaf_task_design
  - scenario_review
scope: modality
diseases: []
modalities:
  - ultrasound
---

A per-image pixel spacing value used only for reference construction is calibration metadata, not model-visible input. Do not create cm or cm² prediction tasks unless physical scale is visible or explicitly included in the permitted input.
