---
id: breast-cancer-invalid-task-examples
description: Concrete invalid examples collected from breast ultrasound planning failures.
stages:
  - scenario_screening
  - leaf_task_design
  - scenario_review
scope: example
diseases:
  - breast_cancer
modalities:
  - ultrasound
---

Invalid: “Quantify the BI-RADS score from 1 to 6.” BI-RADS is an ordinal `assessment`, so the valid operation is recognition/classification.

Invalid: “Predict lesion diameter in cm” from image-only input when `Pixel_size` is hidden reference metadata. The physical calibration is not identifiable to the evaluated model.
