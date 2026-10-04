---
id: breast-cancer-pathology-basis
description: Keep pathology diagnosis separate from imaging assessment and context.
stages:
  - axis_analysis
  - scenario_screening
  - leaf_task_design
  - scenario_review
scope: disease
diseases:
  - breast_cancer
modalities: []
---

A pathology or disease label is a `diagnosis` target only when the dataset documents its diagnostic basis. Do not relabel BI-RADS or a morphological descriptor as pathology diagnosis.
