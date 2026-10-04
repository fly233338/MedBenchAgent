---
id: clinical-target-boundaries
description: Classify the clinical target by the requested answer and preserve atomic scope.
stages:
  - axis_analysis
  - scenario_screening
  - leaf_task_design
  - scenario_review
scope: common
diseases: []
modalities: []
---

The clinical target follows the requested prediction, not merely an entity mentioned in context. A standardized grade or risk category is `assessment`; a disease or pathology class is `diagnosis`; an observed sign is `finding`; a property of an entity is `attribute`; and a recommended next action is `management`.

Do not create an anatomy task from a fixed organ name shared by every Case. Keep the subject and scope explicit.
