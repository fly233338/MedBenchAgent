---
id: numeric-and-derived
description: Prevent numeric storage types and unsupported derivations from opening quantification tasks.
stages:
  - axis_analysis
  - scenario_screening
  - leaf_task_design
scope: common
diseases: []
modalities: []
---

A numeric value or numeric category code is not automatically a clinical measurement. Quantify only an explicit quantity with entity, scope, unit where applicable, and a registered reference rule.

Derived references require every operand and transformation to be represented by the Resolver policy. Do not invent ratios, aggregations, thresholds, or measurements from field counts.
