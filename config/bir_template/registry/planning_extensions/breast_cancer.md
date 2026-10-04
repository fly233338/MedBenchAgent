---
id: breast_cancer
type: disease
diseases:
  - breast_cancer
stages:
  - axis_analysis
  - scenario_screening
  - leaf_task_design
  - scenario_review
---

# Breast Cancer Task-Planning Rules

These rules provide additional constraints for breast-cancer imaging datasets.

They supplement the core task-planning prompt. They may narrow or clarify the supported task space but must not relax the common evidence-grounding, input-validity, Resolver, or output-schema requirements.

## BI-RADS

BI-RADS is a standardized imaging assessment.

A BI-RADS category should use the `assessment` clinical target.

BI-RADS categories encoded as numbers such as 1–6 remain ordinal categorical assessments. Their numeric representation does not support the `quantify` operation.

Do not describe prediction of a BI-RADS category as measurement or risk quantification merely because the output is numeric.

Do not invent alternative BI-RADS groupings such as:

- benign versus suspicious;
- low-risk versus high-risk;
- selected categories versus all remaining categories;

unless an explicit label mapping is supplied by the dataset or Registry.

BI-RADS should not automatically be interpreted as pathology diagnosis.

## Diagnosis

Breast-cancer diagnosis tasks require a documented diagnostic reference.

A diagnosis field may support targets such as benign versus malignant or a specific pathology type when its semantics establish it as an authoritative diagnostic label.

Do not derive diagnosis solely from:

- BI-RADS;
- shape;
- margin;
- echogenicity;
- posterior acoustic features;
- other imaging descriptors.

These may provide predictive image evidence for a diagnosis task, but they do not themselves constitute the diagnostic reference.

If both an imaging assessment and a pathology diagnosis exist, keep them as separate clinical targets.

## Management

Management includes next-step recommendations such as:

- routine follow-up;
- short-interval follow-up;
- additional imaging;
- biopsy;
- surgical referral;
- treatment-related action.

Management is supported only when an explicit recommendation or management reference is available.

Do not convert BI-RADS, malignancy diagnosis, or general guideline knowledge into a management target unless an explicit registered mapping permits that conversion.

For example, the existence of a BI-RADS category does not by itself authorize creation of a biopsy-versus-follow-up task.

## Findings and attributes

Use `finding` when the observed entity or sign itself is the requested clinical information.

Examples may include:

- lesion presence;
- calcification;
- halo sign;
- skin thickening;
- another explicitly annotated breast-imaging finding.

Use `attribute` when the annotation describes a property of an identified lesion or finding.

Examples may include:

- shape;
- margin;
- orientation;
- echogenicity;
- posterior acoustic behavior;
- composition;
- lesion size or another explicitly supported lesion property.

Do not duplicate the same descriptor as both a finding and an attribute merely to create additional Scenarios.

Follow the documented dataset semantics when a descriptor could plausibly belong to either category.

## Lesion masks and localization

A tumor or lesion mask may support localization when the registered output can be deterministically obtained from that spatial annotation.

Do not infer a more specific lesion identity from a generic mask name.

For example, an `other_lesion` mask does not automatically mean cyst, suspicious lesion, or another specific pathology unless that meaning is documented.

The existence of a mask does not automatically support area, maximum diameter, major axis, physical size, or other geometry-derived targets.

Such targets require corresponding registered Resolver operations.

## Quantitative measurements and spatial calibration

Pixel spacing, pixel size, or other spatial calibration may be useful for constructing a reference measurement.

However, reference availability is different from model-input availability.

For an image-only benchmark, do not construct tasks requiring centimeters, millimeters, square centimeters, or other physical units when the required case-specific calibration is hidden from the evaluated model.

A varying per-image calibration value must not be assumed to be recoverable from the image.

Do not infer support for lesion area, longest diameter, aspect ratio, or similar derived measurements solely from the existence of a segmentation mask.

The complete derivation must be supported by the registered Resolver.

## Scope

Keep breast-imaging annotations at their documented scope.

Do not silently reinterpret:

- Case-level assessment as a region-level annotation;
- pathology diagnosis as a lesion descriptor;
- lesion-level morphology as a whole-study quality label.

When multiple lesions or regions may exist, make sure the target and reference identify the same clinical entity.

## Conservative preference

For breast-cancer datasets, prefer direct and well-documented tasks such as:

- recognition of explicitly annotated findings;
- recognition of lesion attributes;
- prediction of explicitly annotated BI-RADS assessment;
- prediction of explicitly documented diagnosis;
- localization using explicit lesion-region annotations.

Do not create a derived task merely because it would be clinically interesting.

