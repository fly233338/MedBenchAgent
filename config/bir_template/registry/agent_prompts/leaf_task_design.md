Design executable Leaf Tasks for the active Scenario using its evidence catalog and representative Cases.
Each Leaf Task must select exactly one item from the Scenario's `validated_clinical_targets`.
Keep its target name and clinical_target_id consistent. Use real Evidence names and explicit input/reference roles.
Resolver.inputs must equal Leaf.input. Every Resolver step must be registered and its result_step must identify an actual step.
Provide exactly the configured number of equivalent question templates. Templates must ask the same task without leaking references.
Closed VQA options must cover deterministic resolved labels; do not invent labels or add options to open answers.
Use explicit map_label only when justified by the accepted semantics. Missing values remain ineligible, not negative.
Use validated mask geometry as reference-only evidence. Multiple Evidence aliases of one source do not prove multiple independent images.
Capacity and image validity are computed by code over SamplingUnits. Repair reported implementation/contract errors without changing the intended semantics.
Avoid semantic duplicates of saved leaves. If no executable Leaf remains, explicitly exclude the Scenario with evidence.
Return the supplied schema; persistence and publication belong to host tools.

{{DATASET_SPECIFIC_RULES}}
