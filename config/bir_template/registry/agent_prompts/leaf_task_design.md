Design complete Leaf Tasks for the current Scenario. Use explicit model-visible inputs, a valid resolver, the required output contract, equivalent question_templates, and complete options when the operation guidance requires Closed VQA. Keep reference evidence out of model inputs.

Design a clinically meaningful question before selecting its output format.

Classify the operation by the primary task requested of the model,
not by the annotation field type or the surface form of the answer.

Specify the target entity, property, and scope. Every requested fact
must be supported by the permitted input and covered by the reference.

Use medical knowledge to interpret documented concepts and formulate
appropriate questions, but never to invent case-specific facts.

Do not infer absence from missing annotations. Do not request an
exhaustive description or count unless annotation coverage supports it.

For describe, require a coherent paragraph or structured report that
integrates multiple supported observations. Do not expand a single
categorical label into a narrative.

Question variants must preserve the same target, scope, comparison
direction, uncertainty, units, and answer requirements.

First define the clinical question, target entity, property, and scope before selecting the resolver and output format. Do not mechanically turn every annotation field into a task.
Do not add unannotated negative findings, diagnoses, or management recommendations.
The input and reference scopes must match: a local image must not be paired with a full examination report as its answer; a lesion annotation must not be used to ask about all abnormalities; and a lesion must be specified before asking about its margins or size.
