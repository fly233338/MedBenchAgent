Assess the supplied Axis using only the bounded evidence. Return one decision for every allowed value, with evidence_basis and a concise reason. Apply the image-only benchmark policy. Do not construct Scenarios or Leaf Tasks.

Supported means that at least one clinically meaningful task is supported by the permitted input and has verifiable reference evidence. Do not mark a value supported only because a field exists. Distinguish insufficient evidence from explicit lack of support.

The host supplies `describe_synthesis_allowed` and `describe_support_rule`. Follow them exactly.

When `describe_synthesis_allowed` is false, require a direct report-level or structured narrative reference. Do not create a narrative by combining separate categorical values, text annotations, region annotations, metadata, or evidence counts.

When `describe_synthesis_allowed` is true, apply the supplied target, scope, reference, and answer requirements before combining evidence.
