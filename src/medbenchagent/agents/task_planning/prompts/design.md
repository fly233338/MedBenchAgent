# Operation: design_leaves
Design executable Leaves from the bound discovered plan and prepared TaskData. Inspect actual Cases,
coverage, source bindings, and requirements. Use accept_task_data and accept_data_requirement only
after the relevant content meets its acceptance conditions. Processor completion is not acceptance.

Define a focused medical task with a clear objective and capability_category. Specify LeafAssembly:
ordered input fields, question templates and bindings, reference selection, output meaning, and any
fixed options or label mapping. Use only declared visible fields plus supported reserved placeholders
in templates. Fixed answer options are allowed; per-Case reference answers must not enter the question.
Do not encode calculations or arbitrary code in templates. Request prepared derivations when needed.

For the initial strict contract, every Case assigned to the Scenario in the Leaf's exactly bound
TaskData must satisfy the Leaf's required fields, reference type, and label mapping. Zero Cases cannot
make an executable Leaf. Do not hide an
invalid subset through generation sampling or an informal filter. Return a precise data requirement
or explicit planning revision when the intended population needs correction.

Use save_semantic_task for definitions and exclude_semantic_task only with an evidenced semantic reason.
Preserve blocked but meaningful work. Publish the plan, distinguishing executable coverage from
unfinished scope. Generation will assemble your definition, not reinterpret it.
