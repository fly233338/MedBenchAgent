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

先明确临床问题、目标对象、属性和范围，再选择 resolver 与输出形式。不能把每个 annotation 字段机械改成一道题。
要求整合多个可验证事实；参考仅支持单个类别时，禁止设计段落描述。
不得添加未标注的阴性所见、诊断或管理建议。
“输入和参考范围一致”尤其重要：给一张局部图，却拿完整检查报告作答案；给一个病灶标注，却问所有异常；没有指定病灶，却问其边界或大小——这些任务即使 JSON、resolver 和选项全部合法，临床上仍然不成立。
