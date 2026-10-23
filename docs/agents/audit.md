# Audit

确定性 audit_candidate 复用已绑定的 GenerationSpec、TaskPlan、TaskData 重建题目，检查结构、
引用、输入/reference 边界、答案一致性、重复和数量。报告绑定候选内容，不能套用在新候选上。
它不声称完成医学语义判断，也不使用另一个模型循环。
