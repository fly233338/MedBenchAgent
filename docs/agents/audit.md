# Audit

audit.generation 提供新候选的确定性检查，不新增模型 Agent。
针对确切 GenerationSpec、TaskPlan 和 TaskData 重建预期题目，检查：
- 对象、Case、执行配置与内容身份。
- 问题、输入顺序、reference、选项和溯源一致性。
- 输入/reference 分离及空间 reference 目标与坐标规则。
- 重复 task/Case 和同任务重复题目内容。
- 实际数量、分配配额及原始目标绑定。

AuditReport 绑定候选 Artifact ID/digest，记录检查数量、结果和可定位 findings。
Audit 保持候选只读；需要隔离或修改时由 Generation 形成新候选。
Delivery 再核对实际内容与该报告一致。旧 audit.task_items 保留给旧入口，待统一清理。
