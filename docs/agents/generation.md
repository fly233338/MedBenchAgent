# Generation 执行契约（后端后续接入）

当前实现 generation.contracts.GenerationSpec 和 validate_generation_spec，尚未注册 Generation
Agent、prepare_task_execution 工具或生成后端。closed_vqa_template 是已定义的配置标识，
不能作为后端已实现的声明。旧 generation.items 仍服务旧入口。

GenerationSpec 绑定 TaskPlan Artifact ID/digest、task_id/task_revision、DatasetRef/digest，
以及可选已接受 derived_data ID/digest。首个配置限定 closed_vqa / label：
case_ids、输入 EvidenceSelection、reference_field、identity/label_map 参考规则、
templates、options、sample_count 和 seed。

确定性校验检查当前 TaskPlan、任务未排除且语义 feasible、Dataset 和派生数据身份、
输入选择与任务证据语义一致、模型输入明确 model_visible、reference 不混入输入，
参考值落在闭集选项内。模板只允许 options、task_objective 插值。
sample_count 是目标数量，本轮不执行采样，也不声称能够达到目标。

后续 prepare_task_execution 将接受 Generation 提出的具体配置，执行校验并发布正式配置。
模板、选项、绑定或采样修订必须实际改变配置；自由文本只解释修改原因。
语义问题返回 Supervisor → Planner，缺少数据表示返回数据需求；后端不支持应报告能力缺口。

Review 报告必须绑定准确的候选和依赖身份；提交候选前执行必需的确定性 Audit。
Delivery 核验计划、数据、候选、报告和真实数量，并产生可独立读取的包。
这些运行能力尚未迁入新 Compilation。
