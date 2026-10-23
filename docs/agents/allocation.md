# Allocation

Generation 的确定性工具按选定 Spec 的实际 Case 容量平衡分配原始 requested_count。
容量不足保留目标与真实 shortfall，不通过重复模板或批次凑数。Allocation 绑定 benchmark_request，
没有隐藏的模型偏好解析或旧 BIR 依赖。
