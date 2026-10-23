# Generation

generate 绑定已发布的 Leaf、accepted TaskData、必要需求和原始 benchmark_request。
prepare_task_execution 保存实际执行参数；Allocation 按真实容量分配，backend 确定性组题。

支持 closed/open VQA、报告、定位、检测；输入是有序 asset/text/value 集合，没有全局单图或 PNG/JPEG 限制。
input/reference 分离和答案/模板检查复用后端逻辑。复杂计算和原始数据转换交给 DP。

候选保留失败、隔离、实际数量和 shortfall。选中的批次不能有重叠 attempted task/Case 对，
包括失败尝试。语义修订回 Planner，缺少表示回 DP；不补造答案或降低 requested_count。
