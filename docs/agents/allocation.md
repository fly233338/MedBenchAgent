# Allocation

generation.execution.allocate_counts 提供无 BIR/模型依赖的确定性分配。
Generation 提供执行配置对应的真实容量和绑定 benchmark_request 的原始数量目标；
按任务均衡分配，在各任务容量内继续分配剩余数量。

Allocation 保存目标引用、requested、counts 和 capacities。容量不足不修改 requested，
最终候选和交付报告计算实际 shortfall。没有独立 Allocation Agent 或额外分配状态库。
旧 planning.allocation 仍服务旧入口，清理留待下一轮。
