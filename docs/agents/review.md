# Review

review_plan 与 review_candidate 对精确绑定的目标只读，写自己的发现和覆盖报告。
候选审查先运行完整确定性 Audit，再检查每个生成任务的实际题目样本和需要的源证据。

approved 针对整个绑定目标；未审查任务/场景写入 unchecked_scope 并保持 false。
抽查每个任务的部分题目与完全没审查某个任务不同。拒绝批准仍可 completed。
发现记录对象、依据、阻断性质及建议责任方，Supervisor 据此发起局部返工。
