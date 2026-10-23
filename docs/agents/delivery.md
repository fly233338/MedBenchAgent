# Delivery

publish 是确定性节点，核验选定候选、Audit、Review 与原始数量目标的精确绑定，重算 Audit，
写入独立包及 complete.json 成员摘要，并通过 evaluation.loader 重读。

包包含题目、有序资源引用、隔离记录、任务计划、执行规格、审查报告和实际数量说明。
非空受限交付遵循冻结 goal_spec.allow_limited_delivery，保留 requested_count 并说明限制；
空候选、阻断问题或错内容批准不能发布。交付操作 completed 与最终目标 partial 可以同时成立。
export 复制包内所有被摘要绑定的文件，不重新生成数据。
