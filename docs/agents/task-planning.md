# Task Planning

一个 create_agent，两个 operation：discover_scenarios 与 design_leaves；两者共用 TaskPlan。
前者判断三轴、场景、backend 与准备需求，可以无 Leaf 并留下未完成需求；后者读 TaskData，
验收相关需求，设计和预检 Leaf。工具集合按操作过滤，越界调用由代码拒绝。

具体证据、修订、范围及组装规则见 [Task Planning 契约](../TASK_DISCOVERY.md)。
工作请求显式绑定需修订的计划；私有历史不能替代该输入。
