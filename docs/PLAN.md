# 目标协作流程与单一 Compilation 执行链

## 已实现

- WorkRequest/Delegate 显式 operation 和带产物归属的 scope；请求、完成结果、目标结果分别检查。
- DP profile/prepare_task_data；Planner discover_scenarios/design_leaves；Generation generate；Review 双操作；确定性 publish。
- 第一轮 Planner 可 completed 并返回待准备需求；设计完成按本次范围检查。
- 准备需求结构化字段/关系/粒度/最低覆盖；TaskData 绑定真实需求产物及内容。
- TaskData 验收证据使用对应 Reader；长 Case 支持 text_offset 续读。
- Leaf 非空、字段、答案/映射和空间元信息检查复用真实组题路径。
- 按操作限制工具；公共/角色/操作 Prompt 组合冻结；工作材料与静态指令分离。
- 有界修复、无进展停止、不可变产物及原生恢复；Finish 另行核验 succeeded/partial/blocked/failed。
- CLI、核心 TUI、Studio 共用图与 Assembly；登记/批量/导出均使用当前契约。
- 删除旧 Workflow/BIR/Resolver/Harness/Preparation、独立 Coding 发布包装、旧示例协议及对应配置/测试。
- 保留数据读取/转换、几何、源文件工具和独立 RAG；没有删除整个 data/generation/runtime 目录。

## 验证边界

只扩展两个能力场景：真实图/Fake Model 的完整交接与中断恢复，以及审查绑定、局部 Leaf 修订、
数据复用和重新生成/审查/交付。保留并调整直接受影响的 SDK、Supervisor、契约和入口检查。

本轮不运行全量 pytest，不执行真实模型、真实 OpenHands 编程、医学全量样本或吞吐验收。
离线图贯通证明接口和执行路径可用，不证明模型能稳定自主选择全部步骤。

收口结果：
- 18 项直接相关检查通过；局部返工补强后只复测两个能力场景，均通过。
- Ruff、内部模块导入、11 个 Prompt 组合加载通过。
- 测试收集 46 项，无悬空导入；没有执行全量测试。
- CLI 与模块入口帮助、TUI 核心界面及共用 Runner 空 Checkpoint 恢复 smoke 通过。
- Studio 工厂离线装配及输入 Schema 检查通过；未启动真实 Studio 服务运行模型。
- Wheel 构建成功，文档相对链接与 git diff --check 通过。

旧运行格式不迁移；已有正式文件仍保留在原目录。

## 后续

使用实际配置观察小规模自主协作，依据具体失败改进 Prompt；按实际数据需求补充表示能力。
不提前建设通用筛选 DSL、复杂调度框架或全面质量验证平台。

## Prompt 来源

包资源共 13 份：common、五个模型角色与 Coding、DP/Planner 四个操作片段、Coding 交接和 structured repair。
实际冻结 11 个调用组合，repair 片段随适用组合加载。工具说明保留在唯一工具定义旁；动态请求、
类型生成的 Schema、查询结果及 Host 错误由代码装配。旧 Registry Prompt 与 Preparation 指令已移除。
本轮沿用现有 prompt-authoring skill，保留 AGENTS.md，不建立额外开发期角色体系。
