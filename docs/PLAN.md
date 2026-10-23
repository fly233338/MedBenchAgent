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

## 上轮验证记录（841ea33）

只扩展两个能力场景：真实图/Fake Model 的完整交接与中断恢复，以及审查绑定、局部 Leaf 修订、
数据复用和重新生成/审查/交付。保留并调整直接受影响的 SDK、Supervisor、契约和入口检查。

该轮未运行全量 pytest，未执行真实模型、真实 OpenHands 编程、医学全量样本或吞吐验收。
离线图贯通证明接口和执行路径可用，不证明模型能稳定自主选择全部步骤。

收口结果：
- 18 项直接相关检查通过；局部返工补强后只复测两个能力场景，均通过。
- Ruff、内部模块导入、11 个 Prompt 组合加载通过。
- 测试收集 46 项，无悬空导入；没有执行全量测试。
- CLI 与模块入口帮助、TUI 核心界面及共用 Runner 空 Checkpoint 恢复 smoke 通过。
- Studio 工厂离线装配及输入 Schema 检查通过；未启动真实 Studio 服务运行模型。
- Wheel 构建成功，文档相对链接与 git diff --check 通过。

旧运行格式不迁移；已有正式文件仍保留在原目录。

## 遗留清理与最终收口（2026-10-23）

在 refactor@841ea33 上完成以下清理，分模块迁移、契约删除、依赖文档和验收记录四个提交：

- Supervisor、Coding 适配器与 Worker 迁入 agents；Assembly 注入 DP，Worker 使用新模块入口。
  删除旧模块与失效 Dockerfile，没有转发模块。
- 删除旧来源目录索引、孤立 sampling/tables 工具、contained_path、单文件 freeze_prompt。
  保留文件/目录摘要、医学读取/转换、呈现与几何能力。
- 删除 ImageEntry、重复静态 annotation Schema；资产统一使用 AssetEntry，模型可见 Schema 来自当前类型。
  metadata.semantic_role 不再接受 role 别名解释。
- BenchmarkItem 删除 sampling_unit_id、split、view_ids、panel_id、sampling_order、input_bindings；
  保留实际生产的有序 inputs、模板选择与溯源字段，Evidence.view_id 和 Dataset Sampling/Split 引用保持有效。
- 删除 Run 的三个单值实现选择器和不可达 Scripted 分支；保留专业角色启停与测试注入。
  DP/Planner/Generation/Review 只接收 operation 指令映射，Supervisor/Coding 使用组合文本。
- 评估侧只保留现行交付包 loader；删除旧指标配置及专属测试、历史汇总脚本和外部评估依赖。
  EVALUATION 文档只说明当前交付契约和外部评估边界。
- 主环境移除无用途的直接依赖；RAG 使用 rag extra，医学读取保留 data-processing extra，
  openpyxl 归开发依赖，rich 明确为直接依赖。保留 Coding 独立环境依赖，Ruff 目标改为 py312。
  锁文件保留包的版本均未变化，移除 duckdb、pandas、pytz、tzdata、xlrd。

### 实际验收

- `uv run --locked --extra rag --extra data-processing pytest -q`：48 项通过，无跳过；
  包括完整协作、中断恢复、Coding 重放、局部修订、领域工具、RAG、CLI/TUI 和 Studio 离线装配。
  pynrrd 产生一条上游 datetime.utcnow 弃用警告。
- 扩展交付检查后，单独复测迁移读取、题目/资源篡改拒绝和局部返工场景，通过。
- 新增真实 Worker 子进程入口检查，以故意不匹配的依赖版本验证结构化失败回执，未调用 SDK 模型。
- 运行配置拒绝已删除字段；实现指纹不匹配时拒绝恢复且不改写文件；冻结指令恢复不读取当前 Prompt。
- Ruff、源码 compileall、受维护 Markdown 相对链接与 git diff --check 通过。
- `uv build` 从 sdist 构建 wheel 成功；包内保留 13 份 Prompt、11 个组合，没有已删除模块和静态 Schema。
- 临时干净环境按锁文件安装基础依赖与 wheel：56 个核心模块导入、两个 CLI 帮助入口、Prompt 加载和
  Worker 子进程回执检查通过。该环境没有 chromadb、pypdf、pandas、duckdb、openpyxl、pytest 或 OpenHands。

本轮未运行真实模型协作、真实 OpenHands 数据处理或完整医学数据验收。已有用户数据、Checkpoint 和
正式产物文件未清除；旧 Run 和旧格式不迁移，也不承诺旧格式可由当前 Reader 读取。

## 后续

使用实际配置观察小规模自主协作，依据具体失败改进 Prompt；按实际数据需求补充表示能力。
不提前建设通用筛选 DSL、复杂调度框架或全面质量验证平台。

## Prompt 来源

包资源共 13 份：common、五个模型角色与 Coding、DP/Planner 四个操作片段、Coding 交接和 structured repair。
实际冻结 11 个调用组合，repair 片段随适用组合加载。工具说明保留在唯一工具定义旁；动态请求、
类型生成的 Schema、查询结果及 Host 错误由代码装配。旧 Registry Prompt 与 Preparation 指令已移除。
本轮沿用现有 prompt-authoring skill，保留 AGENTS.md，不建立额外开发期角色体系。
