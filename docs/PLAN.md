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

## Agent 模块化与领域能力扩展（2026-10-23）

基于 refactor@7a713a7，按五组提交完成多图修复、角色包迁移、原生 Skill 接线、目标/范围协作与离线验收。

- BrEaST 主图使用唯一 Evidence ID，病例标签关联图像集合。多主图 Mask 无明确映射时保持未绑定，
  Mapping 记录未知关系；单图 Mask 保留明确关联，缺失标签不补造。
- 每个角色拥有 agent/tools/资源；DP、Planner 的私有状态与辅助逻辑归 state，Delivery 为确定性 node。
  Coding 嵌入 DP，公共交接和组合器归 agents._shared。旧模块直接删除；迁移时核对 11 个 Prompt 组合文本一致，
  随后仅补充 Supervisor 的 readiness 查询说明。
- 三组包内 Skill 随 Run 冻结正文与随附资源，配置仅选择受维护名称；SDK 1.49.6 原生加载与调用，
  禁止自动加载公共/用户/项目 Skill 和记忆。允许列表、资源摘要或解析结果不一致时拒绝执行。
- data.tabular 提供实际 CSV/XLSX 统计与身份检查，data.sources 提供有界来源检查；影像和几何直接复用现有 API。
  未创建顶层 Tool Manager、全局 Skill 注册器或平行数据处理实现。
- Run 由目标和来源派生角色集合，只加载/发现必要模型。Profile 不初始化 Planner/Generation/Review，
  Canonical Scenario Plan 不初始化 DP/Coding；Executable Plan 不启用 Generation/Review/Delivery。
  原始来源启动时检查可读性，后续引用由 Reader 保持内容校验，不默认复制整份来源。
- 就绪建议按 Requirement、Scenario、Task 返回准确范围与依赖，并通过现有 Supervisor inspect 分页。
  Requirement 不展开同场景需求；Task 修订不授权修改同场景其他任务。建议与委派/完工检查使用同一契约。
- RAG 保持独立，文档明确现行 retrieve 调用与未来文档摘要/检索配置绑定要求，没有预建 Retriever 抽象。

### BrEaST 结构离线场景

动态生成四个 Case、多图、Mask 和一个缺失标签，使用真实 Graph/create_agent/工具与持久化 Coding 替身。
场景的预设决策只属于测试脚本，不写入生产协调图：

| 委派 | 范围及实际结果 |
| --- | --- |
| Profile → Scenario Discovery | 观察全部四个 Case，保留分类和空间两个方向及两个需求 |
| 分类准备 | 只处理 classification 需求，产生三条有标签的 TaskData，记录缺失标签 Case |
| 空间准备 | 只处理 spatial 需求；指定的多图 Case 缺少 Mask 映射，返回具体 blocker |
| Leaf Design → Generation | 只处理分类场景/任务；请求四题，实际候选三题，缺口一题 |
| Review → Leaf Repair | 审查拒绝措辞，只修订 classify Leaf，复用已接受 TaskData |
| Generation → Review → Delivery | 重新生成/批准后交付三题，保留空间需求，最终 partial |

准备后中断再恢复；禁止读取当前 Prompt/Skill 来源，重放已完成 SDK 请求不再次执行代码。
检查 Checkpoint 中全部 11 次 WorkRecord 的 operation、scope、输入、回执与阻塞原因；TaskData 引用和修改时间保持不变。

### 实际验收

- 全套轻量离线测试 61 项通过，无跳过；唯一 pytest 警告来自 pynrrd 的 datetime.utcnow 弃用。
- 后续补强的 Skill 解析失败、冻结资源、目标配置隔离及综合场景分别定向复测通过。
- 独立 .venv-coding 中运行真实 SDK 的 Skill 加载和 InvokeSkillExecutor，验证坏 Skill 不会静默缺失；
  同环境执行实际图像读取与 Mask 几何函数，不调用模型。
- Ruff、源码 compileall、文档相对链接及 git diff --check 通过。
- sdist → wheel 构建通过。修正了 Skill 目录重复 force-include；wheel 包含 13 份 Prompt、3 份 SKILL.md 和参考资源。
- 干净基础环境安装 wheel 后，72 个核心模块导入、11 个组合加载、Skill 冻结/验证和两个 CLI 帮助入口通过；
  该环境未安装 OpenHands、RAG、表格或测试可选依赖。SDK 专属检查在独立 Coding 环境执行。

边界：未调用真实模型、未开展真实自主 Coding 或全医学数据验收。旧运行不迁移；原有数据、Checkpoint 和正式产物未删除。

## TUI 用户入口与实时 Agent 活动（2026-10-23）

基于 refactor@e2d023e，参考 develop 的聊天、问答和事件筛选体验，按四组提交完成输入适配、活动接线、
单列工作台和恢复验收。没有引入旧 Workflow、旧来源发现状态文件、任务队列或新的执行账本。

- runtime.intake 管理 UI 草稿，辅助填写明确路径和数量；用户确认目标后自动生成 SourceManifest/Run。
  SourceManifest.key_files 保存已选来源内的相对文件引用和用途。目录可读性检查使用有界 scandir，
  输入检查与连接检测在后台执行，不做预先全量数据扫描。
- 首次建立默认实验，优先沿用显式模板、MEDBENCH_CONFIG 或已有 local.toml；连接向导列出模型、
  检查所需 Coding SDK 版本并选择 Skill。非敏感设置供新运行使用，密码输入只进入本进程环境。
- TUI 改为单列聊天、运行摘要和紧凑 Agent 表，Coding 位于 DP 下。活动详情可按角色筛选并查看
  有界工具参数、命令和输出；历史、产物、配置按需打开。保留阅读位置、未读提示、草稿和原地回答。
- 原生模型/工具 middleware 与 SDK callbacks 写入同一运行的诊断事件。事件关联请求、父 DP 请求与工具调用，
  过滤凭据、截断输出、不记录 SDK thought/reasoning。ActivityReader 使用字节游标、有界缓存和事件去重。
- 父图增加 paused，运行服务在保存边界更新状态并保留 next；等待实际 interrupt 和终止状态优先。
  运行锁与 Checkpoint 共同区分执行、暂停、等待、结束和无执行器的异常中断。问答按 interrupt ID 校验，
  回答只在 Checkpoint 接受后记入 UI 消息，不维护 pending_answer 执行状态。
- 交付数量来自正式产物，Review 拒绝与工作失败分别展示。日志不驱动恢复，修复身份、无进展算法和专业循环未改写。
- 删除原 tui.py 的品牌大图、常驻表单和原始 JSON 日志主界面；UI 分为 app/widgets/dialogs。
  wheel 打包现有 config/default.toml 与 Coding requirements，安装后可读取默认配置；没有第二份受维护配置。

### 离线验收

- `uv run --locked --extra rag --extra data-processing pytest -q`：71 项通过，无跳过；唯一警告为 pynrrd 的上游弃用提示。后续准备运行/Studio 边界和 UI 回归定向复测通过。
- TUI 端到端：自然语言输入含空格目录 → 确认 4 题/关键文件 → 长 Coding 调用期间看到活动 → 请求暂停 →
  保存后继续 → 聊天中回答 interrupt → 局部 Leaf 修订 → 交付 3/4、缺口 1、结果 partial。
  重开窄终端后仍显示结束结果，已完成 Coding 未重执行，回答未重复记录。
- 原生 SDK 1.49.6 回调测试使用真实 ActionEvent/ObservationEvent，检查父请求关联和推理/密钥过滤；不调用模型。
- 独立检查覆盖半行日志、重复读取、角色筛选、运行切换隔离、长输出、错误路径、空输入、多模型选择、
  缺失 Coding 环境、配置不保存密钥、首次启动配置优先级、暂停边界和过期回答拒绝。
- Textual Pilot 覆盖 80×24、100×35、140×45，验证输入框可见、历史阅读不跳动、活动详情与设置弹窗；
  对实际端到端界面导出 SVG 并渲染检查。修正了退出轮询访问卸载控件、结束时活动短暂清空等问题。
- Ruff、源码 compileall、sdist → wheel 构建、必要 CLI 入口和受维护 Markdown 链接检查通过。
- 干净基础环境安装 wheel，77 个核心模块与冻结资源可加载，80×24 TUI 和设置界面可启动；
  不依赖 OpenHands、RAG、openpyxl 或 pytest。默认配置与仓库源文件逐字节一致。

本轮仍为本地合成数据、真实 Graph/Fake Model 与 SDK 事件替身验证，未运行真实模型自主医学数据处理。
旧运行不迁移；既有数据、Checkpoint 和正式产物未删除。

## 后续

使用实际配置观察小规模自主协作，依据具体失败改进 Prompt；按实际数据需求补充表示能力。
不提前建设通用筛选 DSL、复杂调度框架或全面质量验证平台。

## Prompt 来源

包资源共 13 份：common、五个模型角色与 Coding、DP/Planner 四个操作片段、Coding 交接和 structured repair。
实际冻结 11 个调用组合，repair 片段随适用组合加载。工具说明保留在唯一工具定义旁；动态请求、
类型生成的 Schema、查询结果及 Host 错误由代码装配。旧 Registry Prompt 与 Preparation 指令已移除。
本轮沿用现有 prompt-authoring skill，保留 AGENTS.md，不建立额外开发期角色体系。
