# Data layer contracts

数据层把只读来源转换为可查询、可追溯的 Dataset Bundle。LangGraph 负责协调，数据模块负责确定性执行和验证。

## 数据集内容与派生视图

Case/Evidence 的 asset value 直接保存最终 locator。普通 JPEG/PNG/WebP 引用原 source，零复制。
转换图片固定保存于 Prepared/<dataset>/images/，image_manifest.jsonl 只登记这些实际 PNG。
没有 convert 规则时跳过 Image；混合数据只转换需要转换的部分。
Mask 是附带来源 digest 的 Case/reference feature，不进入模型输入。

Understand 冻结路径规则 mapping 和 typed conversion plan。Text 生成 Cases，并将 accounting 留在 runtime。
Published revision 只有 cases.jsonl、image_manifest.jsonl、metadata.json；不包含图片副本。
metadata 保存来源绑定和 direct 图片集合摘要、字段语义及 sampling/split。访问时验证来源和 converted 图片内容。

Profile、Card、DuckDB/access 缓存在 revision 外的 .derived/，可删除重建。DatasetAccess 始终绑定
immutable revision 的 cases/image manifest，根目录 current.json 仅作为指针，不保存图片 projection。

## 多来源映射与查询

多来源字段通过 understand 阶段的 accepted mapping 合并、保留或排除。值映射、缺失状态、数组边界和来源空间都保留在 mapping 和运行诊断中。

`DatasetIndex` 绑定一个扁平访问收据，提供 Case 读取、字段值分页、Asset 查询、数据事实和 Scenario 上下文。来源关系与结构容量从这些事实及 Resolver 推导，不保留旧协议查询入口。

Evidence Catalog 是完整数据事实清单，不是任务候选清单。Task Planning 需要区分图像可观察证据、患者/临床上下文、设备/行政元数据和仅用于校准的 spacing 等字段；字段存在本身不能支持一个 Axis 或 Leaf Task。

规划阶段不会把 access manifest 和全部 Case 内容直接交给模型。准备索引产生唯一的扁平
`DatasetAccessReceipt`，规划使用它生成 Dataset Overview、Evidence Catalog、Relation
Summary 和有界 Case Samples。每个规划证据文件记录 dataset revision 与输入
digest；读取工具只返回分页或当前 Axis/Scenario 所需的切片。

Receipt 在绑定和读取边界检查数据集身份、文件路径和内容摘要，所有消费者使用相同字段，不维护嵌套访问镜像。
case/patient 计数取自实际 profile，缺少信息时不以零或空路径代替；可选 mapping、Card、sampling/split 的路径与摘要必须成对出现。

Mask 的 bbox、面积、质心、component、等效直径和 plane feature 必须由准备阶段写入
Evidence，并绑定对应 Asset digest；Resolver 不读取原始 mask 文件。结构事实区分 primary
asset 才能证明 multi-image 或 multi-view；region→mask 引用不等于多视图关系。
NIfTI volume 保持原生 `[dim0, dim1, dim2]` shape，plane 固定使用 `axis=2`；bbox 使用
voxel-edge normalization，centroid 使用 voxel-center normalization，volume component 使用
26-connectivity，单 plane 使用 8-connectivity。spacing 和 physical measurement 只有来源单位
明确时才保留，未知单位保持缺失。
未知 role、visibility、关系或 manifest 顺序保持 unknown 并由规划代码保守排除，不能用字段数量或整数值猜测
index、时间或医学语义。

规划上下文可以引用准备阶段保存的 `mapping.json` 和语义证据。字段语义来自 metadata.json；宿主将字段含义、
关系说明、覆盖限制和代表性 Case 组织成有界结构化 `leaf_context`；缺失的语义保持
显式缺失。该上下文与 Evidence Catalog 一起传给 Leaf 设计，不依赖被截断的历史工具摘要。

## 运行契约

准备和构建使用 Run 冻结的配置、Catalog snapshot、Card 副本、事件日志和 operation ledger。业务依赖由 BIR canonical 文档校验；工具调用按参数和依赖幂等重放。

编译级 BIR 位于 `experiments/<session>/compilations/<id>/BIR/`，保存单数据集绑定、冻结的
`DatasetAccessReceipt` 和规划 Evidence 的 canonical 引用。Run 保存 BIR 路径、摘要和文档引用，
不维护全局 BIR revision；各业务文档通过 BIRStore 按自己的 revision/digest 校验写回。

绑定、规划、生成分别使用 BIR 根目录的 `dataset_binding.json`、`task_plan.json`、`generation.json`。
共享状态不维护按数据集分组的 Plan、访问数据或生成结果字典；批处理在 Session 中创建独立 Compilation。

新运行存储在 `.runtime/<run>/control`、`inputs`、`agents` 和 `preparation` 下。会话只持有编译引用；旧运行不迁移。
