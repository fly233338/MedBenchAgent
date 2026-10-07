# Data layer contracts

数据层把只读来源转换为可查询、可追溯的 Dataset Bundle。LangGraph 负责协调，数据模块负责确定性执行和验证。

## 数据集内容与派生视图

Case/Evidence 的 asset value 直接保存最终普通图片相对路径；image_manifest.jsonl 以相同路径为键，
记录实测内容摘要、格式、尺寸以及来源身份。不存在 asset → request → presentation 的解析链。
Mask 是附带来源 digest 的 Case/reference feature，不强制生成 mask 文件或单独 feature manifest。

Understand 只冻结 mapping 和 image_conversion_plan。Text 生成 Cases，Image 生成实际图片和唯一 manifest。
验证只读；最后组包 cross-check 后 Publisher 原子提交，不修改 Case、不隔离或修复 manifest。
Published revision 为 cases.jsonl、image_manifest.jsonl、images/ 和必要的 metadata.json。
metadata 保留字段语义与采样/分组顺序；mapping、转换策略和观测材料留在 runtime。

Profile、Card、DuckDB/access 缓存在 revision 外的 .derived/，可删除重建。DatasetAccess 始终绑定
immutable revision 的 cases/image manifest，根目录 current.json 仅作为指针，不保存图片 projection。

## 多来源映射与查询

多来源字段通过 understand 阶段的 accepted mapping 合并、保留或排除。值映射、缺失状态、数组边界和来源空间都保留在 mapping、provenance 和运行诊断中。

`DatasetQuery` 绑定一个 access manifest 和 cache revision，提供字段值、来源覆盖、记录采样、数组长度、配对和协议容量查询。查询返回准确总数和分页结果，供规划 Agent 使用。

Evidence Catalog 是完整数据事实清单，不是任务候选清单。Task Planning 需要区分图像可观察证据、患者/临床上下文、设备/行政元数据和仅用于校准的 spacing 等字段；字段存在本身不能支持一个 Axis 或 Leaf Task。

规划阶段不会把 access manifest 和全部 Case 内容直接交给模型。宿主先生成
`DatasetAccessReceipt`，再生成 Dataset Overview、Evidence Catalog、Relation
Summary 和有界 Case Samples。每个规划证据文件记录 dataset revision 与输入
digest；读取工具只返回分页或当前 Axis/Scenario 所需的切片。

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

准备和构建使用 Run 冻结的配置、Catalog snapshot、Card 副本、事件日志和 operation ledger。产物依赖由 `runtime/dependencies.py` 校验；工具调用按参数和依赖幂等重放。

实验级 BIR 位于 `experiments/<experiment>/BIR/`，保存 Dataset Selection、冻结的
`DatasetAccessReceipt` 和规划 Evidence 的 canonical 引用。Run 只保存 BIR 路径、revision
和摘要；Evidence、规划和交付阶段通过 BIRStore 读取并以 revision/digest 校验写回。

新运行存储在 `.runtime/<run>/control`、`inputs`、`agents` 和 `workers` 下。现有旧运行目录不属于当前运行时。
