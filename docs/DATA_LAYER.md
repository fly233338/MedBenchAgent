# Data layer contracts

数据层把只读来源转换为可查询、可追溯的 Dataset Bundle。LangGraph 负责协调，数据模块负责确定性执行和验证。

## 来源与准备

每个逻辑数据集由一个或多个 `Source` 组成。Source 保存稳定 ID、根目录、类型和可选 annotation 文件。准备阶段记录来源清单、文件摘要、总体数量、字段语义、适配器和执行回执。

OpenHands Coding Session 只能在 Worker workspace 中写候选文件。宿主验证来源 locator、记录身份、图像路径、finding 关系、缺失值和完整总体后，才发布 Dataset、Profile、Card 和 Catalog 条目。

## Dataset Bundle

已发布 Bundle 提供（annotation.v5）：

- `annotation.jsonl`：以 Case/Evidence 为核心的 canonical 记录；
- `asset_manifest.jsonl`：image/volume Asset、来源和完整性摘要；
- `preparation_report.json`：完整输入、接收、隔离和失败账本；
- Profile：字段、结构、覆盖率、值分布和容量索引；
- Card：数据集范围、模态、字段角色和来源证据；
- preparation metadata：执行计划、来源摘要、验证回执和 provenance。

Bundle、Profile、Card 和 Catalog 都使用内容摘要绑定输入。新版本写入新的目录，已登记版本保持可复用。

## 执行与验证

固定算子和 generated Python adapter 使用相同的输入、输出和 provenance 契约。Adapter 输出候选 annotation 与 provenance；它不发布 namespace、可信 provenance、Catalog 或完成标记。

总体检查器生成完整 Case/Asset population。框架逐项对账候选记录，验证 source locator、Evidence ref、Asset digest、重复身份和缺失资产。影响医学语义完整性的错误按整 Case 隔离，数量账本区分接收、隔离和可选 Evidence 丢弃。

Asset 处理保留原始来源引用。PNG/JPEG、单文件 DICOM、DICOM Series 和 NIfTI 通过统一 Asset Adapter 读取。Native Asset 与确定性的 Presentation 分离，显示窗口、维度和像素元数据作为独立决策保存。

## Harmonization 与查询

多来源字段通过显式 harmonization plan 合并、保留或排除。值映射、缺失状态、数组边界和来源空间都保留在计划和 provenance 中。

`DatasetQuery` 绑定一个 access manifest 和 cache revision，提供字段值、来源覆盖、记录采样、数组长度、配对和协议容量查询。查询返回准确总数和分页结果，供规划 Agent 使用。

## 运行契约

准备和构建使用 Run 冻结的配置、Catalog snapshot、Card 副本、事件日志和 operation ledger。产物依赖由 `runtime/dependencies.py` 校验；工具调用按参数和依赖幂等重放。

新运行存储在 `.runtime/<run>/control`、`inputs`、`agents` 和 `workers` 下。现有旧运行目录不属于当前运行时。
