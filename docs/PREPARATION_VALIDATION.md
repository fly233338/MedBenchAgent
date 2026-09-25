# 准备主链验证清单

本次按要求不新增或修改测试代码。下列内容用于新契约的后续测试迁移与验收；不能将本清单视为已执行证明。

## 契约与事实

- Fingerprint 只来自当前 Source 的真实检查；文档、用户回答、未知或跨 Source 回执不能伪装为全量事实。
- 全量计数、抽样元数据与有界预览区分覆盖范围和分母；追加无关 inspection 不使 Spec 失效。
- Spec 对四类身份、总体、缺失、finding 配对与空间含义完整记录；解释性决策、custom 和编码转换需要精确证据。
- 无依据新增输出字段、未映射值、重复目标、非 canonical target、代码提前赋 namespace/可信 provenance 被拒。
- Source、所选事实、语义或代码修订只使对应 Source 的下游失效；旧文件不覆盖。

## 两类 executor 与 locator

- 单表：字符串 ID 前导零、空值、阴性、unknown、显式筛选和实际 unique_images。
- 多表：复合键、支持基数、空键/未匹配策略、many-to-many 膨胀、相同字段名和来源行合并。
- 同一数据分别走 built_in/generated_python，比较 canonical 结果、总体与质量账本，不要求执行步骤编号相同。
- 一图零/一/多 finding：逐对象 locator；标签/框来自不同对象时拒绝；不只删除坏 finding。
- 表格 locator：CSV 引号内换行、工作表和 delimiter、不存在逻辑行、Parquet/Excel 行序；逻辑行不能使用 DuckDB 更新后的物理 rowid。
- JSON：对象/数组集合、嵌套 Pointer、~0/~1、错误索引、JSONL 记录内定位、64 MiB JSON 限制与大型 JSONL 流式扫描。
- mask：空 foreground 结果、指定不存在 instance、多标签、非二维、mask/image 尺寸不一致、不允许 resize。
- 不存在源文件/对象、重复或未使用 provenance、重复 record_ref、漏掉 finding locator 均不能进入 Annotation。
- 生成脚本可直接选择；语法错误可修正；没有 fallback_reason 要求；未知二进制格式缺少 reader 时阻断。

## 总体、质量与人工回答

- 对两类 executor 检查遗漏、额外图像、相同数量不同成员、重复 ID/路径、ID→路径错配。
- 独立总体来自来源检查，不由脚本输出或重复执行结果反推。
- 只有固定总体检查器不支持所需操作时才可记录能力缺口；业务失败、缺依赖、缺语义证据和对账错误不能替代能力判断。
- 人工审核前校验全部候选的 locator 和清单；提供完整文件及分页。明确批准才允许继续，拒绝需要修订，含糊回答继续澄清。
- 问题与批准绑定 Source/Fingerprint/Spec/ExecutionPlan/CandidateExtraction；其中任一变化后旧批准失效。
- 人工审核标记只表示总体选择审核，完整性未独立证明；不得豁免其他检查。
- 坏图、缺图、无效标注、bbox 越界、维度不支持按整图隔离；重复身份、来源引用、mask 对应几何错误阻断提取。
- 缺 decoder、磁盘写入失败和运行环境故障不是坏图；无可用图像不发布。
- 候选数 = 最终保留 + 整图隔离 + 等价重复移除；上游筛选另记，不能更改预期总体抵消损失。

## 恢复、环境与发布

- Docker 使用固定镜像、只读输入和禁网；Spec、输入说明和 provenance 也通过只读挂载传入。
- 显式 local 可执行 Python；auto→local 在脚本启动前拒绝，并要求显式选择和新运行；中途失败不降级。
- 两次输出内容一致；时间戳、随机顺序和外部变化导致差异时拒绝；源变化在执行前后及发布前检查。
- 完成回执存在但候选被改写、丢失或部分生成时不得复用；不完整目录保留，下一次尝试使用干净目录。
- 各阶段与完成标记/checkpoint 写回之间中断；恢复不重复发布、不改写已发布准备索引。
- Canonical 导入自动通过同一候选验收；原 image manifest 位置有效，保证仅覆盖输入 annotation。
- 多 Source 依赖不交叉覆盖；全部有效后才 harmonize；未对齐字段保留来源空间，不补阴性。
- 新 preparation sidecar 的内容、索引或引用损坏时 admission/reuse 拒绝；旧 bundle 不原地补证。
- 最终 plan_ref 指向 bundle 内 execution_plan.json，继续能查 Spec、Adapter、完整总体和用户审核记录。
- Profile、Card、Catalog、DatasetQuery 与现有 benchmark 下游保持读取接口；旧 checkpoint 明确要求新运行。

## 执行方式与边界

- 回归命令：`uv run ruff check .`、`uv run pytest`。
- 现有直接调用 `save_preparation_plan`、`write_fallback` 或断言旧 Mapping 的测试需要迁移到新工具和契约。
- 新测试应围绕行为与实际风险，不重复测试每个简单包装。
- 真实 Docker、实际数据集、模型决策与规模吞吐应单独集成验收；本地子进程样例和 mock 不替代这些结果。
- 模型解释、人工总体选择审核和自定义 Python 聚合均有明确保证边界，不作为医学语义的形式化证明。
