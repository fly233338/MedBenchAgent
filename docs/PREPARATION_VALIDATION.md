# 准备主链验收

准备主链的目标是把来源事实、语义决定、执行候选和最终 Bundle 分开保存，并让每个发布结论可复核。

## 来源事实

- 记录完整来源清单、文件摘要、表头、图像数量和总体覆盖范围。
- 每条 annotation、finding、mask 和图像路径都绑定 source-relative locator。
- 用户决议和医学语义解释保存引用、作用范围和影响的字段。
- 字段处置区分 Case 医学证据、分组/来源/资产元数据和排除的运营/行政噪声；年龄等医学上下文
  可以保留在 `cases.jsonl`，设备序列号、UID、软件版本等无任务价值字段应按来源证据排除，
  并在 `mapping.json` 与 `notes/mapping.md` 中记录理由。

## 候选与执行

- 固定算子和 generated adapter 都读取冻结输入并写入 Worker workspace。
- 候选输出包含 annotation、image population、provenance 和完成回执。
- 独立重复执行比较总体、记录身份、输出内容和质量账本。

## 发布前检查

- 验证来源路径、记录 ID、finding 关系、字段映射、mask 几何和图像 manifest。
- 统计保留记录、整图隔离和等价重复；Profile、Card 和 Catalog 绑定最终 Bundle。
- 仅接受通过完整验证的候选；不确定的总体选择进入明确的用户审核。
- 数据准备不决定 benchmark 题目；Task Planning 将完整 Evidence Catalog 作为事实清单，
  再按输入结构和医学语义判断是否支持 Axis。

## 恢复

准备状态、Worker 会话、工具操作和 LangGraph checkpoint 都可以从新 Run 目录恢复。已完成的候选和发布 Bundle 保持内容摘要，重复恢复不会重新发布相同版本。

详细字段契约见 [DATA_LAYER.md](DATA_LAYER.md)。
# annotation validation

准备候选必须包含 `cases.jsonl`、`asset_manifest.jsonl`、`provenance.jsonl` 和
`preparation_report.json`。宿主逐 Case 验证 Evidence ID、ref、Asset 引用、source locator 和
digest，并完成完整 population 账本。单个坏 Case 隔离并记录原因；映射、Schema 或 Adapter
级别的系统性错误阻止发布。准备阶段不生成任务、答案或模型输入选择。

