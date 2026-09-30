# 准备主链验收

准备主链的目标是把来源事实、语义决定、执行候选和最终 Bundle 分开保存，并让每个发布结论可复核。

## 来源事实

- 记录完整来源清单、文件摘要、表头、图像数量和总体覆盖范围。
- 每条 annotation、finding、mask 和图像路径都绑定 source-relative locator。
- 用户决议和医学语义解释保存引用、作用范围和影响的字段。

## 候选与执行

- 固定算子和 generated adapter 都读取冻结输入并写入 Worker workspace。
- 候选输出包含 annotation、image population、provenance 和完成回执。
- 独立重复执行比较总体、记录身份、输出内容和质量账本。

## 发布前检查

- 验证来源路径、记录 ID、finding 关系、字段映射、mask 几何和图像 manifest。
- 统计保留记录、整图隔离和等价重复；Profile、Card 和 Catalog 绑定最终 Bundle。
- 仅接受通过完整验证的候选；不确定的总体选择进入明确的用户审核。

## 恢复

准备状态、Worker 会话、工具操作和 LangGraph checkpoint 都可以从新 Run 目录恢复。已完成的候选和发布 Bundle 保持内容摘要，重复恢复不会重新发布相同版本。

详细字段契约见 [DATA_LAYER.md](DATA_LAYER.md)。
