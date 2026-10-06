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
- 候选输出包含 canonical Case、Asset、provenance、执行报告和 output manifest。
  `submission.json` 记录 coding agent 的实际提交状态；它不替代宿主验收回执。
- 独立重复执行比较总体、记录身份、输出内容和质量账本。

## 发布前检查

- 验证来源路径、记录 ID、finding 关系、字段映射、带 digest 的 mask 几何、展示 manifest 和采样顺序。
- 统计保留记录、整图隔离和等价重复；Profile、Card 和 Catalog 绑定最终 Bundle。
- 仅接受通过完整验证的候选；不确定的总体选择进入明确的用户审核。
- 数据准备不决定 benchmark 题目；Task Planning 将完整 Evidence Catalog 作为事实清单，
  再按输入结构和医学语义判断是否支持 Axis。

## 恢复

准备状态、阶段 Worker attempt、工具操作和 LangGraph checkpoint 都可以从新 Run 目录恢复。
每个 attempt 保存 `inputs/context.json`、`inputs/prompt.md`、`state/request.json`、`state/result.json` 和
输出文件；不再维护通用 memory、receipt 或 resume 文件。只有同一 stage、同一 attempt、同一 input digest 且状态仍为 running 时才续接 conversation；新的
retry 或 mapping revision 创建新的 attempt identity。成功回执和文件 digest 有效时可以复用，重复恢复不会重新发布相同内容。

详细字段契约见 [DATA_LAYER.md](DATA_LAYER.md)。
# annotation validation

准备候选必须包含 `cases.jsonl`、`asset_manifest.jsonl`、`provenance.jsonl` 和
`preparation_report.json`。宿主逐 Case 验证 Evidence ID、ref、Asset 引用、source locator 和
digest，并完成完整 population 账本。单个坏 Case 隔离并记录原因；映射、Schema 或 Adapter
级别的系统性错误阻止发布。准备阶段不生成任务、答案或模型输入选择。

若候选提供 presentation、split 或 sampling manifest，宿主同时验证记录引用、输出文件 digest、
Case/Asset/view identity 和冻结顺序。缺失或不一致的 manifest 不能用于 view-dependent task。


## Field Semantics for Planning

Prepared mapping evidence may include a `field_semantics` section for retained fields. Each entry
records source-backed `semantic_role`, `value_kind`, `scope`, `unit`, `input_visibility`, coverage,
and missingness. Planning treats an absent semantic role as `unknown`; it never infers a clinical
numeric reference from storage type or aggregate counts. Calibration-only and reference-only fields
remain unavailable to an image-only benchmark model.

## Canonical validation and publication

There are two host gates. `validate_handoff()` accepts an Understand candidate, checks the source report, population, mapping and questions, and performs one independent inspector replay. `validate_prepare_candidate()` runs candidate validation and population reconciliation once, then writes the report used by publication. `AcceptedHandoff` is created only after the first gate. `DatasetPublisher` consumes the second gate receipt, checks the candidate digest, materializes valid Cases and their reference closure, and advances the immutable revision and `current.json` pointer. It does not invoke a second validator.
