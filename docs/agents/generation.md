# Generation Agent 与后端

agents.generation 通过 create_agent 接入；generation.backends / execution 实现确定性业务操作。
Generation 的直接输入是 TaskPlan 中可执行 Leaf 及其已接受 TaskData，不读取原始目录推断任务。

## 工具

查询后端、TaskData、批次/候选样本、失败与隔离记录；prepare_task_execution 发布执行配置；
allocate_generation_budget 分配真实容量；generate_task_batch 生成指定批次；
quarantine_generated_items 记录隔离理由；finalize_generation 提交候选。
propose_data_requirement 可发布执行性缺口，返回 Supervisor 协调。

Allocation 必须绑定 benchmark_request，保留原始数量目标。容量不足体现为 shortfall。
相同任务和 Case 不通过模板变化重复计数；重新执行时显式选择替换批次，不能合并重叠尝试。

## 当前实现的后端

| Backend | Task Family | 输出 |
| --- | --- | --- |
| vqa | closed_vqa / open_vqa | label、labels、number、text、index |
| report_generation | report_generation | text |
| visual_grounding | visual_grounding | bbox |
| detection | detection | detections |

后端不按原始文件后缀或图像数量分派。输入字段可含多个有序资源、文本和数值；
资源引用由 TaskDataReader 校验，原始解析和转换由 Data Processing 负责。
这表示能够组装已准备的数据，不表示后端能自行解码所有医学格式。

空间 reference 是带 asset_id、box、coordinate_system 的对象；检测是额外带 label 的对象列表。
坐标系统支持 normalized_xyxy、pixel_xyxy、physical_xyzxyz，物理坐标需要 unit。
所有目标必须锚定实际输入资产。复杂派生值由数据处理预先计算。

## 执行与产物

GenerationSpec 绑定 TaskPlan/Leaf 修订、TaskData、Case 范围、sample_count 和 seed。
Leaf 持有模板、字段、选项及答案规则。后端按固定种子选择 Case/模板，产生 BenchmarkItem。
题目使用有序 inputs，每项有 kind（asset/text/value）、role 和 value；asset value 带实际文件位置和摘要。
输入资源可以是多种文件格式，题目不再要求 input_images。

批次记录成功题目和 CaseFailure；候选包含实际 items.jsonl、quarantine.jsonl、成员摘要、
执行配置/批次引用、数量目标和真实统计。候选发布不代表已经通过审查或交付。
