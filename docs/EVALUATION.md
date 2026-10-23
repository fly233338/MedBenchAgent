# Benchmark 交付与外部评估边界

Compiler 负责构建和交付 Benchmark。仓库内维护的评估侧接口只有
`medbenchagent.evaluation.loader.load_benchmark_package(directory)` 及其包完整性检查；
模型推理、指标选择、评分和模型部署由外部评估程序负责。

## 读取交付包

```python
from medbenchagent.evaluation.loader import load_benchmark_package

package = load_benchmark_package("path/to/delivery")
manifest = package["manifest"]
items = package["items"]
```

输入是已完成的交付目录或通过 `medd2b export` 导出的包。读取不需要构建 Run、Checkpoint、
原始数据目录或准备缓存。当前发布器产生 portable 包，搬移整个目录后仍可读取。

- `complete.json` 使用 `benchmark-delivery` 标识，绑定交付文件及资源摘要。
- `benchmark_manifest.json` 使用 `benchmark-manifest` 标识，记录任务身份、数量与交付限制。
- `benchmark.jsonl` 的每行必须符合当前 `BenchmarkItem`；不接受旧字段或历史格式。
- 包同时保留 Audit、Review、TaskPlan、GenerationSpec、隔离记录和编译报告。

Loader 检查完成标记、文件摘要、Benchmark 身份、题目唯一性、任务引用和数量。
每个资源输入也会检查实际文件及摘要；portable 路径不得逃出包目录。文件缺失或内容改变时拒绝读取。

## 外部程序的输入契约

每道题使用有序 `inputs`，条目类型为 asset、text 或 value。asset 值包含资源路径与摘要；
Loader 将其路径解析为当前文件的绝对路径。保留输入顺序、角色、多资产和文本/数值输入，
不假设所有任务只有一张 PNG/JPEG。来源路径属于溯源信息，不是推理输入位置。

外部程序根据 `task_paradigm`、`output_spec` 和实际 `answer` 定义评估方式，并记录自己的模型、
预处理与指标配置。Compiler 不包含评估配置类、模型适配器、推理 CLI、历史结果格式转换或评估环境依赖。

## 导出

```powershell
uv run --locked medd2b export PATH_TO_DELIVERY NEW_DESTINATION
```

导出前后都验证包，并且目标目录必须不存在。交付短缺和隔离数量保留在 manifest/report 中；
外部评估不得将有限交付误记为原请求已经全部完成。
