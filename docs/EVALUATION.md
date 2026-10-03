# Model inference and evaluation overlay

Evaluation is provisioned separately on the model-serving host. The main
`medbenchagent` package no longer includes or maintains the evaluator; place the
ignored repository-root `evaluation/` overlay beside the checkout before running
these commands.

The evaluation loader accepts `benchmark-manifest` and the `asset_manifest.jsonl`. Legacy
legacy construction artifacts are outside the supported workflow.

The evaluation overlay runs generated benchmark artifacts against local Medical
VLM checkpoints. Use the audited JSONL when available. The loader requires the
matching `benchmark_manifest.json` and the registered dataset's
`asset_manifest.jsonl`; it validates task and asset contracts and resolves every asset by
`asset_id`. Source paths stored in manifests are never
used for inference.

Run evaluation in a separate Conda environment rather than the project's locked
`uv` environment:

```bash
conda create -n medbench-eval python=3.12 -y
conda activate medbench-eval
python -m pip install -r requirements/evaluation.txt
```

If the server needs a CUDA-specific PyTorch wheel, install the matching `torch`
and `torchvision` build first, then install the requirements file. The repository
does not assume a CUDA version. The evaluation environment does not need the full
MedBenchAgent dependency set or an editable package install.

To use FlashAttention-2, install a build compatible with the active PyTorch and
server CUDA toolchain after the base requirements:

```bash
python -m pip install packaging psutil ninja
python -m pip install flash-attn --no-build-isolation
```

Keep `flash-attn` outside the base requirements because its build is specific to
the server GPU, CUDA, and PyTorch versions. Use the CLI's default SDPA when
FlashAttention-2 is unavailable.

From the repository root, verify the source modules directly:

```bash
PYTHONPATH=src:. python -m evaluation.inference --help
PYTHONPATH=src:. python -m evaluation.eval --help
```

Supported adapter keys are:

| Adapter key | Model family | Parsed bbox coordinate space |
| --- | --- | --- |
| `breastgpt_8b` | BreastGPT-8B | `[0,1000]` |
| `qwen2_5_vl` | Backward-compatible Qwen2.5-VL key | Processed Qwen image |
| `qwen2_5_vl_7b_instruct` | Qwen2.5-VL-7B-Instruct | Processed Qwen image |
| `qwen2_5_vl_32b_instruct` | Qwen2.5-VL-32B-Instruct | Processed Qwen image |
| `internvl3_8b` | InternVL3-8B | `[0,1000]` |
| `internvl3_5_8b` | InternVL3.5-8B | `[0,1000]` |
| `qwen3_vl_8b_instruct` | Qwen3-VL-8B-Instruct | Fixed `[0,1000]` |
| `qwen3_vl_8b_thinking` | Qwen3-VL-8B-Thinking | Fixed `[0,1000]` |
| `qwen3_vl_32b_instruct` | Qwen3-VL-32B-Instruct | Fixed `[0,1000]` |
| `lingshu_7b` | Lingshu-7B | Processed Qwen image |
| `lingshu_32b` | Lingshu-32B | Processed Qwen image |
| `hulu_med_7b` | Hulu-Med-7B | Original PNG |
| `hulu_med_32b` | Hulu-Med-32B | Original PNG |
| `qoq_med_vl_7b` | QoQ-Med-VL-7B | Processed Qwen image |
| `qoq_med_vl_32b` | QoQ-Med-VL-32B | Processed Qwen image |
| `chiron_o1_8b` | Chiron-o1-8B | `[0,1000]` |
| `medgemma_1_5_4b_it` | MedGemma-1.5-4B-IT | Normalized `[0,1]` yxyx, converted to xyxy |

BreastGPT requires the local ModelScope checkpoint directory, including its bundled
`plugin/` package. Download the complete snapshot before running inference:

```python
from modelscope import snapshot_download

snapshot_download("YYangYang/BreastGPT-8B", local_dir="/path/to/BreastGPT-8B")
```

Then use `--adapter breastgpt_8b --model-path /path/to/BreastGPT-8B`. The adapter loads
the checkpoint's official `breastgpt` model and template registrations through
MS-SWIFT. Its legacy `<|box_start|>...<|box_end|>` 2D grounding output is normalized
to the evaluator's single-box contract while the complete response remains available
as `raw_prediction`.

InternVL3-8B limits dynamic preprocessing to six aspect-ratio tiles and keeps the
official thumbnail when multiple tiles are used, for at most seven `448x448`
inputs per image. Other InternVL-family adapters retain the official limit of 12.

MedGemma is a gated checkpoint. Accept Google's Health AI Developer Foundations
terms before downloading it, then point the adapter at the local model directory.
MedGemma keeps the shared task prompt for non-bbox tasks and appends a model-specific
requirement to end with `FINAL_ANSWER:`. Its adapter prefers that boundary and otherwise
recognizes the
checkpoint-observed `answer is`, `category is X.`, Markdown option, and `\boxed{X}`
letter forms. For bbox tasks it follows MedGemma's native localization protocol: one
JSON-list object with normalized `[y0,x0,y1,x1]` `box_2d` coordinates. The adapter
strictly validates `[0,1]`, converts yxyx to canonical xyxy, and then scales to the
original image. Conflicting or malformed candidates remain parse errors.

All bbox prompts state the coordinate protocol owned by their adapter family. Qwen3,
InternVL, Chiron, and BreastGPT request fixed `[0,1000]` xyxy coordinates; Qwen2.5,
Lingshu, and QoQ request absolute xyxy coordinates in the resized image processed by
the model; MedGemma requests normalized `[0,1]` yxyx coordinates; and Hulu requests
original-image pixel coordinates. BreastGPT requests its native grounding-token format
through the bundled MS-SWIFT template and reports grounding in `[0,1000]`.
QoQ uses its native medical
reasoning prompt and extracts the final balanced `\boxed{...}` payload. Qwen3
Thinking extracts the suffix after the last `</think>`. InternVL3.5 uses explicit
`<think>...</think>` reasoning and `<answer>...</answer>` final-answer sections. Chiron
preserves its checkpoint system message, appends the official step-by-step trigger,
requires the `### The final answer is:` boundary, and extracts the suffix after its
last occurrence. The extractor also accepts the checkpoint-observed form without
the leading Markdown heading, `The final answer is:`. Hulu uses its processor's
`use_think=False` decode as the answer. Full reasoning remains in `raw_prediction`; an absent
official separator falls back to the full response, while an empty extracted
answer fails strict parsing. Letter responses may be one bare valid letter, a
parenthesized letter such as `(A)`, `(B) DENSITY B`, `(B. Asymmetry)`, or `（A）`, the Lingshu-style form with one
trailing ASCII period such as `A.`, or a valid letter
and period followed by option text or an explanation such as `A. Benign` or
`A. because ...`. Multiple letters without this prefix form and other wrappers
remain parse errors. Free text must be non-empty, and bbox responses must contain exactly
one distinct box. The bbox parser accepts `bbox` or `bbox_2d` fields in JSON, Markdown,
or surrounding text, plus a bare four-number list or tuple when it is the complete answer.
Repeated identical boxes are deduplicated; zero or multiple distinct boxes remain parse
errors before the existing coordinate-space checks and rescaling.

The InternVL3.5 adapter installs a model-specific reasoning system prompt based on
InternVL's optional Thinking Mode protocol. It requires all reasoning inside
`<think>...</think>` and the task-constrained final answer inside
`<answer>...</answer>`. The complete response remains in `raw_prediction`; the adapter
extracts the last complete answer section and retains the prior `</think>` suffix as a
legacy fallback. InternVL3 remains on the shared direct-answer prompt and behavior.
Chiron and InternVL3.5 explicitly
set `pad_token_id` to the tokenizer EOS ID to avoid the open-end generation warning.

For a single model/dataset combination, run one process per GPU with the current
Conda environment's Python:

```bash
PYTHONPATH=src:. python -m torch.distributed.run \
  --standalone \
  --nproc-per-node 4 \
  --module evaluation.inference \
  --adapter qwen2_5_vl_7b_instruct \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --benchmark /path/to/benchmark.jsonl \
  --benchmark-manifest /path/to/benchmark_manifest.json \
  --asset-manifest /path/to/asset_manifest.jsonl \
  --output-specs label labels number text bbox index \
  --output results/qwen2_5_vl_7b_instruct.json \
  --batch-size 4

PYTHONPATH=src:. python -m evaluation.eval \
  --input results/qwen2_5_vl_7b_instruct.json \
  --output results/qwen2_5_vl_7b_instruct.report.json
```

Each process loads one complete model on its `LOCAL_RANK`, processes a
round-robin item shard, and writes a temporary rank file. Rank zero creates the
ordered inference JSON after every rank succeeds. Completion is coordinated by
per-rank files rather than a post-inference NCCL barrier, so uneven inference times
are not limited by NCCL's default 10-minute process-group timeout. Generation is
greedy, with
defaults of 512 new tokens, SDPA, and 0.5 correctness thresholds for both BLEU
and bbox IoU. Existing inference outputs and temporary parts are protected
unless `--overwrite` is supplied; interrupted jobs do not resume. `--task-types`
is required, accepts one or more unique values from `label`, `bbox`, and `text`,
and filters samples before model loading. Every requested type must be present in the
benchmark. The selected order is recorded in inference metadata.
Each inference item also records the effective model `coordinate_size` used by the bbox
parser. This field is optional in the backward-compatible `medbench-inference`
schema, so older files remain readable.

If an existing inference JSON contains BreastGPT label outputs such as `（A）`,
the model does not need to be run again. Recompute the report directly from the
stored `raw_prediction` values:

```bash
PYTHONPATH=src:. python -m evaluation.eval \
  --input results/breastgpt_8b.json \
  --output results/breastgpt_8b.reparsed.report.json \
  --reparse
```

`--reparse` only re-extracts letter answers and recalculates their scores; bbox and
free-text results continue to use their recorded predictions. The original inference
JSON is not modified, and no checkpoint, GPU, or benchmark/image manifest is needed.

To recover bbox predictions from an existing inference file, reparse its stored raw
responses with the recorded coordinate space:

```bash
PYTHONPATH=src:. python -m evaluation.eval \
  --input results/qwen3_vl_8b_instruct.json \
  --output results/qwen3_vl_8b_instruct.reparsed.report.json \
  --reparse-bbox
```

`--reparse-bbox` prefers each item's `coordinate_size`. Legacy Qwen3-VL files may omit
that field because their fixed `1000x1000` protocol is recoverable from adapter metadata;
Qwen3 Thinking reparsing uses only the final answer after the last `</think>`. Older
files from dynamic-coordinate adapters must contain `coordinate_size` and otherwise
fail explicitly instead of guessing a protocol. Reparsing only writes the new report
and never modifies the inference JSON.

For MedGemma, bbox reparsing applies the corrected native normalized-yxyx protocol even
when an older inference item recorded the former, incorrect `896x896` coordinate space.

The CLI default remains SDPA for portability. Pass
`--attn-implementation flash_attention_2` to opt in. PyTorch SDPA may itself
select a fused CUDA attention kernel, so compare end-to-end throughput on the
target server rather than assuming a fixed speed ratio. The repository ignores
the entire `scripts/` directory; local batch automation and server paths are not
tracked by Git.

Each torchrun rank loads a complete checkpoint. `NPROC_PER_NODE` adds independent
data-parallel replicas and does not shard a 32B model across GPUs. A 32B BF16
checkpoint therefore requires a GPU that can hold the complete model plus runtime
memory; multi-GPU tensor parallelism and quantization are outside this evaluator.

The evaluator recomputes item scores and checks the recorded result. One report file
is a response-format-keyed object whose selected `label`, `bbox`, and `text`
values are complete `medbench-evaluation` reports. Each section independently
contains `metadata`, `summary`, `metric_groups`, `categories`, `tasks`, and `items`.
It reports accuracy, semantic-option macro-F1, SacreBLEU, and mean IoU according to each
question type's manifest metric. For label and text sections, `metric_groups` keeps
the existing two hierarchical display values. For bbox sections it instead exposes the
native `acc_at_iou` and `miou` names: each question type first contributes its threshold
accuracy and mean IoU to an equal-weight mean within its parent task, then the parent-task
values are averaged with equal weight. Optional
`--bleu-threshold` and `--bbox-iou-threshold` values recompute report
correctness without rerunning the model. The report's per-item `items` list
retains both the normalized `prediction` and the model's complete
`raw_prediction` for output auditing.

For bbox sections, summary, category, task, and metric-group records use `miou` and
`acc_at_iou`; per-item records use `iou`. Acc@IoU uses the report's configurable
`bbox_iou` threshold (0.5 by default). This single-box output contract has no confidence
score, so the evaluator does not claim standard AP/mAP.

Every letter task also records semantic-option `macro_f1`, regardless of its
primary metric. Free-text tasks record `macro_f1: null`; bbox sections omit macro-F1
entirely. Category and summary macro-F1 values average the available letter-task
macro-F1 values and are `null` when no letter task is present.

Model integrations live under `evaluation.models`. Add one model
module implementing `ModelAdapter.generate`, then add its explicit name and
factory to the registry in that package. Adapters own checkpoint loading, chat
templates, batch preprocessing, decoding, and any model-specific coordinate
space; common parsing and scoring remain model-independent.

