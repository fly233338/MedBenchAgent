#!/usr/bin/env bash

# Activate the evaluation Conda environment before running this script.
CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1
BATCH_SIZE=1
MAX_NEW_TOKENS=512
DTYPE=bfloat16
ATTN_IMPLEMENTATION=flash_attention_2
BLEU_THRESHOLD=0.5
BBOX_IOU_THRESHOLD=0.5
OUTPUT_ROOT=results/evaluation
OVERWRITE=0

QWEN2_5_VL_7B_INSTRUCT_NAME=qwen2_5_vl_7b_instruct
QWEN2_5_VL_7B_INSTRUCT_ADAPTER=qwen2_5_vl_7b_instruct
QWEN2_5_VL_7B_INSTRUCT_PATH=/public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct

QWEN2_5_VL_32B_INSTRUCT_NAME=qwen2_5_vl_32b_instruct
QWEN2_5_VL_32B_INSTRUCT_ADAPTER=qwen2_5_vl_32b_instruct
QWEN2_5_VL_32B_INSTRUCT_PATH=/public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-32B-Instruct

INTERNVL3_5_8B_NAME=internvl3_5_8b
INTERNVL3_5_8B_ADAPTER=internvl3_5_8b
INTERNVL3_5_8B_PATH=/public/home/rxa-jj-wzk/models/InternVL3.5/InternVL3_5-8B

QWEN3_VL_8B_INSTRUCT_NAME=qwen3_vl_8b_instruct
QWEN3_VL_8B_INSTRUCT_ADAPTER=qwen3_vl_8b_instruct
QWEN3_VL_8B_INSTRUCT_PATH=/public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-8B-Instruct

QWEN3_VL_32B_INSTRUCT_NAME=qwen3_vl_32b_instruct
QWEN3_VL_32B_INSTRUCT_ADAPTER=qwen3_vl_32b_instruct
QWEN3_VL_32B_INSTRUCT_PATH=/public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-32B-Instruct

QWEN3_VL_8B_THINKING_NAME=qwen3_vl_8b_thinking
QWEN3_VL_8B_THINKING_ADAPTER=qwen3_vl_8b_thinking
QWEN3_VL_8B_THINKING_PATH=/public/home/rxa-jj-wzk/models/Qwen3-VL-Thinking/Qwen3-VL-8B-Thinking

LINGSHU_7B_NAME=lingshu_7b
LINGSHU_7B_ADAPTER=lingshu_7b
LINGSHU_7B_PATH=/public/home/rxa-jj-wzk/models/Lingshu/Lingshu-7B

LINGSHU_32B_NAME=lingshu_32b
LINGSHU_32B_ADAPTER=lingshu_32b
LINGSHU_32B_PATH=/public/home/rxa-jj-wzk/models/Lingshu/Lingshu-32B

HULU_MED_7B_NAME=hulu_med_7b
HULU_MED_7B_ADAPTER=hulu_med_7b
HULU_MED_7B_PATH=/public/home/rxa-jj-wzk/models/Hulu-Med/Hulu-Med-7B

HULU_MED_32B_NAME=hulu_med_32b
HULU_MED_32B_ADAPTER=hulu_med_32b
HULU_MED_32B_PATH=/public/home/rxa-jj-wzk/models/Hulu-Med/Hulu-Med-32B

QOQ_MED_VL_7B_NAME=qoq_med_vl_7b
QOQ_MED_VL_7B_ADAPTER=qoq_med_vl_7b
QOQ_MED_VL_7B_PATH=/public/home/rxa-jj-wzk/models/QoQ-Med-VL/QoQ-Med-VL-7B

QOQ_MED_VL_32B_NAME=qoq_med_vl_32b
QOQ_MED_VL_32B_ADAPTER=qoq_med_vl_32b
QOQ_MED_VL_32B_PATH=/public/home/rxa-jj-wzk/models/QoQ-Med-VL/QoQ-Med-VL-32B

CHIRON_O1_8B_NAME=chiron_o1_8b
CHIRON_O1_8B_ADAPTER=chiron_o1_8b
CHIRON_O1_8B_PATH=/public/home/rxa-jj-wzk/models/Chiron-o1/Chiron-o1-8B

SELECTED_MODELS=(
  QWEN2_5_VL_7B_INSTRUCT
  QWEN3_VL_8B_INSTRUCT
  QWEN3_VL_8B_THINKING
  INTERNVL3_5_8B
  QWEN2_5_VL_32B_INSTRUCT
  QWEN3_VL_32B_INSTRUCT
  LINGSHU_7B
  HULU_MED_7B
  QOQ_MED_VL_7B
  CHIRON_O1_8B
  LINGSHU_32B
  HULU_MED_32B
  QOQ_MED_VL_32B
)

DATASET_1_NAME=dataset_1
DATASET_1_BENCHMARK=/path/to/dataset_1/benchmark.jsonl
DATASET_1_BENCHMARK_MANIFEST=/path/to/dataset_1/benchmark_manifest.json
DATASET_1_IMAGE_MANIFEST=/path/to/dataset_1/image_manifest.jsonl

DATASET_2_NAME=dataset_2
DATASET_2_BENCHMARK=/path/to/dataset_2/benchmark.jsonl
DATASET_2_BENCHMARK_MANIFEST=/path/to/dataset_2/benchmark_manifest.json
DATASET_2_IMAGE_MANIFEST=/path/to/dataset_2/image_manifest.jsonl

DATASET_3_NAME=dataset_3
DATASET_3_BENCHMARK=/path/to/dataset_3/benchmark.jsonl
DATASET_3_BENCHMARK_MANIFEST=/path/to/dataset_3/benchmark_manifest.json
DATASET_3_IMAGE_MANIFEST=/path/to/dataset_3/image_manifest.jsonl

DATASET_4_NAME=dataset_4
DATASET_4_BENCHMARK=/path/to/dataset_4/benchmark.jsonl
DATASET_4_BENCHMARK_MANIFEST=/path/to/dataset_4/benchmark_manifest.json
DATASET_4_IMAGE_MANIFEST=/path/to/dataset_4/image_manifest.jsonl

SELECTED_DATASETS=(
  DATASET_1
  DATASET_2
  DATASET_3
  DATASET_4
)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ "${OUTPUT_ROOT}" = /* ]]; then
  RESOLVED_OUTPUT_ROOT="${OUTPUT_ROOT}"
else
  RESOLVED_OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if (( ${#SELECTED_MODELS[@]} == 0 )); then
  echo "SELECTED_MODELS must contain at least one model prefix." >&2
  exit 1
fi
if (( ${#SELECTED_DATASETS[@]} == 0 )); then
  echo "SELECTED_DATASETS must contain at least one dataset prefix." >&2
  exit 1
fi
if [[ "${OVERWRITE}" != 0 && "${OVERWRITE}" != 1 ]]; then
  echo "OVERWRITE must be 0 or 1." >&2
  exit 1
fi

require_variable() {
  local variable_name="$1"
  if [[ ! -v "${variable_name}" ]]; then
    echo "Missing required variable: ${variable_name}" >&2
    exit 1
  fi
  local variable_value="${!variable_name}"
  if [[ -z "${variable_value}" ]]; then
    echo "Required variable is empty: ${variable_name}" >&2
    exit 1
  fi
}

for model_prefix in "${SELECTED_MODELS[@]}"; do
  for suffix in NAME ADAPTER PATH; do
    require_variable "${model_prefix}_${suffix}"
  done
  adapter_variable="${model_prefix}_ADAPTER"
  adapter="${!adapter_variable}"
  if ! python -c \
    'import sys; from medbenchagent.evaluation.models import ADAPTERS; raise SystemExit(0 if sys.argv[1] in ADAPTERS else 1)' \
    "${adapter}"; then
    echo "Unknown adapter key: ${adapter}" >&2
    exit 1
  fi
  model_path_variable="${model_prefix}_PATH"
  if [[ ! -d "${!model_path_variable}" ]]; then
    echo "Model directory does not exist: ${!model_path_variable}" >&2
    exit 1
  fi
done

for dataset_prefix in "${SELECTED_DATASETS[@]}"; do
  for suffix in NAME BENCHMARK BENCHMARK_MANIFEST IMAGE_MANIFEST; do
    require_variable "${dataset_prefix}_${suffix}"
  done
  for suffix in BENCHMARK BENCHMARK_MANIFEST IMAGE_MANIFEST; do
    path_variable="${dataset_prefix}_${suffix}"
    if [[ ! -f "${!path_variable}" ]]; then
      echo "Dataset input does not exist: ${!path_variable}" >&2
      exit 1
    fi
  done
done

mkdir -p "${RESOLVED_OUTPUT_ROOT}"

for model_prefix in "${SELECTED_MODELS[@]}"; do
  model_name_variable="${model_prefix}_NAME"
  adapter_variable="${model_prefix}_ADAPTER"
  model_path_variable="${model_prefix}_PATH"
  model_name="${!model_name_variable}"
  adapter="${!adapter_variable}"
  model_path="${!model_path_variable}"
  model_output_directory="${RESOLVED_OUTPUT_ROOT}/${model_name}"
  mkdir -p "${model_output_directory}"

  for dataset_prefix in "${SELECTED_DATASETS[@]}"; do
    dataset_name_variable="${dataset_prefix}_NAME"
    benchmark_variable="${dataset_prefix}_BENCHMARK"
    benchmark_manifest_variable="${dataset_prefix}_BENCHMARK_MANIFEST"
    image_manifest_variable="${dataset_prefix}_IMAGE_MANIFEST"
    dataset_name="${!dataset_name_variable}"
    benchmark="${!benchmark_variable}"
    benchmark_manifest="${!benchmark_manifest_variable}"
    image_manifest="${!image_manifest_variable}"
    inference_output="${model_output_directory}/${dataset_name}.inference.json"
    report_output="${model_output_directory}/${dataset_name}.report.json"

    echo "Model: ${model_name}"
    echo "Adapter: ${adapter}"
    echo "Dataset: ${dataset_name}"
    echo "Inference output: ${inference_output}"
    echo "Report output: ${report_output}"

    inference_command=(
      python -m torch.distributed.run
      --standalone
      --nproc-per-node "${NPROC_PER_NODE}"
      --module medbenchagent.evaluation.inference
      --adapter "${adapter}"
      --model-path "${model_path}"
      --benchmark "${benchmark}"
      --benchmark-manifest "${benchmark_manifest}"
      --image-manifest "${image_manifest}"
      --output "${inference_output}"
      --batch-size "${BATCH_SIZE}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --dtype "${DTYPE}"
      --attn-implementation "${ATTN_IMPLEMENTATION}"
      --bleu-threshold "${BLEU_THRESHOLD}"
      --bbox-iou-threshold "${BBOX_IOU_THRESHOLD}"
    )
    if [[ "${OVERWRITE}" == 1 ]]; then
      inference_command+=(--overwrite)
    fi
    "${inference_command[@]}"

    evaluation_command=(
      python -m medbenchagent.evaluation.eval
      --input "${inference_output}"
      --output "${report_output}"
      --bleu-threshold "${BLEU_THRESHOLD}"
      --bbox-iou-threshold "${BBOX_IOU_THRESHOLD}"
    )
    "${evaluation_command[@]}"
  done
done
