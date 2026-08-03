CUDA_VISBLE_DEVICES=1,2
uv run torchrun \
  --standalone \
  --nproc-per-node=2 \
  --module medbenchagent.evaluation.inference \
  --adapter qwen2_5_vl \
  --model-path /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct \
  --benchmark /public/home/rxa-jj-wzk/projects/fyl/medbenchagent/benchmarks/runs/e619f930-fd27-424a-9c4f-56af08d3bc29/benchmark.breast-us.audited.jsonl \
  --benchmark-manifest /public/home/rxa-jj-wzk/projects/fyl/medbenchagent/benchmarks/runs/e619f930-fd27-424a-9c4f-56af08d3bc29/benchmark_manifest.json \
  --image-manifest datasets/Normalized/BrEaST/image_manifest.jsonl \
  --output results/qwen2_5_vl_breast_us.json \
  --batch-size 8 \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --overwrite