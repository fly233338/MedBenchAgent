# ISIC2018 server-side preparation

The source tree is read-only. Run these commands from the repository root on the server after copying this artifact directory. The source dataset is available at `/public/home/rxa-jj-wzk/data/fyl/ISIC2018`.

## 1. Prepare annotation.jsonl

```bash
uv run python datasets/Normalized/ISIC2018/scripts/prepare_annotations.py \
  --source-root /public/home/rxa-jj-wzk/data/fyl/ISIC2018 \
  --output-dir datasets/Normalized/ISIC2018 \
  --plan-path datasets/Normalized/ISIC2018/conversion_plan.json \
  --annotation-path datasets/Normalized/ISIC2018/annotation.jsonl \
  --report-path datasets/Normalized/ISIC2018/annotation_report.json \
  --verify-determinism
```

The converter expects the extracted test directories already present below `/public/home/rxa-jj-wzk/data/fyl/ISIC2018`. It excludes the training-input archive because no matching Task 1/2 training ground truth is supplied.

## 2. Copy original JPEGs into the annotation artifact

The source images are RGB JPEGs. The image script copies their original bytes without grayscale conversion, resizing, re-encoding, crop, rotation, or flip. Per the image-preparation contract, images are placed under the artifact's `images/` directory and the manifest is beside `annotation.jsonl`:

```bash
uv run python datasets/Normalized/ISIC2018/scripts/prepare_images.py \
  --annotation datasets/Normalized/ISIC2018/annotation.jsonl \
  --source-root /public/home/rxa-jj-wzk/data/fyl/ISIC2018 \
  --image-output-dir datasets/Normalized/ISIC2018/images \
  --manifest-path datasets/Normalized/ISIC2018/image_manifest.jsonl \
  --encoding original_jpeg
```

This step processes only the 2,512 JPEGs referenced by `annotation.jsonl`. The manifest keeps the existing `png_path` key for evaluator backward compatibility, but its values point to `.jpg` files such as `images/ISIC_0000001.jpg`. Existing identical JPEGs and manifests are accepted; different existing outputs are rejected.

The evaluation loader now accepts both PNG and JPEG assets through this manifest. The annotation construction workflow continues to use the source-relative `image.image_path` values.

## 3. Optional fast checks before the full run

```bash
uv run python datasets/Normalized/ISIC2018/scripts/prepare_annotations.py --help
uv run python datasets/Normalized/ISIC2018/scripts/prepare_images.py --help
```
