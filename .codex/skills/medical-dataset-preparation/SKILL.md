---
name: medical-dataset-preparation
description: Prepare a medical imaging dataset for benchmark generation by normalizing image-centric annotations, converting supported two-dimensional images to deterministic 8-bit PNGs, and building an image_id-to-PNG manifest. Use when a dataset needs annotation.jsonl plus image_manifest.jsonl without changing the annotation contract.
---

# Medical Dataset Preparation

Keep annotation facts and pixels separate. Do not change `annotation.jsonl` to point to PNGs or add conversion metadata to it.

1. Use `$normalize-medical-image-dataset` first. Follow its inventory, confirmed conversion plan, deterministic converter, and validation workflow.
2. Use `$convert-medical-images-to-png` on source images. Keep the generated `images/`, conversion manifest, mapping CSV, and report. Do not resize, crop, flip, or otherwise spatially transform images.
3. Build the final mapping with:

```powershell
python .codex\skills\medical-dataset-preparation\scripts\build_image_manifest.py `
  --annotation <prepared_dataset\annotation.jsonl> `
  --conversion-manifest <prepared_dataset\image_conversion_manifest.jsonl> `
  --output <prepared_dataset\image_manifest.jsonl>
```

The script performs exact normalized-relative-path joins only, rejects missing or ambiguous matches, and writes in annotation order. Consumers resolve `png_path` relative to the manifest directory by `image_id`; never use `BenchmarkItem.image_path` to locate a PNG.

Keep the prepared result limited to annotation, image manifest, converted images, and the normalization/conversion artifacts. The dataset catalog continues to register only `annotation.jsonl`.
