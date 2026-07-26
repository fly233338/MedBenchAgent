---
name: prepare-medical-images
description: Convert only the source images referenced by a canonical annotation.jsonl into deterministic 8-bit PNGs and an exact verified image_manifest.jsonl. Use after annotation preparation when MedBenchAgent image assets must be materialized.
---

# Prepare Medical Images

Consume a confirmed `annotation.jsonl`. Do not reinterpret annotations, discover additional images, or use fuzzy path matching.

## Inputs

Obtain the canonical annotation file, source-image root, and a processed-image directory inside the annotation artifact directory.

## Workflow

Run the bundled orchestrator:

```text
uv run python .codex/skills/prepare-medical-images/scripts/prepare_images.py --annotation <artifact/annotation.jsonl> --source-image-root <source> --processed-image-dir <artifact/images>
```

It resolves each `annotation.image_path` beneath the source root, converts only those files in annotation order, builds the exact `image_id` mapping, and verifies every output PNG. Conversion is locked and deterministic; do not request resize, crop, rotation, flip, or another spatial transform.

## Output

```text
<artifact_dir>/
├── annotation.jsonl
├── image_manifest.jsonl
├── image_preparation_report.json
└── <processed_images>/
```

Each manifest row contains `image_id`, source-relative path, artifact-relative PNG path, dimensions, `L` or `RGB` mode, SHA-256, conversion status, and spatial-transform metadata.

Finish when every annotation maps to one existing verified PNG, no unreferenced source image was converted, all paths remain inside their declared roots, and unchanged reruns are byte-identical.
