"""Real files keyed directly by Case image locators."""
from pathlib import Path
from PIL import Image
from medbenchagent.data.sources import digest_file
from medbenchagent.planning.resolver import ResolverContext


def image_context(root, cases, context=None):
    root = Path(root)
    context = context or ResolverContext()
    for case in cases:
        for evidence in case["evidence"]:
            if evidence["type"] != "asset" or evidence.get("mask"):
                continue
            key = evidence["value"]
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                Image.new("RGB", (4, 4), "red").save(path, format="PNG")
            context.assets[key] = {"path": key, "format": "png", "kind": "image", "source_id": "s",
                "source_path": key, "source_digest": digest_file(path), "file_digest": digest_file(path),
                "width": 4, "height": 4, "metadata": {"input_visibility": "model_visible"}}
    context.dataset_root = str(root)
    return context
