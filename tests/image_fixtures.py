"""Small real presentations for tests exercising model-input consumers."""
from pathlib import Path
from PIL import Image

from medbenchagent.data.manifests import PresentationRecord
from medbenchagent.data.sources import digest_file
from medbenchagent.planning.resolver import ResolverContext


def image_context(root, cases, context=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    context = context or ResolverContext()
    records = []
    for case in cases:
        for index, evidence in enumerate(case["evidence"]):
            if evidence["type"] != "asset" or evidence.get("mask"):
                continue
            key = evidence["value"]
            path = root / f"{key}.png"
            if not path.exists():
                Image.new("RGB", (4, 4), "red").save(path)
            context.assets[key] = {"asset_id": key, "format": "png", "kind": "image",
                "source_path": path.name, "file_digest": digest_file(path), "metadata": {"input_visibility": "model_visible"}}
            records.append(PresentationRecord(presentation_id=f"{case['case_id']}:{index}", case_id=case["case_id"],
                asset_id=key, path=path.name, input_digest=digest_file(path), output_digest=digest_file(path),
                mode="native", view_id=evidence.get("view_id"), order=index))
    context.manifests["presentation"] = records
    context.presentation_root = str(root)
    return context
