"""Frozen source documentation; planning never reopens raw source directories."""

import json
import shutil
from pathlib import Path
from medbenchagent.data.sources import digest_file
from medbenchagent.data.validation import source_path
from medbenchagent.runtime.run_artifacts import atomic_write_json

TEXT_FORMATS = {"", ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".xml"}


def document_index(bundle):
    path = Path(bundle) / "documentation/index.json"
    if not path.is_file():
        raise ValueError("Prepared documentation index is missing; prepare a new version")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "dataset-documentation.v1":
        raise ValueError("Unsupported frozen documentation index")
    return value


def freeze_document(bundle, source_id, root, path):
    bundle = Path(bundle)
    if (bundle / "ready.json").exists():
        raise ValueError("Published documentation is immutable")
    original = source_path(root, path)
    suffix = original.suffix.lower()
    if suffix not in TEXT_FORMATS | {".pdf"}:
        raise ValueError("Only UTF-8 text and text-extractable PDF documentation are supported")
    index_path = bundle / "documentation/index.json"
    index = document_index(bundle) if index_path.exists() else {"schema_version": "dataset-documentation.v1", "documents": []}
    for item in index["documents"]:
        if item["source_id"] == source_id and item["path"] == path:
            if item["file_digest"] != digest_file(original):
                raise ValueError("Source documentation changed; start a new preparation")
            return item
    if suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
        try:
            reader = PdfReader(original)
            if reader.is_encrypted:
                raise ValueError("Encrypted source PDFs are unsupported; provide an unencrypted text document")
            pages = [(page.extract_text() or "") for page in reader.pages]
        except PdfReadError as exc:
            raise ValueError(f"Source PDF cannot be read: {path}: {exc}") from exc
        if not any(text.strip() for text in pages):
            raise ValueError("Source PDF has no extractable text; OCR is not supported")
    else:
        pages = [original.read_text(encoding="utf-8")]
    number = len(index["documents"]) + 1
    relative = f"documentation/{source_id}/{number}{suffix}"
    target = bundle / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    text_path = f"documentation/{source_id}/{number}.pages.json"
    atomic_write_json(bundle / text_path, {"pages": pages})
    item = {"source_id": source_id, "path": path, "file_path": relative, "text_path": text_path,
            "format": suffix.lstrip(".") or "text", "pages": len(pages),
            "file_digest": digest_file(target), "text_digest": digest_file(bundle / text_path)}
    index["documents"].append(item)
    atomic_write_json(index_path, index)
    return item


def read_document(bundle, source_id, path, *, page=1, offset=0, limit=6000):
    item = next((d for d in document_index(bundle)["documents"] if d["source_id"] == source_id and d["path"] == path), None)
    if item is None:
        raise ValueError("Document was not frozen for this prepared version")
    for key, expected in (("file_path", "file_digest"), ("text_path", "text_digest")):
        target = source_path(bundle, item[key])
        if digest_file(target) != item[expected]:
            raise ValueError("Frozen documentation changed; prepare a new version")
    pages = json.loads((Path(bundle) / item["text_path"]).read_text(encoding="utf-8"))["pages"]
    if not 1 <= page <= len(pages) or offset < 0:
        raise ValueError("Invalid document page or offset")
    text = pages[page - 1]
    return {"source_id": source_id, "path": path, "page": page, "pages": len(pages),
            "text": text[offset:offset + limit], "next_offset": offset + limit if offset + limit < len(text) else None}
