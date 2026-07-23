from __future__ import annotations

import importlib.util
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZE_SCRIPT = (
    PROJECT_ROOT
    / ".codex"
    / "skills"
    / "normalize-medical-image-dataset"
    / "scripts"
    / "validate_annotation.py"
)
FETCH_SCRIPT = (
    PROJECT_ROOT
    / ".codex"
    / "skills"
    / "collect-medical-reference-materials"
    / "scripts"
    / "fetch_sources.py"
)
NORMALIZE_SCHEMA = PROJECT_ROOT / ".codex" / "skills" / "normalize-medical-image-dataset" / "schema.json"
PROJECT_SCHEMA = PROJECT_ROOT / "src" / "medbenchagent" / "schemas" / "annotation.v1.schema.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalize = _load_module("normalize_medical_dataset_skill", NORMALIZE_SCRIPT)
fetch = _load_module("collect_medical_references_skill", FETCH_SCRIPT)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _annotation(image_path: str, image_id: str = "image-1") -> dict:
    return {
        "image_id": image_id,
        "image_path": image_path,
        "patient_id": None,
        "case_id": None,
        "study_id": None,
        "annotation": {"label": "example"},
    }


def _manifest_row(source: str, png: str) -> dict:
    return {
        "source_relative_path": source,
        "source_filename": Path(source).name,
        "status": "converted",
        "png_relative_path": png,
    }


def test_normalization_schema_uses_direct_annotation_modality_and_anatomy() -> None:
    skill_schema = json.loads(NORMALIZE_SCHEMA.read_text(encoding="utf-8"))
    project_schema = json.loads(PROJECT_SCHEMA.read_text(encoding="utf-8"))

    annotation_properties = skill_schema["properties"]["annotation"]["properties"]
    assert skill_schema == project_schema
    assert {"modality", "anatomy"} <= set(annotation_properties)
    assert "metadata" not in annotation_properties


def test_normalization_validator_checks_schema_manifest_and_files(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    manifest_path = tmp_path / "image_conversion_manifest.jsonl"
    image_root = tmp_path / "converted"
    png_path = image_root / "images" / "scan.png"
    png_path.parent.mkdir(parents=True)
    png_path.write_bytes(b"png-placeholder")
    _write_jsonl(annotation_path, [_annotation("images/scan.png")])
    _write_jsonl(manifest_path, [_manifest_row("source/scan.dcm", "images/scan.png")])

    report = normalize.validate(
        annotation_path,
        image_manifest=manifest_path,
        image_root=image_root,
        repo_root=PROJECT_ROOT,
    )

    assert report["valid"] is True
    assert report["record_count"] == 1
    assert report["image_files_checked"] == 1


def test_normalization_validator_requires_final_png_manifest_path(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    manifest_path = tmp_path / "image_conversion_manifest.jsonl"
    _write_jsonl(annotation_path, [_annotation("source/scan.dcm")])
    _write_jsonl(manifest_path, [_manifest_row("source/scan.dcm", "images/scan.png")])

    with pytest.raises(normalize.ValidationError, match="still references the source image"):
        normalize.validate(
            annotation_path, image_manifest=manifest_path, repo_root=PROJECT_ROOT
        )


def test_normalization_validator_rejects_missing_and_ambiguous_mapping(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    manifest_path = tmp_path / "image_conversion_manifest.jsonl"
    _write_jsonl(annotation_path, [_annotation("images/missing.png")])
    _write_jsonl(manifest_path, [_manifest_row("a/scan.dcm", "images/a.png")])
    with pytest.raises(normalize.ValidationError, match="missing from the converted manifest"):
        normalize.validate(
            annotation_path, image_manifest=manifest_path, repo_root=PROJECT_ROOT
        )

    _write_jsonl(annotation_path, [_annotation("images/shared.png")])
    _write_jsonl(
        manifest_path,
        [
            _manifest_row("a/scan.dcm", "images/shared.png"),
            _manifest_row("b/scan.dcm", "images/shared.png"),
        ],
    )
    with pytest.raises(normalize.ValidationError, match="ambiguous png_relative_path"):
        normalize.validate(
            annotation_path, image_manifest=manifest_path, repo_root=PROJECT_ROOT
        )


def _text_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    text = "authoritative medical imaging guideline " * 8
    stream.set_data(f"BT /F1 10 Tf 20 250 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    destination = io.BytesIO()
    writer.write(destination)
    return destination.getvalue()


class _ReferenceHandler(BaseHTTPRequestHandler):
    pdf_bytes = _text_pdf()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.end_headers()
            return
        if self.path in {"/text.pdf", "/same.pdf"}:
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(self.pdf_bytes)
            return
        if self.path == "/page":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<!doctype html><html><body>guideline page</body></html>")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def reference_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_reference_collector_downloads_deduplicates_and_keeps_links(
    tmp_path: Path, reference_server: str
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "materials"
    rows = [
        {"title": "Primary PDF", "landing_url": f"{reference_server}/text.pdf"},
        {"title": "Same content", "landing_url": f"{reference_server}/same.pdf"},
        {"title": "Duplicate URL", "landing_url": f"{reference_server}/text.pdf"},
        {"title": "HTML page", "landing_url": f"{reference_server}/redirect"},
        {"title": "Unavailable", "landing_url": f"{reference_server}/missing"},
    ]
    _write_jsonl(candidates, rows)

    report = fetch.collect(candidates, output, min_pdf_text_chars=20)

    manifest = [
        json.loads(line)
        for line in (output / "reference_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report == {
        "output": str(output.resolve()),
        "candidate_count": 5,
        "rag_ready_count": 1,
        "link_only_count": 1,
    }
    assert [row["download_status"] for row in manifest] == [
        "downloaded",
        "duplicate_content",
        "duplicate_url",
        "downloaded",
        "link_only",
    ]
    assert len(list((output / "rag_ready").glob("*.pdf"))) == 1
    assert len(list((output / "downloads").glob("*.html"))) == 1
    assert manifest[0]["sha256"] == manifest[1]["sha256"]
    assert manifest[4]["landing_url"].endswith("/missing")
    assert "HTTP 404" in manifest[4]["error"]


def test_reference_collector_rejects_invalid_candidate_without_output(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "materials"
    _write_jsonl(candidates, [{"title": "Bad", "landing_url": "file:///private.pdf"}])

    with pytest.raises(fetch.CollectionError, match=r"http\(s\) URL"):
        fetch.collect(candidates, output)
    assert not output.exists()
