# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.28,<1",
#   "pypdf>=5,<7",
# ]
# ///
"""Download screened medical references and classify files for local PDF RAG."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pypdf import PdfReader

DEFAULT_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_MIN_PDF_TEXT_CHARS = 100
CANDIDATE_FIELDS = (
    "title",
    "landing_url",
    "download_url",
    "publisher",
    "source_type",
    "publication_date",
    "version",
    "language",
    "disease",
    "modality",
    "selection_reason",
    "search_query",
    "searched_domain",
    "search_date",
)


class CollectionError(RuntimeError):
    """Raised when candidates or output locations are invalid."""


def _canonical_url(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionError(f"{field} must be a non-empty http(s) URL")
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise CollectionError(f"{field} must be an http(s) URL: {value}")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CollectionError(f"candidate file does not exist: {path}")
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise CollectionError(f"candidate line {line_number}: blank lines are not allowed")
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CollectionError(f"candidate line {line_number}: invalid JSON") from exc
            if not isinstance(candidate, dict):
                raise CollectionError(f"candidate line {line_number}: must be an object")
            title = candidate.get("title")
            if not isinstance(title, str) or not title.strip():
                raise CollectionError(f"candidate line {line_number}: title is required")
            normalized = {field: candidate.get(field) for field in CANDIDATE_FIELDS}
            normalized["title"] = title.strip()
            normalized["landing_url"] = _canonical_url(
                candidate.get("landing_url"), field=f"candidate line {line_number} landing_url"
            )
            download_url = candidate.get("download_url")
            normalized["download_url"] = (
                _canonical_url(
                    download_url, field=f"candidate line {line_number} download_url"
                )
                if download_url
                else None
            )
            candidates.append(normalized)
    if not candidates:
        raise CollectionError("candidate file contains no sources")
    return candidates


def _slug(value: str, limit: int = 90) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    rendered = "".join(char if char.isalnum() or char in "-_" else "_" for char in normalized)
    rendered = re.sub(r"_+", "_", rendered).strip("_.-")
    return (rendered or "reference")[:limit]


def _download(client: httpx.Client, url: str, max_bytes: int) -> tuple[bytes, httpx.Response]:
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise CollectionError(f"download exceeds {max_bytes} bytes")
                chunks.append(chunk)
            return b"".join(chunks), response
    except httpx.HTTPStatusError as exc:
        raise CollectionError(f"HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise CollectionError(f"{type(exc).__name__}: {exc}") from exc


def _classify(data: bytes, content_type: str, final_url: str) -> tuple[str, str]:
    media_type = content_type.split(";", 1)[0].strip().lower()
    prefix = data[:512].lstrip().lower()
    if data.startswith(b"%PDF-"):
        return "pdf", "application/pdf"
    if media_type in {"text/html", "application/xhtml+xml"} or prefix.startswith(
        (b"<!doctype html", b"<html")
    ):
        return "html", media_type or "text/html"
    if data.startswith(b"PK\x03\x04") and (
        "wordprocessingml" in media_type or urlsplit(final_url).path.lower().endswith(".docx")
    ):
        return "docx", media_type or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    suffix = Path(urlsplit(final_url).path).suffix.lower().lstrip(".")
    extension = suffix if suffix and suffix.isalnum() and len(suffix) <= 8 else "bin"
    return extension, media_type or "application/octet-stream"


def _pdf_text_chars(data: bytes) -> tuple[int, str | None]:
    try:
        reader = PdfReader(io.BytesIO(data))
        count = sum(len((page.extract_text() or "").strip()) for page in reader.pages)
        return count, None
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def _base_result(candidate: dict[str, Any], accessed_at: str) -> dict[str, Any]:
    return {
        **candidate,
        "requested_url": candidate.get("download_url") or candidate["landing_url"],
        "final_url": None,
        "accessed_at": accessed_at,
        "http_status": None,
        "media_type": None,
        "download_status": "link_only",
        "local_path": None,
        "sha256": None,
        "byte_count": None,
        "extracted_text_chars": None,
        "rag_compatible": False,
        "duplicate_of": None,
        "error": None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _report(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["download_status"]) for row in rows)
    ready = len(
        {
            str(row["local_path"])
            for row in rows
            if row["rag_compatible"] and row["local_path"]
        }
    )
    lines = [
        "# Medical Reference Collection Report",
        "",
        f"- Candidates: {len(rows)}",
        f"- RAG-ready PDFs: {ready}",
        f"- Downloaded unique files: {counts.get('downloaded', 0)}",
        f"- Duplicate URLs: {counts.get('duplicate_url', 0)}",
        f"- Duplicate content: {counts.get('duplicate_content', 0)}",
        f"- Link only: {counts.get('link_only', 0)}",
        "- Pass `rag_ready/` to the current LocalPdfRag.",
        "",
    ]
    unresolved = [row for row in rows if row["download_status"] == "link_only"]
    if unresolved:
        lines.extend(["## Link-only sources", ""])
        for row in unresolved:
            title = str(row["title"]).replace("|", "\\|")
            reason = str(row.get("error") or "No direct file was saved.")
            lines.append(f"- [{title}]({row['landing_url']}): {reason}")
        lines.append("")
    return "\n".join(lines)


def collect(
    candidates_path: Path,
    output_path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    min_pdf_text_chars: int = DEFAULT_MIN_PDF_TEXT_CHARS,
) -> dict[str, Any]:
    if max_bytes <= 0:
        raise CollectionError("--max-bytes must be positive")
    if min_pdf_text_chars < 0:
        raise CollectionError("--min-pdf-text-chars must be non-negative")
    candidates = _read_candidates(candidates_path.resolve())
    output = output_path.expanduser().resolve()
    if output == Path(output.anchor):
        raise CollectionError("output must not be a filesystem root")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CollectionError(f"output must be absent or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    accessed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    results: list[dict[str, Any]] = []
    seen_urls: dict[str, dict[str, Any]] = {}
    seen_hashes: dict[str, dict[str, Any]] = {}
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "MedBenchAgent-ReferenceCollector/1.0"},
        ) as client:
            for candidate in candidates:
                result = _base_result(candidate, accessed_at)
                requested_url = str(result["requested_url"])
                if requested_url in seen_urls:
                    prior = seen_urls[requested_url]
                    result.update(
                        {
                            "download_status": "duplicate_url",
                            "final_url": prior["final_url"],
                            "http_status": prior["http_status"],
                            "media_type": prior["media_type"],
                            "local_path": prior["local_path"],
                            "sha256": prior["sha256"],
                            "byte_count": prior["byte_count"],
                            "extracted_text_chars": prior["extracted_text_chars"],
                            "rag_compatible": prior["rag_compatible"],
                            "duplicate_of": prior["landing_url"],
                        }
                    )
                    results.append(result)
                    continue
                try:
                    data, response = _download(client, requested_url, max_bytes)
                    result["final_url"] = str(response.url)
                    result["http_status"] = response.status_code
                    extension, media_type = _classify(
                        data, response.headers.get("content-type", ""), str(response.url)
                    )
                    result["media_type"] = media_type
                    digest = hashlib.sha256(data).hexdigest()
                    result["sha256"] = digest
                    result["byte_count"] = len(data)
                    if digest in seen_hashes:
                        prior = seen_hashes[digest]
                        result.update(
                            {
                                "download_status": "duplicate_content",
                                "local_path": prior["local_path"],
                                "extracted_text_chars": prior["extracted_text_chars"],
                                "rag_compatible": prior["rag_compatible"],
                                "duplicate_of": prior["landing_url"],
                            }
                        )
                    else:
                        extracted_chars: int | None = None
                        parse_error: str | None = None
                        rag_compatible = False
                        target_directory = "downloads"
                        if extension == "pdf":
                            extracted_chars, parse_error = _pdf_text_chars(data)
                            rag_compatible = extracted_chars >= min_pdf_text_chars and not parse_error
                            target_directory = "rag_ready" if rag_compatible else "downloads"
                        filename = f"{_slug(str(candidate['title']))}__{digest[:10]}.{extension}"
                        relative_path = f"{target_directory}/{filename}"
                        destination = staging / target_directory / filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(data)
                        result.update(
                            {
                                "download_status": "downloaded",
                                "local_path": relative_path,
                                "extracted_text_chars": extracted_chars,
                                "rag_compatible": rag_compatible,
                                "error": parse_error,
                            }
                        )
                        seen_hashes[digest] = result
                except CollectionError as exc:
                    result["error"] = str(exc)
                seen_urls[requested_url] = result
                results.append(result)

        _write_jsonl(staging / "candidate_sources.jsonl", candidates)
        _write_jsonl(staging / "reference_manifest.jsonl", results)
        (staging / "rag_ready").mkdir(exist_ok=True)
        (staging / "downloads").mkdir(exist_ok=True)
        (staging / "collection_report.md").write_text(_report(results), encoding="utf-8")
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return {
            "output": str(output),
            "candidate_count": len(results),
            "rag_ready_count": len(
                {
                    str(row["local_path"])
                    for row in results
                    if row["rag_compatible"] and row["local_path"]
                }
            ),
            "link_only_count": sum(row["download_status"] == "link_only" for row in results),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--min-pdf-text-chars", type=int, default=DEFAULT_MIN_PDF_TEXT_CHARS
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = collect(
            args.candidates,
            args.output,
            max_bytes=args.max_bytes,
            min_pdf_text_chars=args.min_pdf_text_chars,
        )
    except CollectionError as exc:
        print(f"collection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
