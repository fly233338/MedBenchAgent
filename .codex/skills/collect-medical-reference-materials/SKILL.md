---
name: collect-medical-reference-materials
description: Search for, screen, download, and inventory disease-specific medical reference materials for local benchmark RAG. Use when Codex must collect current guidelines, consensus statements, reporting standards, or open-access reviews for a named disease; preserve source links when files cannot be downloaded; verify provenance and versions; or prepare a rag_ready directory of text-extractable PDFs.
---

# Collect Medical Reference Materials

Build a reproducible local reference collection for one specified disease. Search with current web access, then use the bundled downloader for deterministic fetching and classification.

## Boundaries

- Require a disease. Accept optional modality, benchmark tasks, languages, date range, and output directory.
- Prefer current government, WHO, and professional-society guidance, then relevant legal open-access reviews.
- Do not bypass authentication, paywalls, robots restrictions, or access controls.
- Do not claim that a link supports a topic until its title, publisher, date/version, and relevance are checked.
- Keep unavailable sources as links. Do not replace them with invented citations.
- Do not OCR scans or convert webpages to PDF in v1.

Read [references/source-selection.md](references/source-selection.md) before searching or writing the candidate file.

## Workflow

1. Expand the disease into verified English and Chinese names and relevant modality/reporting terms.
2. Search official publishers first, then PubMed/PMC and other open-access scholarly sources. Record every query and access date.
3. Screen candidates against the source hierarchy and remove clearly obsolete or irrelevant versions.
4. Write `candidate_sources.jsonl` using the reference schema. Keep one source per line.
5. Run:

```powershell
uv run --script .codex\skills\collect-medical-reference-materials\scripts\fetch_sources.py `
  --candidates <candidate_sources.jsonl> `
  --output <materials-output-directory>
```

6. Review `reference_manifest.jsonl` and `collection_report.md`. Pass only the emitted `rag_ready/` directory to the current `LocalPdfRag`.

## Output Contract

- `candidate_sources.jsonl`: canonical copy of screened inputs.
- `reference_manifest.jsonl`: one result per candidate with provenance, fetch status, hashes, local paths, and RAG compatibility.
- `rag_ready/*.pdf`: downloaded PDFs with enough extractable text for current RAG.
- `downloads/*`: downloaded HTML, office files, binary files, and PDFs without sufficient text.
- `collection_report.md`: counts, ready directory, link-only sources, and errors.

## Completion Criteria

Finish only when every candidate is represented in the manifest, downloaded bytes have hashes, duplicate URLs and content are reconciled, unavailable files retain landing links and reasons, and every file in `rag_ready/` passes PDF parsing and text extraction.
