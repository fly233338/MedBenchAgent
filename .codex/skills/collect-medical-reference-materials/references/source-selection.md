# Source Selection and Candidate Schema

## Selection order

1. Current government, WHO, regulator, or national-program publications.
2. Current professional-society guidelines, consensus statements, lexicons, staging systems, and reporting standards.
3. Peer-reviewed open-access systematic reviews or high-quality narrative reviews that fill a documented gap.

Prefer the publisher's landing page and direct download URL. Confirm version and publication date. Retain an older edition only when needed for a dataset's historical label definition, and state that reason.

Exclude promotional pages, unsourced summaries, duplicate mirrors, inaccessible citations with unverifiable metadata, and papers whose relationship to the disease or imaging task is only speculative.

## Search record

Record query text, search date, language, and searched domain in the agent's workflow log. Include disease synonyms, modality terms, and terms such as `guideline`, `consensus`, `classification`, `reporting`, or `open access review` as appropriate.

## `candidate_sources.jsonl`

Write one JSON object per source. Require `title` and `landing_url`; use `null` for unavailable optional values.

```json
{
  "title": "Example imaging guideline",
  "landing_url": "https://publisher.example/guideline",
  "download_url": "https://publisher.example/guideline.pdf",
  "publisher": "Example Society",
  "source_type": "professional_guideline",
  "publication_date": "2026-01-15",
  "version": "2",
  "language": "en",
  "disease": "example disease",
  "modality": "mammography",
  "selection_reason": "Defines the assessment categories used by the benchmark.",
  "search_query": "example disease mammography guideline",
  "searched_domain": "publisher.example",
  "search_date": "2026-07-20"
}
```

Use only `http` or `https` URLs. `download_url` may be `null`; the downloader will try the landing page and store accessible HTML or retain a link-only result on failure.
