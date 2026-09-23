"""Validate source-bound inspection and semantic references before preparation writes."""

from medbenchagent.runtime.inspection import inspection


def semantic_evidence(context, evidence, source_id):
    if not evidence:
        raise ValueError("Source semantics require document quotations or an answered user decision")
    for item in evidence:
        if item.source == "document":
            receipt = inspection(context, item.reference, source_id=source_id)
            if receipt["operation"] != "read_text":
                raise ValueError("Semantic evidence must reference inspected source documentation")
            text = receipt["value"]["text"]
        else:
            from medbenchagent.runtime.inspection import evidence as read_evidence
            receipt = read_evidence(context, item.reference)
            issue = context.get("issues", {}).get(receipt["value"].get("issue_id"), {})
            if (receipt["kind"] != "user" or issue.get("status") != "resolved"
                    or issue.get("kind") != "semantic" or issue.get("role") != "DataProcessingAgent"
                    or issue.get("dataset_id") != context.get("dataset_scope")
                    or source_id not in issue.get("source_ids", [])
                    or issue.get("answer_reference") != item.reference):
                raise ValueError("Semantic evidence requires a resolved decision covering this source")
            text = receipt["value"]["text"]
        if not item.quote.strip() or item.quote not in text:
            raise ValueError("Semantic evidence requires an exact quotation")


def mapping_evidence(context, mapping):
    source = context["source_scope"]
    if not mapping.population_reference:
        raise ValueError("Mapping requires an inspected population_reference; inspect image identities first")
    receipt = inspection(context, mapping.population_reference, source_id=source)
    if receipt["operation"] != "image_population":
        raise ValueError("Mapping population_reference must identify an image population inspection")
    from medbenchagent.data.sources import digest_file
    if digest_file(receipt["value"]["population_path"]) != receipt["value"]["population_digest"]:
        raise ValueError("Inspected population identities changed")
    if receipt["value"]["unique_images"] != mapping.expected_images:
        raise ValueError("expected_images differs from the inspected authoritative population")
    if not mapping.inspection_references:
        raise ValueError("Mapping requires actual inspection references")
    for reference in mapping.inspection_references:
        inspection(context, reference, source_id=source)
    semantic_evidence(context, mapping.semantic_evidence, source)


def alignment_evidence(context, field):
    if not field.inspection_references:
        raise ValueError("Field alignment requires normalized field inspection references")
    receipts = [inspection(context, ref, source_id=field.source_id) for ref in field.inspection_references]
    if not any(r["operation"] == "inspect_normalized" and (
        r["arguments"].get("field") == field.field or field.field in r["value"].get("fields", {})
    ) for r in receipts):
        raise ValueError("Inspect the actual normalized source field before alignment")
    if field.action in {"merge", "exclude"} or field.semantic_evidence:
        semantic_evidence(context, field.semantic_evidence, field.source_id)
