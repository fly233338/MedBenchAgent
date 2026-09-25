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
            if issue.get("reference") == "population_selection":
                raise ValueError("Population selection approval is not evidence for field semantics")
            if (receipt["kind"] != "user" or issue.get("status") != "resolved"
                    or issue.get("kind") != "semantic" or issue.get("role") != "DataProcessingAgent"
                    or issue.get("dataset_id") != context.get("dataset_scope")
                    or source_id not in issue.get("source_ids", [])
                    or issue.get("answer_reference") != item.reference):
                raise ValueError("Semantic evidence requires a resolved decision covering this source")
            text = receipt["value"]["text"]
        if not item.quote.strip() or item.quote not in text:
            raise ValueError("Semantic evidence requires an exact quotation")


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
