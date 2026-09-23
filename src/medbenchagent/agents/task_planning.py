"""Incremental two-level medical concept and leaf planning."""

from __future__ import annotations
import json
import shutil
from pathlib import Path
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.contexts import TaskPlanningContext
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.rag.pdf import LocalPdfRag, discover_medical_materials
from medbenchagent.settings import get_settings
from medbenchagent.runtime.tool_contracts import (
    PageArguments,
    FieldArguments,
    MaterialArguments,
    MaterialsArguments,
    SearchArguments,
    ReadTextArguments,
)
from medbenchagent.planning.tree import (
    LeafArguments,
    ConceptArguments,
    NodeArguments,
    ExtensionArguments,
    FieldValuesArguments,
    concepts_complete,
    validate_plan,
    validate_draft,
    ExtensionDecision,
    GlobalReview,
)
from medbenchagent.planning.protocol import validate_dataset_protocol
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.inspection import inspected, inspection, inspection_index, save_evidence, evidence as read_evidence_record
from medbenchagent.data.query import DatasetQuery, iter_records
from medbenchagent.planning.scope import ScopeResolution, conditions_for, validate_scope_plan
from medbenchagent.planning.tree import SemanticBasis
from medbenchagent.runtime.tool_contracts import NoArguments
from pydantic import Field

TEXT_DOCUMENT_FORMATS = {"", ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".xml"}
DOCUMENT_FORMATS = TEXT_DOCUMENT_FORMATS | {".pdf", ".doc", ".docx", ".xls", ".xlsx"}


def list_dataset_documents(context, *, offset=0, limit=20, query=""):
    """Page only documentation frozen inside this Dataset version."""
    from medbenchagent.data.documentation import document_index
    require(context, "Dataset")
    bundle = Path(context["annotation_path"]).parent
    docs = [{**{k: d[k] for k in ("source_id", "format", "pages")},
             "path": d["source_id"] + "/" + d["path"], "readable": True}
            for d in document_index(bundle)["documents"] if query.casefold() in d["path"].casefold()]
    return result("success", json.dumps({"documents": docs[offset:offset + limit], "total": len(docs),
                  "next_offset": offset + limit if offset + limit < len(docs) else None}, ensure_ascii=False))


class DocumentArguments(ReadTextArguments):
    page: int = Field(default=1, ge=1)


class ScopeArguments(NoArguments):
    resolutions: list[ScopeResolution]
    semantic_basis: list[SemanticBasis] = Field(default_factory=list)


def _scope_conditions(context, tree):
    conditions = conditions_for(tree, context.get("sample_scope", []))
    if conditions is None:
        raise ValueError("Sample scope is unsupported; remove leaves and finish an unsupported Plan")
    return conditions


def resolve_sample_scope(context, *, resolutions, semantic_basis=None):
    """Resolve every user population restriction with scalar conditions or an explicit unsupported reason."""
    tree = _tree(context)
    tree["scope_resolutions"] = [ScopeResolution.model_validate(r).model_dump() for r in resolutions]
    conditions = conditions_for(tree, context.get("sample_scope", []))
    fields = list(dict.fromkeys(c["field"] for c in conditions or []))
    _subset(_profile(context), fields)
    basis = _validate_semantic_basis(context, {"data_binding": {"supporting_fields": fields}}, semantic_basis or [])
    for condition in conditions or []:
        # Non-text codes always need dataset-specific evidence. Text meanings are
        # checked with the complete user scope and source quotations at review.
        if any(not isinstance(v, str) for v in condition["values"]) and not any(
            b["field"] == condition["field"] and (b["source"] in {"document", "user"} or b["dataset_specific"])
            for b in basis
        ):
            raise ValueError("Coded scope values require dataset-specific semantic evidence")
    _validate_scope_decisions(context, tree, basis)
    tree["scope_semantic_basis"] = basis
    for leaf in tree["tasks"].values():
        leaf["protocol"]["sample_conditions"] = conditions
        leaf["eligible_count"] = None
        tree["concepts"][leaf["concept_id"]]["status"] = "pending"
    return _persist(context, tree, changed=True)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _base(context):
    blueprint = _read(get_settings().paths.blueprint_template)
    blueprint.update(
        benchmark_info=context["benchmark_info"],
        dataset={"dataset_id": context.get("selected_dataset_id")},
    )
    return blueprint


def _profile(context):
    require(context, "Dataset")
    return _read(context["dataset_access_manifest"]["profile_path"])


def list_fields(context, *, offset=0, limit=20, query=""):
    """Page field paths and availability; text search helps discovery but never drops tasks."""
    fields = _profile(context)["fields"]
    rows = [
        {"path": key, "types": value["types"], "non_empty_images": value["non_empty_images"]}
        for key, value in fields.items()
        if query.casefold() in key.casefold()
    ]
    return inspected(context, "list_fields", {"offset": offset, "limit": limit, "query": query},
                     {"total": len(rows), "fields": rows[offset : offset + limit]}, ("Dataset",))


def read_fields(context, *, fields):
    """Read selected field distributions and their object/array structure."""
    from medbenchagent.data.dataset_catalog import resolve_path
    card = _read(resolve_path(context["dataset_entry"]["dataset_card_path"]))
    details = _subset(_profile(context), fields)
    details["field_roles"] = {key: card["semantics"]["field_roles"].get(key) for key in fields}
    return inspected(context, "read_fields", {"fields": fields}, details, ("Dataset", "DatasetCard"))


def read_field_values(context, *, field, offset=0, limit=20):
    """Page complete exact values from the current Dataset index."""
    require(context, "Dataset")
    value = DatasetQuery(context["dataset_access_manifest"]).values(field, offset, limit)
    return inspected(context, "read_field_values", {"field": field, "offset": offset, "limit": limit}, value, ("Dataset",))


def inspect_field_by_source(context, *, field):
    """Inspect exact per-source coverage and vocabulary counts."""
    require(context, "Dataset")
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).by_source(field)))


def sample_records(context, *, offset=0, limit=5, source_id=None):
    """Read a bounded page of canonical records without raw provenance identifiers."""
    require(context, "Dataset")
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).sample(offset, limit, source_id)))


def inspect_field_pair(context, *, left, right):
    """Distinguish same-image coverage from same-array-entity co-occurrence."""
    require(context, "Dataset")
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).pair(left, right)))


def inspect_array(context, *, path):
    """Inspect array cardinality; multiple findings are valid data."""
    require(context, "Dataset")
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).array(path)))


def inspect_protocol_capacity(context, *, protocol):
    """Explain exact eligibility using the same resolver as Generation."""
    require(context, "Dataset")
    protocol = {**protocol, "sample_conditions": _scope_conditions(context, _tree(context))}
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).capacity(protocol)))


def _subset(profile, fields):
    unknown = set(fields) - set(profile["fields"])
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    return {
        "total_images": profile["total_images"],
        "fields": {key: profile["fields"][key] for key in fields},
        "structures": [
            item
            for item in profile["structures"]
            if any(field.startswith(item["path"]) for field in fields)
        ],
    }


def list_materials(context, *, offset=0, limit=20, query=""):
    """Page available run-local medical references."""
    files = [
        m
        for m in discover_medical_materials(get_settings().paths.materials_root)
        if query.casefold() in m["file"].casefold()
    ]
    return result(
        "success", json.dumps({"total": len(files), "materials": files[offset : offset + limit]})
    )


def read_material(context, *, file):
    """Read one PDF's metadata and page count without dumping its contents."""
    from pypdf import PdfReader

    available = {m["file"] for m in discover_medical_materials(get_settings().paths.materials_root)}
    if file not in available:
        raise ValueError("Unknown PDF")
    reader = PdfReader(get_settings().paths.materials_root / file)
    return result(
        "success",
        json.dumps(
            {"file": file, "pages": len(reader.pages), "metadata": str(reader.metadata)[:2000]},
            ensure_ascii=False,
        ),
    )


def select_materials(context, *, files):
    """Select medical references and retain an immutable copy for this run."""
    if files:
        from medbenchagent.runtime.preflight import rag_capability
        capability = rag_capability(get_settings())
        if not capability["available"]:
            raise ValueError("RAG unavailable: " + "; ".join(capability["reasons"]))
    available = (
        {m["file"] for m in discover_medical_materials(get_settings().paths.materials_root)}
        if files
        else set()
    )
    if not set(files) <= available:
        raise ValueError("Select existing PDF paths")
    files = list(dict.fromkeys(files))
    if files == context.get("selected_material_files"):
        require(context, "Materials")
        return result("success", "These references are already selected")
    root = Path(context["operation_directory"]) / "materials"
    root.mkdir(parents=True, exist_ok=True)
    for file in files:
        destination = root / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(get_settings().paths.materials_root / file, destination)
    versions = publish(context, "Materials", {"root": str(root), "files": files}, ())
    return result(
        "success",
        "References selected; reconcile affected planning evidence",
        selected_material_files=files,
        materials_root=str(root),
        **versions,
    )


def _rag(context):
    require(context, "Materials")
    return LocalPdfRag(
        context["materials_root"],
        Path(context["run_directory"]) / ".runtime/rag",
        context["selected_material_files"],
    )


def search_materials(context, *, query, limit=None):
    """Retrieve relevant paragraphs from selected references."""
    chunks = _rag(context).retrieve(query, limit)
    entries = []
    for chunk in chunks:
        ref = save_evidence(context, "search_materials", {"query": query}, chunk, ("Materials",), kind="material")
        entries.append({**chunk, "reference": ref})
    return result("success", json.dumps(entries, ensure_ascii=False)[:6000])


def initialize_planning(context: TaskPlanningContext):
    """Initialize empty optional materials only when retrieval cannot be used."""
    from medbenchagent.runtime.preflight import rag_capability
    context["rag_capability"] = rag_capability(get_settings())
    if not context["rag_capability"]["available"]:
        try:
            require(context, "Materials")
        except ValueError:
            context.update(select_materials(context, files=[])["updates"])
    return context


def read_dataset_document(context, *, path, offset=0, page=1):
    """Read a frozen source_id/path document; returned inspection_reference grounds semantic_basis."""
    from medbenchagent.data.documentation import read_document
    require(context, "Dataset")
    source, separator, relative = path.partition("/")
    if not separator:
        raise ValueError("Use a frozen source ID and relative documentation path")
    value = read_document(Path(context["annotation_path"]).parent, source, relative, page=page, offset=offset)
    return inspected(context, "read_dataset_document", {"path": path, "offset": offset, "page": page}, value, ("Dataset",))


def _validate_semantic_basis(context, protocol, basis):
    from medbenchagent.planning.tree import SemanticBasis
    checked = []
    for item in basis:
        item = SemanticBasis.model_validate(item).model_dump()
        if item["field"] not in protocol["data_binding"]["supporting_fields"]:
            raise ValueError("Semantic basis must reference a bound annotation field")
        reference = item["reference"]
        receipt = read_evidence_record(context, reference)
        expected = {"material": "material", "document": "document", "user": "user"}[item["source"]]
        if receipt["kind"] != expected:
            raise ValueError("Semantic basis source differs from its evidence kind")
        if expected == "user":
            issue = context.get("issues", {}).get(receipt["value"].get("issue_id"), {})
            if (issue.get("status") != "resolved" or issue.get("kind") != "semantic" or issue.get("answer_reference") != reference
                    or issue.get("role") != "TaskPlanningAgent" or issue.get("dataset_id") != context.get("dataset_scope")):
                raise ValueError("User semantic evidence must reference a resolved current decision")
        text = receipt["value"].get("text", "")
        if item["quote"] not in text:
            raise ValueError("Semantic basis requires an exact quotation from current evidence")
        checked.append(item)
    return checked


def _validate_leaf_grounding(context, leaf):
    from medbenchagent.planning.tree import validate_mapping_grounding
    basis = _validate_semantic_basis(context, leaf["protocol"], leaf.get("semantic_basis", []))
    validate_mapping_grounding(leaf["protocol"], leaf.get("mapping_grounding", []), basis)


def apply_template_feedback(context, feedback):
    """Reopen a plan once per feedback version, charging the existing planning budget."""
    marker = Path(context["operation_directory"]) / "template_feedback.json"
    if marker.is_file():
        saved = _read(marker)
        if saved["feedback"] != feedback:
            raise ValueError("Template feedback changed during replay")
        return saved["outcome"]
    tree = _tree(context)
    if tree.get("template_feedback_id") != feedback["feedback_id"]:
        if tree["reworks_used"] >= context.get("limits", {}).get("max_reworks", 3):
            return result("blocked", "Planning rework budget exhausted after template review")
        tree["reworks_used"] += 1
        tree["correction_open"] = True
        tree["review"] = {}
        tree["revision"] += 1
        tree["template_feedback_id"] = feedback["feedback_id"]
        for issue in feedback["issues"]:
            leaf = tree["tasks"][issue["leaf_id"]]
            tree["concepts"][leaf["concept_id"]]["status"] = "pending"
    outcome = _persist(context, tree)
    outcome["updates"]["template_feedback"] = feedback
    atomic_write_json(marker, {"feedback": feedback, "outcome": outcome})
    return outcome


def _tree(context):
    previous = {}
    if context.get("draft_path"):
        previous = _read(context["draft_path"])
        _check_draft(previous)
        try:
            require(context, "PlanningDraft")
        except ValueError:
            # Keep identities and draft work; every decision must be revalidated.
            previous["benchmark_info"] = dict(context["benchmark_info"])
            previous["scope_resolutions"] = []
            previous["scope_semantic_basis"] = []
            previous["dataset"] = {"dataset_id": context.get("selected_dataset_id")}
            previous["revision"] += 1
            previous["review"] = {}
            previous["correction_open"] = False
            previous["discovery_complete"] = False
            previous.pop("coverage", None)
            for concept in previous["concepts"].values():
                concept["status"] = "pending"
            for proposal in previous["proposals"].values():
                proposal["decision"] = None
            _check_draft(previous)
            return previous
        else:
            return previous
    tree = _base(context)
    for concept in tree["concepts"].values():
        concept.update(status="pending", reason="")
    tree.update(
        tasks={},
        revision=0,
        proposals={},
        discovery_complete=False,
        review={},
        reworks_used=previous.get("reworks_used", 0),
        correction_open=False,
        retired_ids=previous.get("retired_ids", []) + list(previous.get("tasks", {})),
    )
    _check_draft(tree)
    return tree


def _check_draft(tree):
    if issues := validate_draft(tree):
        raise ValueError(json.dumps(issues, ensure_ascii=False))


def _persist(context, tree, *, changed=False):
    require(context, "Dataset", "Requirement", "Materials")
    if changed:
        review = tree.get("review", {})
        if review and not tree["correction_open"]:
            if tree["reworks_used"] >= context.get("limits", {}).get("max_reworks", 3):
                return result(
                    "blocked",
                    "Planning rework budget exhausted",
                    unresolved_questions=review.get("issues", []),
                )
            tree["reworks_used"] += 1
            tree["correction_open"] = True
        tree["revision"] += 1
    _check_draft(tree)
    path = Path(context["operation_directory"]) / "planning_tree.json"
    atomic_write_json(path, tree)
    versions = publish(
        context,
        "PlanningDraft",
        {"path": str(path)},
        ("Dataset", "DatasetCard", "Requirement", "Materials", "PlanningConfig", "PlanningDelegation"),
    )
    return result(
        "success",
        "Planning tree saved",
        draft_path=str(path),
        readiness={"planning": False},
        artifacts={"draft": str(path)},
        **versions,
    )


def read_node(context, *, node_id):
    """Read one concept, leaf or extension proposal by its opaque identity."""
    tree = _tree(context)
    for section in ("concepts", "tasks", "proposals"):
        if node_id in tree[section]:
            return result("success", json.dumps(tree[section][node_id], ensure_ascii=False))
    raise ValueError("Unknown node")


def read_proposal(context, *, node_id):
    """Read the full extension proposal and decision, including accepted Concept proposals."""
    tree = _tree(context)
    if node_id not in tree["proposals"]:
        raise ValueError("Unknown extension proposal")
    return result("success", json.dumps(tree["proposals"][node_id], ensure_ascii=False))


def _validate_evidence(context, evidence):
    for item in evidence:
        receipt = read_evidence_record(context, item.get("reference", ""))
        if receipt["kind"] != "material" or any(receipt["value"].get(k) != item[k] for k in ("pdf", "page", "chunk_id")):
            raise ValueError("Medical evidence must reference a retrieved paragraph")


def save_leaf(context, *, leaf_id, concept_id, description, protocol, evidence, semantic_basis=None, mapping_grounding=None):
    """Create or revise a stable leaf identity, including reparenting; execution is entirely Protocol-defined."""
    tree = _tree(context)
    protocol = {**protocol, "sample_conditions": _scope_conditions(context, tree)}
    if (
        concept_id not in tree["concepts"]
        or leaf_id in tree["concepts"]
        or leaf_id in tree["retired_ids"]
    ):
        raise ValueError("Unknown concept or unavailable leaf identity")
    _validate_evidence(context, evidence)
    _subset(_profile(context), protocol["data_binding"]["supporting_fields"])
    capacity = validate_dataset_protocol(protocol, iter_records(context["dataset_access_manifest"]))
    if capacity <= 0:
        raise ValueError("Leaf requires positive eligible_count; revise the Protocol or mark the Concept unsupported")
    basis = _validate_semantic_basis(context, protocol, semantic_basis or [])
    from medbenchagent.planning.tree import validate_mapping_grounding
    grounding = validate_mapping_grounding(protocol, mapping_grounding or [], basis)
    previous = tree["tasks"].get(leaf_id)
    if previous and previous["concept_id"] != concept_id:
        tree["concepts"][previous["concept_id"]]["status"] = "pending"
    tree["tasks"][leaf_id] = {
        "concept_id": concept_id,
        "eligible_count": capacity,
        "description": description,
        "protocol": protocol,
        "evidence": evidence,
        "semantic_basis": basis,
        "mapping_grounding": grounding,
        "category": tree["concepts"][concept_id]["category"],
    }
    tree["concepts"][concept_id]["status"] = "pending"
    output = _persist(context, tree, changed=True)
    output["affected_node_ids"] = [leaf_id, concept_id]
    return output


def remove_leaf(context, *, node_id):
    """Remove a duplicate leaf; merge by revising the survivor and removing the redundant identity."""
    tree = _tree(context)
    if node_id not in tree["tasks"]:
        raise ValueError("Unknown leaf")
    leaf = tree["tasks"].pop(node_id)
    tree["retired_ids"].append(node_id)
    tree["concepts"][leaf["concept_id"]]["status"] = "pending"
    return _persist(context, tree, changed=True)


def complete_concept(context, *, concept_id, status, reason, inspection_references=None):
    """Close a concept after designing its leaves or establish that annotation cannot support it."""
    tree = _tree(context)
    if concept_id not in tree["concepts"]:
        raise ValueError("Unknown concept")
    children = [leaf for leaf in tree["tasks"].values() if leaf["concept_id"] == concept_id]
    if (status == "complete") != bool(children):
        raise ValueError("Complete concepts require leaves; unsupported concepts must have none")
    if children:
        for leaf in children:
            _subset(_profile(context), leaf["protocol"]["data_binding"]["supporting_fields"])
            capacity = validate_dataset_protocol(leaf["protocol"], iter_records(context["dataset_access_manifest"]))
            if capacity <= 0:
                raise ValueError("Remove or revise zero-capacity leaves before completing the Concept")
            leaf["eligible_count"] = capacity
            _validate_evidence(context, leaf["evidence"])
            _validate_leaf_grounding(context, leaf)
    references = inspection_references or []
    if status == "unsupported":
        _unsupported_evidence(context, references)
    tree["concepts"][concept_id].update(status=status, reason=reason, inspection_references=references)
    return _persist(context, tree, changed=True)


def _unsupported_evidence(context, references):
    if not references:
        raise ValueError("Unsupported concepts require inspection references")
    receipts = [inspection(context, ref) for ref in references]
    if any(r["operation"] not in {"list_fields", "read_fields", "read_field_values", "read_dataset_document"} for r in receipts):
        raise ValueError("Unsupported concepts require field or document inspection")
    return receipts


def _concept_evidence(context, tree):
    return {key: _unsupported_evidence(context, concept.get("inspection_references", []))
            for key, concept in tree["concepts"].items() if concept["status"] == "unsupported"}


def coverage_gap(context):
    """After core planning, inspect annotation coverage and propose genuinely uncovered capabilities."""
    tree = _tree(context)
    if not concepts_complete(tree, core_only=True):
        raise ValueError("Complete core planning first")
    used = {
        p
        for leaf in tree["tasks"].values()
        for p in leaf["protocol"]["data_binding"]["supporting_fields"]
    }
    tree["coverage"] = {
        "used_fields": sorted(used),
        "unused_fields": sorted(set(_profile(context)["fields"]) - used),
    }
    outcome = _persist(context, tree)
    outcome["summary"] = json.dumps(tree["coverage"], ensure_ascii=False)
    return outcome


def propose_extension(context, **proposal):
    """Validate a late extension proposal using annotation facts and explicit semantic criteria."""
    tree = _tree(context)
    if not concepts_complete(tree, core_only=True) or tree.get("coverage") is None:
        raise ValueError("Inspect coverage after core planning first")
    key = proposal["concept_id"]
    if (
        key in tree["retired_ids"]
        or key in tree["tasks"]
        or (key in tree["concepts"] and tree["concepts"][key]["origin"] == "core")
    ):
        raise ValueError("Identity already in use")
    _subset(_profile(context), proposal["fields"])
    if not all(_profile(context)["fields"][p]["non_empty_images"] > 0 for p in proposal["fields"]):
        raise ValueError("Extension requires direct nonempty annotation evidence")
    _validate_evidence(context, proposal["evidence"])
    decision = complete_json(
        'Review the extension against annotation evidence and the benchmark goal. Return {"accepted":bool,"reason":str}. '
        "Accept only a distinct clinically meaningful evaluation capability relevant to the goal that cannot fit any existing concept; "
        "a new descriptor or renamed leaf is not a new concept. Do not invent labels.",
        json.dumps(
            {
                "proposal": proposal,
                "requirement": context["benchmark_info"],
                "concepts": tree["concepts"],
                "leaves": tree["tasks"],
                "dataset": _subset(_profile(context), proposal["fields"]),
            },
            ensure_ascii=False,
        ),
    )
    decision = ExtensionDecision.model_validate(decision).model_dump()
    tree["proposals"][key] = {**proposal, "decision": decision}
    if decision["accepted"]:
        tree["concepts"][key] = {
            "description": proposal["description"],
            "category": proposal["category"],
            "origin": "extension",
            "status": "pending",
            "reason": decision["reason"],
        }
        for leaf in tree["tasks"].values():
            if leaf["concept_id"] == key:
                leaf["category"] = proposal["category"]
    elif key in tree["concepts"]:
        if any(leaf["concept_id"] == key for leaf in tree["tasks"].values()):
            raise ValueError("Rejected extension still has leaves; reparent or remove them first")
        del tree["concepts"][key]
    tree["discovery_complete"] = False
    return _persist(context, tree, changed=True)


def finish_discovery(context):
    """Record that coverage gaps and all proposed extensions have been resolved."""
    tree = _tree(context)
    if tree.get("coverage") is None or not concepts_complete(tree):
        raise ValueError("Inspect coverage and finish all accepted concepts first")
    if any(proposal.get("decision") is None for proposal in tree["proposals"].values()):
        raise ValueError("Revalidate extension proposals against current inputs")
    tree["discovery_complete"] = True
    return _persist(context, tree, changed=True)


def _validate_template_repair(context, tree):
    from medbenchagent.planning.protocol import template_specification
    for issue in context.get("template_feedback", {}).get("issues", []):
        leaf = tree["tasks"].get(issue["leaf_id"])
        if leaf and template_specification(leaf["protocol"], tree["benchmark_info"]["language"]).model_dump() == issue["specification"]:
            raise ValueError(f"{issue['leaf_id']}: problem not fixed; template specification is unchanged. {issue['reason']}")


def _validate_scope_decisions(context, tree, basis):
    for resolution in tree.get("scope_resolutions", []):
        if resolution["status"] == "unsupported":
            _unsupported_evidence(context, resolution["inspection_references"])
        if resolution.get("modality_bindings"):
            field = resolution["conditions"][0]["field"]
            for binding in resolution["modality_bindings"]:
                direct = all(isinstance(v, str) and v.strip().casefold() == binding["modality"].strip().casefold() for v in binding["values"])
                eligible = {b["reference"] for b in basis if b["field"] == field and (b["source"] in {"document", "user"} or b["dataset_specific"])}
                refs = set(binding["evidence_references"])
                if (not direct and not refs) or not refs <= eligible:
                    raise ValueError("Modality recoding requires current dataset-specific semantic evidence")


def _validate_scope_evidence(context, tree):
    _validate_scope_decisions(context, tree, tree.get("scope_semantic_basis", []))
    validate_scope_plan(tree, context.get("sample_scope", []))
    conditions = [c for r in tree.get("scope_resolutions", []) for c in r["conditions"]]
    fields = list(dict.fromkeys(c["field"] for c in conditions))
    _subset(_profile(context), fields)
    _validate_semantic_basis(context, {"data_binding": {"supporting_fields": fields}}, tree.get("scope_semantic_basis", []))
    return fields


def review_tree(context):
    """Review full mappings and source evidence; corrections consume persistent rework budget."""
    tree = _tree(context)
    concept_evidence = _concept_evidence(context, tree)
    scope_fields = _validate_scope_evidence(context, tree)
    _validate_template_repair(context, tree)
    if not concepts_complete(tree) or not tree["discovery_complete"]:
        raise ValueError("Finish concepts and extension discovery first")
    if tree.get("review", {}).get("revision") == tree["revision"]:
        return _review_result(context, tree, context["draft_path"])
    leaves = {}
    for key, leaf in tree["tasks"].items():
        protocol = leaf["protocol"]
        fields = protocol["data_binding"]["supporting_fields"]
        _validate_leaf_grounding(context, leaf)
        leaves[key] = {**leaf, "annotation_evidence": _subset(_profile(context), fields),
                       "field_values": {field: DatasetQuery(context["dataset_access_manifest"]).values(field, limit=50)
                                        for field in fields},
                       "material_passages": [read_evidence_record(context, e["reference"])["value"]
                                             for e in leaf["evidence"]]}
    review = complete_json(
        "Review concept hierarchy, duplicated semantics, overlapping leaves, missing siblings and conflicting Protocols. "
        "Check EVERY raw_value-to-option mapping and mapping_grounding against annotation facts and quoted semantic_basis. "
        "Check EVERY sample scope resolution against the original user restriction, actual fields and scope_semantic_basis. "
        "Scope text codes and abbreviations need dataset-specific evidence just like answer mappings. "
        "Do not treat a disease evaluation topic as a positive-case inclusion criterion. "
        "Direct mappings must be genuinely self-describing, not opaque codes or abbreviations disguised as identity mappings. "
        "Check that material marked dataset_specific actually documents THIS dataset encoding; a flag alone is not evidence. "
        "Opaque codes, abbreviations and ambiguous labels require dataset-specific documented meaning or a user decision; "
        "medical convention is not evidence of a dataset's encoding. Missing or contradictory grounding requires revise "
        "with a specific request to obtain evidence or ask_user; never approve a guess. "
        "Check unsupported reasons against concept_inspections; a reference proves inspection, not semantic correctness. "
        "Card is a search hint, not proof. Unsupported concepts are valid: never add leaves merely to fill the Blueprint. "
        'Return {"issues":[{"action":"merge|remove|reparent|revise|add_leaf","node_ids":["id"],"reason":"specific correction"}]}. '
        "An empty list means approved. Propose targeted changes, never rewrite the entire tree.",
        json.dumps(
            {
                "requirement": context["benchmark_info"],
                "delegation": context.get("delegation"),
                "concepts": tree["concepts"],
                "concept_inspections": concept_evidence,
                "leaves": leaves,
                "sample_scope": context.get("sample_scope", []),
                "scope_resolutions": tree.get("scope_resolutions", []),
                "scope_semantic_basis": tree.get("scope_semantic_basis", []),
                "scope_field_values": {field: DatasetQuery(context["dataset_access_manifest"]).values(field, limit=50) for field in scope_fields},
            },
            ensure_ascii=False,
        ),
    )
    issues = GlobalReview.model_validate(review).model_dump()["issues"]
    if any(not set(i["node_ids"]) <= set(tree["concepts"]) | set(tree["tasks"]) for i in issues):
        raise ValueError("Review references unknown nodes")
    tree["review"] = {"revision": tree["revision"], "issues": issues}
    tree["correction_open"] = False
    outcome = _persist(context, tree)
    review_result = _review_result(context, tree, outcome["updates"]["draft_path"])
    review_result["updates"] = outcome["updates"]
    return review_result


def _review_result(context, tree, draft_path):
    issues = tree["review"]["issues"]
    exhausted = issues and tree["reworks_used"] >= context.get("limits", {}).get("max_reworks", 3)
    return result(
        "blocked" if exhausted else "success",
        json.dumps(tree["review"], ensure_ascii=False),
        unresolved_questions=issues,
        artifacts={"draft": draft_path},
        readiness={"planning": False},
    )


def finish(context):
    """Publish the reviewed dataset-local Plan with exact capacities, including unsupported datasets."""
    tree = _tree(context)
    _concept_evidence(context, tree)
    _validate_scope_evidence(context, tree)
    _validate_template_repair(context, tree)
    for leaf in tree["tasks"].values():
        leaf["eligible_count"] = validate_dataset_protocol(leaf["protocol"], iter_records(context["dataset_access_manifest"]))
        _validate_leaf_grounding(context, leaf)
    errors = validate_plan(tree)
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    path = Path(context["operation_directory"]) / "blueprint.json"
    atomic_write_json(path, tree)
    versions = publish(
        context,
        "Plan",
        {"blueprint": str(path)},
        ("Requirement", "Dataset", "DatasetCard", "Materials", "PlanningConfig", "PlanningDraft", "PlanningDelegation"),
    )
    return result(
        "complete",
        "Plan published" + ("; unsupported sample scope: " + "; ".join(
            r["reason"] for r in tree.get("scope_resolutions", []) if r["status"] == "unsupported")
            if any(r["status"] == "unsupported" for r in tree.get("scope_resolutions", [])) else ""),
        working_blueprint_path=str(path),
        readiness={"planning": True},
        artifacts={"plan": str(path)},
        affected_node_ids=list(tree["tasks"]),

        **versions,
    )


def describe(context: TaskPlanningContext):
    try:
        tree = _tree(context)
    except ValueError as exc:
        return {
            "validation_error": str(exc),
            "requirement": context.get("benchmark_info"),
            "stale_draft": True,
            "versions": context.get("artifact_versions", {}),
        }
    return {
        "rag_capability": context.get("rag_capability"),
        "dataset_summary": _profile(context).get("summary", {}) if context.get("dataset_access_manifest") else {},
        "requirement": context.get("benchmark_info"),
        "sample_scope": context.get("sample_scope", []),
        "scope_resolutions": tree.get("scope_resolutions", []),
        "concepts": {
            k: {"description": v["description"], "origin": v["origin"], "status": v["status"]}
            for k, v in tree["concepts"].items()
        },
        "leaves": {
            k: {"concept_id": v["concept_id"], "description": v["description"]}
            for k, v in tree["tasks"].items()
        },
        "review": tree["review"],
        "reworks_used": tree["reworks_used"],
        "discovery_complete": tree["discovery_complete"],
        "dataset_card": context.get("dataset_card_summary"),
        "feedback": context.get("feedback"),
        "delegation": context.get("delegation"),
        "user_requirement": context.get("user_requirement"),
        "template_feedback": context.get("template_feedback"),
        "inspection_references": inspection_index(context),
        "document_discovery": {"tool": "list_dataset_documents", "storage": "frozen bundle documentation"},
        "clarification": context.get("clarification"),
        "dataset": context.get("selected_dataset_id"),
    }


def available_tools(context):
    from medbenchagent.runtime.preflight import rag_capability
    rag = context.get("rag_capability") or rag_capability(get_settings())
    available = {"ask_user", "select_materials"}
    if rag["available"]:
        available |= {"list_materials", "read_material"}
    try:
        require(context, "Dataset")
    except ValueError:
        return available
    available |= {"list_fields", "read_fields", "read_field_values", "read_node", "read_proposal", "inspect_field_by_source", "sample_records", "inspect_field_pair", "inspect_array", "inspect_protocol_capacity", "read_dataset_document", "list_dataset_documents"}
    try:
        require(context, "Materials")
    except ValueError:
        return available
    try:
        tree = _tree(context)
    except ValueError:
        return available
    available |= {"save_leaf", "remove_leaf", "complete_concept", "resolve_sample_scope"}
    if rag["available"] and context.get("selected_material_files"):
        available.add("search_materials")
    if concepts_complete(tree, core_only=True):
        available |= {"coverage_gap", "propose_extension", "finish_discovery"}
    if concepts_complete(tree) and tree["discovery_complete"]:
        available.add("review_tree")
    if not validate_plan(tree):
        available.add("finish")
    return available


def compile_task_planning_agent(*, checkpointer=None):
    from medbenchagent.data.query_contracts import SingleField, Sample, Pair, Array, Capacity
    functions = (
        resolve_sample_scope,
        list_fields,
        read_fields,
        read_field_values,
        inspect_field_by_source, sample_records, inspect_field_pair, inspect_array, inspect_protocol_capacity,
        list_materials,
        read_material,
        read_dataset_document,
        list_dataset_documents,
        select_materials,
        search_materials,
        read_node,
        read_proposal,
        save_leaf,
        remove_leaf,
        complete_concept,
        coverage_gap,
        propose_extension,
        finish_discovery,
        review_tree,
    )
    return compile_agent(
        "TaskPlanningAgent",
        "Build a two-level Medical Concept -> executable Leaf plan. The dataset is already assigned and prepared. Select references (an empty selection is allowed). "
        "Modality scope needs one in/not_in condition and modality_bindings covering each requested modality. "
        "Coded/renamed values need matching semantic_basis references. Unsupported scope requires inspection_references. "
        "Unsupported decisions require inspection_references returned by field/document tools; reuse current evidence when relevant. "
        "Understand each core concept using field inspection and optional RAG BEFORE and during leaf expansion. "
        "Create opaque stable leaf IDs at runtime; never infer execution from IDs. All behavior belongs to Protocol. "
        "Resolve every sample_scope using resolve_sample_scope before creating leaves. Use actual scalar fields and exact "
        "values with in/not_in; missing fields fail selection. Unsupported arrays or ungrounded meanings must be reported, "
        "never replaced by prose constraints. Supply dataset-specific semantic_basis for interpreted scope codes. "
        "Finish every core concept, inspect coverage gaps, propose only genuinely new capabilities, and finish accepted extensions. "
        "RAG can be called at any concept, leaf, protocol or extension step, with revised queries as needed; annotations alone establish labels. "
        "Finish discovery, review globally, make targeted corrections within the rework budget, then publish the dataset-local plan and its exact capacities. "
        "Do not select datasets or allocate sample counts. Use ask_user for unresolved source semantics. "
        "Dataset Card is only a triage/search hint; inspect Profile, exact annotation values and structures for evidence. "
        "Unsupported concepts are valid. Original requirements and benchmark_info take precedence over delegation; "
        "delegation organizes work but cannot introduce new priorities, exclusions or research constraints. "
        "For opaque codes or ambiguous abbreviations first discover local documents using list_dataset_documents, then read_dataset_document, "
        "search_materials or ask_user; never infer code meanings from medical knowledge. Store field interpretation and "
        "exact source quotation in semantic_basis (user reference is the evidence reference of the resolved answer; list_evidence/read_evidence recover prior facts). "
        "Supply mapping_grounding for EVERY option mapping: field, raw_value, option, mode, interpretation and evidence_references. "
        "Direct means the same self-describing annotation string (whitespace/case may differ); numeric codes, abbreviations, "
        "merging or changing meaning require interpreted mode and dataset-specific evidence references. "
        "Address template_feedback before republishing a revised plan. "
        "When revising after feedback, preserve unchanged leaf identities and protocols. Finish only when validation permits.",
        {f.__name__: f for f in functions},
        describe,
        finish,
        args_schemas={
            "resolve_sample_scope": ScopeArguments,
            "read_dataset_document": DocumentArguments,
            "list_dataset_documents": PageArguments,
            "list_fields": PageArguments,
            "read_fields": FieldArguments,
            "read_field_values": FieldValuesArguments,
            "inspect_field_by_source": SingleField,
            "sample_records": Sample,
            "inspect_field_pair": Pair,
            "inspect_array": Array,
            "inspect_protocol_capacity": Capacity,
            "list_materials": PageArguments,
            "read_material": MaterialArguments,
            "select_materials": MaterialsArguments,
            "search_materials": SearchArguments,
            "read_node": NodeArguments,
            "read_proposal": NodeArguments,
            "remove_leaf": NodeArguments,
            "save_leaf": LeafArguments,
            "complete_concept": ConceptArguments,
            "propose_extension": ExtensionArguments,
        },
        available_tools=available_tools,
        read_tools=frozenset({"list_fields", "read_fields", "read_field_values", "inspect_field_by_source", "sample_records",
                              "inspect_field_pair", "inspect_array", "inspect_protocol_capacity", "list_materials", "read_material",
                              "search_materials", "read_dataset_document", "list_dataset_documents", "read_node", "read_proposal"}),
        initialize_context=initialize_planning,
        checkpointer=checkpointer,
    )
