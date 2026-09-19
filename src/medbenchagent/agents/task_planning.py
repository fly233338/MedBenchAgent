"""Incremental two-level medical concept and leaf planning."""

from __future__ import annotations
import json
import shutil
from pathlib import Path
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.rag.pdf import LocalPdfRag, RagUnavailable, discover_medical_materials
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
from medbenchagent.data.query import DatasetQuery, iter_records


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
    return result(
        "success", json.dumps({"total": len(rows), "fields": rows[offset : offset + limit]})
    )


def read_fields(context, *, fields):
    """Read selected field distributions and their object/array structure."""
    from medbenchagent.data.dataset_catalog import resolve_path
    card = _read(resolve_path(context["dataset_entry"]["dataset_card_path"]))
    details = _subset(_profile(context), fields)
    details["field_roles"] = {key: card["semantics"]["field_roles"].get(key) for key in fields}
    return result("success", json.dumps(details, ensure_ascii=False))


def read_field_values(context, *, field, offset=0, limit=20):
    """Page complete exact values from the current Dataset index."""
    require(context, "Dataset")
    return result("success", json.dumps(DatasetQuery(context["dataset_access_manifest"]).values(field, offset, limit), ensure_ascii=False))


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
        retrieved_evidence=[],
        retrieved_passages={},
        **versions,
    )


def _rag(context):
    require(context, "Materials")
    return LocalPdfRag(
        context["materials_root"],
        Path(context["run_directory"]) / ".runtime/rag",
        context["selected_material_files"],
    )


def search_materials(context, *, query, limit=5):
    """Retrieve relevant paragraphs from selected references."""
    try:
        chunks = _rag(context).retrieve(query, limit)
    except RagUnavailable as exc:
        return result("failed", str(exc))
    evidence = list(context.get("retrieved_evidence", []))
    passages = dict(context.get("retrieved_passages", {}))
    for chunk in chunks:
        ref = {key: chunk[key] for key in ("pdf", "page", "chunk_id")}
        if ref not in evidence:
            evidence.append(ref)
        passages[chunk["chunk_id"]] = chunk
    return result("success", json.dumps(chunks, ensure_ascii=False), retrieved_evidence=evidence,
                  retrieved_passages=passages)


def read_dataset_document(context, *, path, offset=0):
    """Read UTF-8 source documentation as source_id/relative/path (see source manifests).

    Use path='dataset_metadata.json' for frozen source quotations. Empty path is not allowed.
    Returned reference and text can ground semantic_basis; quotations must match exactly.
    """
    from medbenchagent.data.validation import source_image_path
    require(context, "Dataset")
    bundle = Path(context["annotation_path"]).parent
    if path == "dataset_metadata.json":
        document = bundle / path
    else:
        source_id, separator, relative = path.partition("/")
        if not separator or source_id not in {p.name for p in (bundle / "source_manifests").glob("*") if p.is_dir()}:
            raise ValueError("Use a registered source ID and relative documentation path")
        source = _read(bundle / "source_manifests" / source_id / "source.json")
        document = source_image_path(source["root"], relative)
    with document.open(encoding="utf-8") as handle:
        handle.read(offset)
        text = handle.read(12000)
    reference = f"{path}:{offset}"
    documents = {**context.get("source_documents", {}), reference: text}
    return result("success", json.dumps({"reference": reference, "text": text}, ensure_ascii=False),
                  source_documents=documents)


def _validate_semantic_basis(context, protocol, basis):
    from medbenchagent.planning.tree import SemanticBasis
    checked = []
    for item in basis:
        item = SemanticBasis.model_validate(item).model_dump()
        if item["field"] not in protocol["data_binding"]["supporting_fields"]:
            raise ValueError("Semantic basis must reference a bound annotation field")
        reference = item["reference"]
        if item["source"] == "material":
            text = context.get("retrieved_passages", {}).get(reference, {}).get("text", "")
        elif item["source"] == "document":
            text = context.get("source_documents", {}).get(reference, "")
        else:
            text = context.get("decisions", {}).get(reference, "")
        if item["quote"] not in text:
            raise ValueError("Semantic basis requires an exact quotation from a retrieved source or user decision")
        checked.append(item)
    return checked


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
    retrieved = context.get("retrieved_evidence", [])
    if any(e not in retrieved for e in evidence):
        raise ValueError("Medical evidence must reference a retrieved paragraph")


def save_leaf(context, *, leaf_id, concept_id, description, protocol, evidence, semantic_basis=None):
    """Create or revise a stable leaf identity, including reparenting; execution is entirely Protocol-defined."""
    tree = _tree(context)
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


def complete_concept(context, *, concept_id, status, reason):
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
            _validate_semantic_basis(context, leaf["protocol"], leaf.get("semantic_basis", []))
    tree["concepts"][concept_id].update(status=status, reason=reason)
    return _persist(context, tree, changed=True)


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


def review_tree(context):
    """Review full mappings and source evidence; corrections consume persistent rework budget."""
    tree = _tree(context)
    if not concepts_complete(tree) or not tree["discovery_complete"]:
        raise ValueError("Finish concepts and extension discovery first")
    if tree.get("review", {}).get("revision") == tree["revision"]:
        return _review_result(context, tree, context["draft_path"])
    leaves = {}
    for key, leaf in tree["tasks"].items():
        protocol = leaf["protocol"]
        fields = protocol["data_binding"]["supporting_fields"]
        _validate_semantic_basis(context, protocol, leaf.get("semantic_basis", []))
        leaves[key] = {**leaf, "annotation_evidence": _subset(_profile(context), fields),
                       "field_values": {field: DatasetQuery(context["dataset_access_manifest"]).values(field, limit=50)
                                        for field in fields},
                       "material_passages": [context.get("retrieved_passages", {}).get(e["chunk_id"], {})
                                             for e in leaf["evidence"]]}
    review = complete_json(
        "Review concept hierarchy, duplicated semantics, overlapping leaves, missing siblings and conflicting Protocols. "
        "Check EVERY raw_value-to-option mapping against annotation facts and quoted semantic_basis. "
        "Opaque codes, abbreviations and ambiguous labels require dataset-specific documented meaning or a user decision; "
        "medical convention is not evidence of a dataset's encoding. Missing or contradictory grounding requires revise "
        "with a specific request to obtain evidence or ask_user; never approve a guess. "
        "Card is a search hint, not proof. Unsupported concepts are valid: never add leaves merely to fill the Blueprint. "
        'Return {"issues":[{"action":"merge|remove|reparent|revise|add_leaf","node_ids":["id"],"reason":"specific correction"}]}. '
        "An empty list means approved. Propose targeted changes, never rewrite the entire tree.",
        json.dumps(
            {
                "requirement": context["benchmark_info"],
                "delegation": context.get("delegation"),
                "concepts": tree["concepts"],
                "leaves": leaves,
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
    for leaf in tree["tasks"].values():
        leaf["eligible_count"] = validate_dataset_protocol(leaf["protocol"], iter_records(context["dataset_access_manifest"]))
        _validate_semantic_basis(context, leaf["protocol"], leaf.get("semantic_basis", []))
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
        "Plan published",
        working_blueprint_path=str(path),
        readiness={"planning": True},
        artifacts={"plan": str(path)},
        affected_node_ids=list(tree["tasks"]),

        **versions,
    )


def describe(context):
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
        "dataset_summary": _profile(context).get("summary", {}) if context.get("dataset_access_manifest") else {},
        "requirement": context.get("benchmark_info"),
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
        "semantic_sources": {"documents": context.get("source_documents", {}),
                             "user_decisions": context.get("decisions", {})},
        "source_ids": [p.name for p in (Path(context["annotation_path"]).parent / "source_manifests").glob("*") if p.is_dir()],
        "clarification": context.get("clarification"),
        "dataset": context.get("selected_dataset_id"),
    }


def available_tools(context):
    available = {
        "list_materials",
        "read_material",
        "select_materials",
        "ask_user",
    }
    try:
        require(context, "Dataset")
    except ValueError:
        return available
    available |= {"list_fields", "read_fields", "read_field_values", "read_node", "read_proposal", "inspect_field_by_source", "sample_records", "inspect_field_pair", "inspect_array", "inspect_protocol_capacity", "read_dataset_document"}
    try:
        require(context, "Materials")
    except ValueError:
        return available
    try:
        tree = _tree(context)
    except ValueError:
        return available
    available |= {"search_materials", "save_leaf", "remove_leaf", "complete_concept"}
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
        list_fields,
        read_fields,
        read_field_values,
        inspect_field_by_source, sample_records, inspect_field_pair, inspect_array, inspect_protocol_capacity,
        list_materials,
        read_material,
        read_dataset_document,
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
        "Understand each core concept using field inspection and optional RAG BEFORE and during leaf expansion. "
        "Create opaque stable leaf IDs at runtime; never infer execution from IDs. All behavior belongs to Protocol. "
        "Finish every core concept, inspect coverage gaps, propose only genuinely new capabilities, and finish accepted extensions. "
        "RAG can be called at any concept, leaf, protocol or extension step, with revised queries as needed; annotations alone establish labels. "
        "Finish discovery, review globally, make targeted corrections within the rework budget, then publish the dataset-local plan and its exact capacities. "
        "Do not select datasets or allocate sample counts. Use ask_user for unresolved source semantics. "
        "Dataset Card is only a triage/search hint; inspect Profile, exact annotation values and structures for evidence. "
        "Unsupported concepts are valid. Follow the structured delegation within the original user's requirements. "
        "For opaque codes or ambiguous abbreviations obtain dataset-specific documentation using read_dataset_document, "
        "search_materials or ask_user; never infer code meanings from medical knowledge. Store field interpretation and "
        "exact source quotation in semantic_basis (user reference is the decision key in semantic_sources.user_decisions). "
        "When revising after feedback, preserve unchanged leaf identities and protocols. Finish only when validation permits.",
        {f.__name__: f for f in functions},
        describe,
        finish,
        args_schemas={
            "read_dataset_document": ReadTextArguments,
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
        checkpointer=checkpointer,
    )
