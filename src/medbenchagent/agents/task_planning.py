"""Incremental two-level medical concept and leaf planning."""

from __future__ import annotations
import json
import shutil
from pathlib import Path
from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.data.dataset_catalog import (
    dataset_by_id,
    load_dataset_catalog,
    recommend_dataset,
    resolve_annotation_path,
)
from medbenchagent.runtime.dependencies import publish, require, snapshot_dataset
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.rag.pdf import LocalPdfRag, RagUnavailable, discover_medical_materials
from medbenchagent.settings import get_settings
from medbenchagent.runtime.tool_contracts import (
    SelectDatasetArguments,
    ProfileArguments,
    PageArguments,
    FieldArguments,
    MaterialArguments,
    MaterialsArguments,
    SearchArguments,
)
from medbenchagent.planning.tree import (
    LeafArguments,
    ConceptArguments,
    NodeArguments,
    ExtensionArguments,
    WeightsArguments,
    FieldValuesArguments,
    concepts_complete,
    validate_plan,
    ExtensionDecision,
    GlobalReview,
)
from medbenchagent.planning.protocol import validate_dataset_protocol
from medbenchagent.planning.sample_counts import allocate_leaf_counts
from medbenchagent.runtime.llm import complete_json
from medbenchagent.data.validation import load_annotation_jsonl


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


def select_dataset(context, *, dataset_id=None, annotation_path=None):
    """Select standard inputs; omit both arguments to recommend catalog data."""
    require(context, "Requirement")
    if (
        dataset_id is not None
        and dataset_id == context.get("selected_dataset_id")
        and (
            annotation_path is None
            or annotation_path == context.get("dataset_entry", {}).get("annotation_path")
        )
    ):
        require(context, "DatasetSelection")
        return result("success", "This dataset is already selected in the run")
    if annotation_path:
        if dataset_id is None:
            raise ValueError("Direct input requires dataset_id")
        entry = {
            "id": dataset_id,
            "name": dataset_id,
            "annotation_path": annotation_path,
            "description": "Standard local dataset",
            "available": True,
        }
    else:
        catalog = load_dataset_catalog()
        dataset_id = (
            dataset_id or recommend_dataset(context["benchmark_info"], catalog)["dataset_id"]
        )
        entry = dataset_by_id(catalog, dataset_id)
    path = snapshot_dataset(
        resolve_annotation_path(entry), Path(context["operation_directory"]) / "input"
    )
    versions = publish(
        context, "DatasetSelection", {"path": str(path), "dataset_id": dataset_id}, ()
    )
    return result(
        "success",
        "Dataset input snapshot selected; build its profile next",
        selected_dataset_id=dataset_id,
        annotation_path=str(path),
        dataset_entry=entry,
        dataset_access_manifest=None,
        **versions,
    )


def profile_dataset(context, *, revision="1", refresh=False):
    """Build this run's dataset cache; refresh creates a new immutable Dataset version."""
    require(context, "DatasetSelection")
    if not refresh and context.get("dataset_access_manifest"):
        require(context, "Dataset")
        return result("success", "Current Dataset is already prepared")
    profile, access = prepare_annotation_dataset(
        context["annotation_path"],
        Path(context["operation_directory"]) / "cache",
        revision=revision,
    )
    from medbenchagent.data.validation import load_annotation_jsonl, validate_dataset
    from medbenchagent.data.image_assets import load_image_manifest

    validate_dataset(
        load_annotation_jsonl(context["annotation_path"]),
        load_image_manifest(Path(context["annotation_path"]).parent / "image_manifest.jsonl"),
    )
    versions = publish(
        context,
        "Dataset",
        {"annotation": context["annotation_path"], "access": access},
        ("DataConfig", "DatasetSelection"),
    )
    return result(
        "success",
        f"Dataset ready: {profile['total_images']} images, {len(profile['fields'])} fields",
        dataset_access_manifest=access,
        **versions,
    )


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
    return result("success", json.dumps(_subset(_profile(context), fields), ensure_ascii=False))


def read_field_values(context, *, field, offset=0, limit=20):
    """Page exact annotation values and counts; profile examples may be truncated."""
    from collections import Counter
    from medbenchagent.generation.context import scalar_evidence_candidates

    _subset(_profile(context), [field])
    counts = Counter(
        json.dumps(v["value"], ensure_ascii=False)
        for row in load_annotation_jsonl(context["annotation_path"])
        for v in scalar_evidence_candidates(row, field)
    )
    rows = [{"value": json.loads(value), "count": count} for value, count in sorted(counts.items())]
    return result(
        "success",
        json.dumps(
            {"total": len(rows), "values": rows[offset : offset + limit]}, ensure_ascii=False
        ),
    )


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
    for chunk in chunks:
        ref = {key: chunk[key] for key in ("pdf", "page", "chunk_id")}
        if ref not in evidence:
            evidence.append(ref)
    return result("success", json.dumps(chunks, ensure_ascii=False), retrieved_evidence=evidence)


def _tree(context):
    previous = {}
    if context.get("draft_path"):
        previous = _read(context["draft_path"])
        try:
            require(context, "PlanningDraft")
        except ValueError:
            # Keep identities and draft work; every decision must be revalidated.
            previous["revision"] += 1
            previous["review"] = {}
            previous["correction_open"] = False
            previous["discovery_complete"] = False
            previous.pop("allocation", None)
            previous.pop("coverage", None)
            for concept in previous["concepts"].values():
                concept["status"] = "pending"
            for proposal in previous["proposals"].values():
                proposal["decision"] = None
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
    return tree


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
        tree.pop("allocation", None)
    path = Path(context["operation_directory"]) / "planning_tree.json"
    atomic_write_json(path, tree)
    versions = publish(
        context,
        "PlanningDraft",
        {"path": str(path)},
        ("Dataset", "Requirement", "Materials", "PlanningConfig"),
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


def _validate_evidence(context, evidence):
    retrieved = context.get("retrieved_evidence", [])
    if any(e not in retrieved for e in evidence):
        raise ValueError("Medical evidence must reference a retrieved paragraph")


def save_leaf(context, *, leaf_id, concept_id, description, protocol, evidence):
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
    validate_dataset_protocol(protocol, load_annotation_jsonl(context["annotation_path"]))
    previous = tree["tasks"].get(leaf_id)
    if previous and previous["concept_id"] != concept_id:
        tree["concepts"][previous["concept_id"]]["status"] = "pending"
    tree["tasks"][leaf_id] = {
        "concept_id": concept_id,
        "description": description,
        "protocol": protocol,
        "evidence": evidence,
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
        records = load_annotation_jsonl(context["annotation_path"])
        for leaf in children:
            _subset(_profile(context), leaf["protocol"]["data_binding"]["supporting_fields"])
            validate_dataset_protocol(leaf["protocol"], records)
            _validate_evidence(context, leaf["evidence"])
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
    if not concepts_complete(tree, core_only=True) or "coverage" not in tree:
        raise ValueError("Inspect coverage after core planning first")
    key = proposal["concept_id"]
    if key in tree["tasks"] or (
        key in tree["concepts"] and tree["concepts"][key]["origin"] == "core"
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
    elif key in tree["concepts"]:
        if any(leaf["concept_id"] == key for leaf in tree["tasks"].values()):
            raise ValueError("Rejected extension still has leaves; reparent or remove them first")
        del tree["concepts"][key]
    tree["discovery_complete"] = False
    return _persist(context, tree, changed=True)


def finish_discovery(context):
    """Record that coverage gaps and all proposed extensions have been resolved."""
    tree = _tree(context)
    if "coverage" not in tree or not concepts_complete(tree):
        raise ValueError("Inspect coverage and finish all accepted concepts first")
    if any(proposal.get("decision") is None for proposal in tree["proposals"].values()):
        raise ValueError("Revalidate extension proposals against current inputs")
    tree["discovery_complete"] = True
    return _persist(context, tree, changed=True)


def review_tree(context):
    """Review compact concept/leaf summaries; correction rounds consume persistent Harness rework budget."""
    tree = _tree(context)
    if not concepts_complete(tree) or not tree["discovery_complete"]:
        raise ValueError("Finish concepts and extension discovery first")
    if tree.get("review", {}).get("revision") == tree["revision"]:
        return result("success", json.dumps(tree["review"]))
    leaves = {}
    for key, leaf in tree["tasks"].items():
        protocol = leaf["protocol"]
        leaves[key] = {
            "concept_id": leaf["concept_id"],
            "description": leaf["description"],
            "execution_type": protocol["execution_type"],
            "metric": protocol["metric"],
            "data_binding": protocol["data_binding"],
            "question_intent": protocol["question_intent"],
            "medical_constraints": protocol["medical_constraints"],
            "options": protocol["answer_contract"]["options"],
            "mapped_value_count": len(protocol["answer_contract"]["value_mapping"]),
        }
    review = complete_json(
        "Review concept hierarchy, duplicated semantics, overlapping leaves, missing siblings and conflicting Protocols. "
        'Return {"issues":[{"action":"merge|remove|reparent|revise|add_leaf","node_ids":["id"],"reason":"specific correction"}]}. '
        "An empty list means approved. Propose targeted changes, never rewrite the entire tree.",
        json.dumps(
            {
                "requirement": context["benchmark_info"],
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
    outcome["summary"] = json.dumps(tree["review"], ensure_ascii=False)
    if issues and tree["reworks_used"] >= context.get("limits", {}).get("max_reworks", 3):
        outcome["status"] = "blocked"
        outcome["unresolved_questions"] = issues
    return outcome


def allocate(context, *, weights):
    """Allocate final relative weights against exact eligible image capacities, then report only final total shortage."""
    tree = _tree(context)
    if tree.get("review", {}).get("revision") != tree["revision"] or tree["review"]["issues"]:
        raise ValueError("Current global review must pass before allocation")
    if not tree["tasks"]:
        return result(
            "blocked", "No executable leaves have eligible data", readiness={"planning": False}
        )
    if set(weights) != set(tree["tasks"]):
        raise ValueError("Provide weights and reasons for exactly all leaves")
    records = load_annotation_jsonl(context["annotation_path"])
    capacities = {
        key: validate_dataset_protocol(leaf["protocol"], records)
        for key, leaf in tree["tasks"].items()
    }
    allocation = allocate_leaf_counts(
        {k: v["weight"] for k, v in weights.items()},
        capacities,
        context["benchmark_info"]["sample_count"],
    )
    allocation.update(
        revision=tree["revision"], weight_reasons={k: v["reason"] for k, v in weights.items()}
    )
    tree["allocation"] = allocation
    for key, count in allocation["counts"].items():
        tree["tasks"][key]["sample_count"] = count
    outcome = _persist(context, tree)
    outcome["summary"] = json.dumps(allocation)
    if not allocation["allocated_count"]:
        outcome["status"] = "blocked"
    return outcome


def finish(context):
    """Publish only a validated, reviewed current Plan; explicit final planning shortage is allowed."""
    tree = _tree(context)
    errors = validate_plan(tree)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(context["operation_directory"]) / "blueprint.json"
    atomic_write_json(path, tree)
    versions = publish(
        context,
        "Plan",
        {"blueprint": str(path)},
        ("Requirement", "Dataset", "Materials", "PlanningConfig", "PlanningDraft"),
    )
    return result(
        "complete",
        "Plan published",
        working_blueprint_path=str(path),
        readiness={"planning": True},
        artifacts={"plan": str(path)},
        affected_node_ids=list(tree["tasks"]),
        warnings=[f"Planning shortfall: {tree['allocation']['planning_shortfall']}"]
        if tree["allocation"]["planning_shortfall"]
        else [],
        **versions,
    )


def describe(context):
    try:
        tree = _tree(context)
    except ValueError:
        return {
            "requirement": context.get("benchmark_info"),
            "stale_draft": True,
            "versions": context.get("artifact_versions", {}),
        }
    return {
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
        "allocation": tree.get("allocation"),
        "feedback": context.get("feedback"),
        "clarification": context.get("clarification"),
        "dataset": context.get("selected_dataset_id"),
    }


def available_tools(context):
    available = {
        "select_dataset",
        "profile_dataset",
        "list_materials",
        "read_material",
        "select_materials",
        "ask_user",
    }
    try:
        require(context, "Dataset")
    except ValueError:
        return available
    available |= {"list_fields", "read_fields", "read_field_values", "read_node"}
    try:
        require(context, "Materials")
    except ValueError:
        return available
    available |= {"search_materials", "save_leaf", "remove_leaf", "complete_concept"}
    tree = _tree(context)
    if concepts_complete(tree, core_only=True):
        available |= {"coverage_gap", "propose_extension", "finish_discovery"}
    if concepts_complete(tree) and tree["discovery_complete"]:
        available.add("review_tree")
    if tree.get("review", {}).get("revision") == tree["revision"] and not tree["review"]["issues"]:
        available.add("allocate")
    if not validate_plan(tree):
        available.add("finish")
    return available


def compile_task_planning_agent(*, checkpointer=None):
    functions = (
        select_dataset,
        profile_dataset,
        list_fields,
        read_fields,
        read_field_values,
        list_materials,
        read_material,
        select_materials,
        search_materials,
        read_node,
        save_leaf,
        remove_leaf,
        complete_concept,
        coverage_gap,
        propose_extension,
        finish_discovery,
        review_tree,
        allocate,
    )
    return compile_agent(
        "TaskPlanningAgent",
        "Build a two-level Medical Concept -> executable Leaf plan. First select/profile data and select references (an empty selection is allowed). "
        "Understand each core concept using field inspection and optional RAG BEFORE and during leaf expansion. "
        "Create opaque stable leaf IDs at runtime; never infer execution from IDs. All behavior belongs to Protocol. "
        "Finish every core concept, inspect coverage gaps, propose only genuinely new capabilities, and finish accepted extensions. "
        "RAG can be called at any concept, leaf, protocol or extension step, with revised queries as needed; annotations alone establish labels. "
        "Finish discovery, review globally, make targeted corrections within the rework budget, then assign relative weights with reasons and allocate. "
        "A zero weight deliberately excludes a leaf from allocation. Use ask_user for unresolved source semantics. "
        "When revising after feedback, preserve unchanged leaf identities and protocols. Finish only when validation permits.",
        {f.__name__: f for f in functions},
        describe,
        finish,
        args_schemas={
            "select_dataset": SelectDatasetArguments,
            "profile_dataset": ProfileArguments,
            "list_fields": PageArguments,
            "read_fields": FieldArguments,
            "read_field_values": FieldValuesArguments,
            "list_materials": PageArguments,
            "read_material": MaterialArguments,
            "select_materials": MaterialsArguments,
            "search_materials": SearchArguments,
            "read_node": NodeArguments,
            "remove_leaf": NodeArguments,
            "save_leaf": LeafArguments,
            "complete_concept": ConceptArguments,
            "propose_extension": ExtensionArguments,
            "allocate": WeightsArguments,
        },
        available_tools=available_tools,
        checkpointer=checkpointer,
    )
