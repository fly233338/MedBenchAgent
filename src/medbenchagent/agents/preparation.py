"""Source-scoped preparation decisions over the existing execution and artifact runtime."""

import json
from pathlib import Path
from pydantic import Field
from typing import Literal

from medbenchagent.data.contracts import (
    Contract, Text, Source, SourceFingerprint, SemanticPreparationSpec, ExecutionPlan,
    JsonPopulation,
)
from medbenchagent.data.sources import track_files, verify_source, digest_file, compare_outputs, remove_work_directory
from medbenchagent.data.preparation_evidence import semantic_evidence
from medbenchagent.data.preparation import CAPABILITIES
from medbenchagent.runtime.dependencies import publish, require, read_index
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.inspection import inspection, inspected
from medbenchagent.runtime.run_artifacts import atomic_write_json


class FingerprintArguments(Contract):
    inspection_references: list[Text] = Field(min_length=1)


class SpecArguments(Contract):
    spec: SemanticPreparationSpec


class ExecutionArguments(Contract):
    plan: ExecutionPlan


class AdapterArguments(Contract):
    code: Text


class JsonPopulationArguments(Contract):
    population: JsonPopulation


class JsonStructureArguments(Contract):
    path: Text
    collection: str = ""


class CapabilityArguments(Contract):
    inspection_reference: Text
    required_operation: Literal["aggregate_filter", "nested_array_join", "derived_identity"]


def bound(context, *kinds):
    """Check stored content as well as dependency versions before using preparation artifacts."""
    records = require(context, *kinds)
    index, seen = read_index(context)["versions"], set()
    def verify(record):
        if record["id"] in seen:
            return
        seen.add(record["id"])
        for path, digest in record.get("content_digests", {}).items():
            if not Path(path).is_file() or digest_file(path) != digest:
                raise ValueError(f"Preparation artifact content changed: {path}")
        for version in record["dependencies"].values():
            verify(index[version])
    for record in records.values():
        verify(record)
    return records


def _record(context, kind):
    return next(iter(bound(context, kind).values()))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spec(context):
    bound(context, "SemanticPreparationSpec")
    return SemanticPreparationSpec.model_validate(_read(context["semantic_spec_path"]))


def _decisions(spec):
    return [spec.population, *spec.identities.values(), *spec.relationships, *spec.fields,
            spec.missing, spec.findings, spec.spatial]


def _population(context, reference):
    receipt = inspection(context, reference, source_id=context["source_scope"])
    if receipt["operation"] not in {"image_population", "json_population", "identity_import_population"}:
        raise ValueError("Independent population requires a trusted population inspection")
    value = receipt["value"]
    if digest_file(value["population_path"]) != value["population_digest"]:
        raise ValueError("Inspected population changed")
    return value


def save_source_fingerprint(context, *, inspection_references):
    """Freeze selected factual receipts, including their coverage; unrelated later reads do not invalidate it."""
    source = Source.model_validate(context["active_source"])
    source_artifact = _record(context, "Source")
    verify_source(context["source_manifest"])
    receipts, paths = [], []
    for ref in dict.fromkeys(inspection_references):
        receipt = inspection(context, ref, source_id=source.id)
        if receipt["kind"] not in {"inspection", "document"}:
            raise ValueError("Fingerprint requires actual source inspections")
        operation = receipt["operation"]
        coverage = "full" if operation in {"table", "join", "image_population", "json_population", "json_structure", "identity_import_population"} else "sample" if operation == "metadata" else "unknown"
        value = receipt["value"]
        denominator = next((value[k] for k in ("rows", "unique_images", "total") if isinstance(value, dict) and k in value), None)
        receipts.append({"reference": ref, "operation": operation, "arguments": receipt["arguments"],
                         "coverage": coverage, "denominator": denominator,
                         "facts": {k: v for k, v in value.items() if k not in {"text", "samples", "entries"}} if isinstance(value, dict) else {}})
        paths.append(str(Path(context["run_directory"]) / ".runtime/evidence" / (ref + ".json")))
    path = Path(context["operation_directory"]) / "fingerprint.json"
    used_path = path.with_name("used_files.jsonl")
    used_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    with sqlite3.connect(Path(context["source_manifest"]) / "used.sqlite") as db, used_path.open("w", encoding="utf-8") as handle:
        for relative, digest in db.execute("SELECT path,digest FROM used ORDER BY path"):
            handle.write(json.dumps({"path": relative, "digest": digest}, ensure_ascii=False) + "\n")
    fingerprint = SourceFingerprint(dataset_id=context["dataset_scope"], source=source,
                                    source_artifact=source_artifact["id"],
                                    files_path=str(Path(context["source_manifest"]) / "files.jsonl"),
                                    used_files_path=str(used_path),
                                    inspections=receipts, capabilities=CAPABILITIES)
    atomic_write_json(path, fingerprint.model_dump())
    versions = publish(context, "SourceFingerprint", {"path": str(path), "receipts": [{"path": p} for p in paths],
                                                      "files_path": fingerprint.files_path, "used_files_path": str(used_path)}, ("Source",))
    return result("success", "Source facts frozen; semantic decisions are separate", fingerprint_path=str(path), **versions)


def save_semantic_spec(context, *, spec):
    """Record source-supported meanings, identity, missing values and geometry without executable steps."""
    bound(context, "SourceFingerprint")
    spec = SemanticPreparationSpec.model_validate(spec)
    allowed = {item["reference"] for item in _read(context["fingerprint_path"])["inspections"]}
    evidence_paths = []
    for decision in _decisions(spec):
        if not set(decision.inspection_references) <= allowed:
            raise ValueError("Semantic facts must belong to the selected fingerprint; freeze required inspections first")
        for ref in decision.inspection_references:
            inspection(context, ref, source_id=context["source_scope"])
        if decision.interpretation == "interpreted" or decision.semantic_evidence:
            semantic_evidence(context, decision.semantic_evidence, context["source_scope"])
        evidence_paths.extend(str(Path(context["run_directory"]) / ".runtime/evidence" / (e.reference + ".json"))
                              for e in decision.semantic_evidence)
    if spec.population_reference not in allowed:
        raise ValueError("Population evidence must belong to the fingerprint")
    if context["active_source"]["kind"] == "raw":
        if spec.population.interpretation != "interpreted":
            raise ValueError("Raw population selection requires semantic evidence")
        if not {"image.image_id", "image.image_path"} <= {f.target for f in spec.fields}:
            raise ValueError("Declare image ID and path mappings explicitly")
    if spec.population_mode == "independent":
        value = _population(context, spec.population_reference)
        if spec.expected_images is not None and value["unique_images"] != spec.expected_images:
            raise ValueError("Source-declared expected_images differs from inspected population")
    else:
        receipt = inspection(context, spec.population_reference, source_id=context["source_scope"])
        if receipt["operation"] != "population_capability" or receipt["value"].get("supported") is not False:
            raise ValueError("Human review requires a trusted population capability gap")
        if spec.population.interpretation != "interpreted":
            raise ValueError("Special population selection requires semantic evidence")
    for field in spec.fields:
        if field.source_path:
            known_paths = set()
            for ref in field.inspection_references:
                observed = inspection(context, ref, source_id=context["source_scope"])
                arguments = observed["arguments"]
                known_paths.update(arguments.get(key) for key in ("path", "left", "right"))
                known_paths.update(step["path"] for step in arguments.get("steps", []) if step.get("op") == "read")
            if field.source_path not in known_paths:
                raise ValueError("Field source_path must be covered by its own factual inspections")
            track_files(context["source_manifest"], [field.source_path])
    path = Path(context["operation_directory"]) / "spec.json"
    atomic_write_json(path, spec.model_dump())
    versions = publish(context, "SemanticPreparationSpec", {"path": str(path),
                       "evidence": [{"path": p} for p in sorted(set(evidence_paths))]}, ("SourceFingerprint",))
    return result("success", "Semantic specification saved", semantic_spec_path=str(path), **versions)


def save_dataset_adapter(context, *, code):
    """Save Python as a first-class extractor. Emit wrapped annotation and extraction-provenance.v1.

    Read MEDBENCH_SOURCE_ROOT, MEDBENCH_SPEC_PATH and MEDBENCH_INPUTS_PATH. Write only candidates
    under MEDBENCH_WORK_DIR. Supply source-locator.v1 objects and finding target JSON Pointers.
    Human-review specs additionally require population.jsonl (source_id, image_id, image_path).
    Code never assigns trusted provenance, namespaces, approval, normalized images or ready markers.
    """
    bound(context, "SemanticPreparationSpec")
    try:
        compile(code, "adapter.py", "exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python at line {exc.lineno}: {exc.msg}") from exc
    path = Path(context["operation_directory"]) / "adapter.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    versions = publish(context, "DatasetAdapter", {"path": str(path)}, ("SemanticPreparationSpec",))
    return result("success", "Dataset adapter saved", annotation_script_path=str(path),
                  adapter_reference=next(iter(require(context, "DatasetAdapter").values()))["id"], **versions)


def save_execution_plan(context, *, plan):
    """Choose built_in or generated_python and bind the current specification and runtime."""
    spec = _spec(context)
    plan = ExecutionPlan.model_validate(plan)
    deps = ["SemanticPreparationSpec", "SourceFingerprint"]
    if plan.identity_import and context["active_source"]["kind"] != "annotation":
        raise ValueError("Identity import requires an annotation source")
    if context["active_source"]["kind"] == "annotation" and not plan.identity_import:
        raise ValueError("Use a raw source for semantic recoding of canonical imports")
    if plan.executor == "generated_python":
        adapter = _record(context, "DatasetAdapter")
        if adapter["id"] != plan.adapter_reference:
            raise ValueError("Execution plan references an obsolete adapter")
        deps.append("DatasetAdapter")
    elif plan.steps:
        if spec.population_mode == "human_review":
            raise ValueError("Built-in operators use independent population inspection; custom selection uses generated_python")
        track_files(context["source_manifest"], [s.path for s in plan.steps if s.op == "read"])
        targets = set(plan.steps[-1].fields) | set(plan.steps[-1].constants)
        declared = {f.target for f in spec.fields}
        if any(not any(d == target or d.startswith(target + "[].") or d.startswith(target + ".") for d in declared)
               for target in targets):
            raise ValueError("Every emitted field/constant needs a semantic field decision")
    path = Path(context["operation_directory"]) / "execution_plan.json"
    record = {"plan": plan.model_dump(), "spec_version": _record(context, "SemanticPreparationSpec")["id"],
              "fingerprint_version": _record(context, "SourceFingerprint")["id"], "runtime": context["execution_runtime"]}
    atomic_write_json(path, record)
    versions = publish(context, "ExecutionPlan", {"path": str(path)}, tuple(deps))
    return result("success", "Execution plan saved", execution_plan_path=str(path), **versions)


def inspect_json_population(context, *, population):
    """Independently enumerate a JSON object collection using direct identity fields and equality filters."""
    from medbenchagent.agents.data_processing import _run, _inventory
    spec = JsonPopulation.model_validate(population)
    track_files(context["source_manifest"], [spec.path])
    folder = Path(context["operation_directory"]) / "population"
    value = _run(context, folder, request={"operation": "json_population", "source_id": context["source_scope"],
                                         "population": spec.model_dump()})
    value.update(population_path=str(folder / "population.jsonl"), population_digest=digest_file(folder / "population.jsonl"))
    return _inventory(context, "json_population:" + spec.model_dump_json(), value,
                      operation="json_population", arguments=spec.model_dump())


def inspect_population_capability(context, *, inspection_reference, required_operation):
    """Record a fixed inspector capability gap for a semantically required operation, never an execution failure."""
    receipt = inspection(context, inspection_reference, source_id=context["source_scope"])
    if receipt["operation"] not in {"table", "join", "json_population", "json_structure"}:
        raise ValueError("Inspect source table/object structure before checking population capabilities")
    supported = required_operation not in {"aggregate_filter", "nested_array_join", "derived_identity"}
    value = {"supported": supported, "required_operation": required_operation,
             "inspection_reference": inspection_reference,
             "reason": "Population inspectors support relational selection and direct JSON fields, not this operation. "
                       "The semantic spec must justify why it is required; data errors are not capability gaps."}
    return inspected(context, "population_capability", {"required_operation": required_operation}, value,
                     ("Source",), source_id=context["source_scope"])


def inspect_json_structure(context, *, path, collection=""):
    """Inspect bounded JSON object examples and full collection count without interpreting their meaning."""
    from medbenchagent.data.locators import json_objects
    from medbenchagent.agents.data_processing import _inventory
    track_files(context["source_manifest"], [path])
    samples, count, types = [], 0, {}
    for locator, item in json_objects(context["source_root"], path, collection):
        count += 1
        if isinstance(item, dict):
            for name, value in item.items():
                types.setdefault(name, set()).add(type(value).__name__)
        if len(samples) < 3:
            samples.append({"locator": locator.model_dump(), "preview": json.dumps(item, ensure_ascii=False)[:1000]})
    return _inventory(context, "json:" + path + collection, {"rows": count, "types": {k: sorted(v) for k, v in types.items()},
                      "samples": samples, "coverage": {"rows": "full", "types": "full", "samples": "sample"}},
                      operation="json_structure", arguments={"path": path, "collection": collection})


def read_candidate_population(context, *, offset=0, limit=20, query=""):
    """Page the full special-population review list; no sample replaces the complete file."""
    candidate = _candidate(context)
    spec = _spec(context)
    path = candidate / "population.jsonl" if spec.population_mode == "human_review" else Path(_population(context, spec.population_reference)["population_path"])
    rows, count = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if query and query not in item["image_id"]:
                continue
            index = count
            count += 1
            if offset <= index < offset + limit:
                rows.append(item)
    return result("success", json.dumps({"path": str(path), "total": count, "entries": rows,
                  "next_offset": offset + limit if offset + limit < count else None}, ensure_ascii=False))


def execute_extraction(context):
    """Run the selected extractor twice; publish candidates only, never an accepted Annotation."""
    from medbenchagent.agents.data_processing import _run
    from medbenchagent.data.execution import output_digests, require_script_runtime
    bound(context, "ExecutionPlan")
    spec = _spec(context)
    execution = _read(context["execution_plan_path"])
    plan = ExecutionPlan.model_validate(execution["plan"])
    if execution["runtime"] != context["execution_runtime"]:
        raise ValueError("Execution runtime changed")
    verify_source(context["source_manifest"])
    if any(step.op == "mask_to_bbox" for step in plan.steps):
        with (Path(context["source_manifest"]) / "files.jsonl").open(encoding="utf-8") as handle:
            track_files(context["source_manifest"], (json.loads(line)["path"] for line in handle))
    first, repeat = (Path(context["operation_directory"]) / name for name in ("candidate", "repeat"))
    if plan.executor == "generated_python":
        require_script_runtime(context["execution_runtime"])
        # Python can read any source file. Preserve the existing conservative dependency policy.
        with (Path(context["source_manifest"]) / "files.jsonl").open(encoding="utf-8") as handle:
            track_files(context["source_manifest"], (json.loads(line)["path"] for line in handle))
        inputs = Path(context["operation_directory"]) / "inputs.json"
        from medbenchagent.data.contracts import ExtractionProvenance
        atomic_write_json(inputs, {"source": context["active_source"], "capabilities": CAPABILITIES,
                                  "provenance_schema": ExtractionProvenance.model_json_schema()})
        arguments = {"script": context["annotation_script_path"], "specification": context["semantic_spec_path"], "inputs": str(inputs)}
    elif plan.identity_import:
        arguments = {"request": {"operation": "identity_import", "source": context["active_source"]}}
    else:
        arguments = {"request": {"operation": "execute_plan", "source": context["active_source"],
                                 "plan": plan.model_dump()}}
    for target in (first, repeat):
        _run(context, target, **arguments)
        if not (target / "provenance.jsonl").is_file():
            raise ValueError("Extractor must emit provenance.jsonl")
        if spec.population_mode == "human_review" and not (target / "population.jsonl").is_file():
            raise ValueError("Special populations require a complete population.jsonl")
    compare_outputs(first, repeat)
    verify_source(context["source_manifest"])
    remove_work_directory(repeat, context["operation_directory"])
    receipt = Path(context["operation_directory"]) / "candidate.json"
    atomic_write_json(receipt, {"outputs": output_digests(first), "directory": str(first)})
    versions = publish(context, "CandidateExtraction", {"path": str(receipt),
                       "outputs": [{"path": str(first / name)} for name in output_digests(first)]}, ("ExecutionPlan",))
    return result("success", "Repeatable candidates ready for independent validation", candidate_path=str(receipt), **versions)


def _candidate(context):
    from medbenchagent.data.execution import verify_output_digests
    bound(context, "CandidateExtraction")
    receipt = _read(context["candidate_path"])
    directory = Path(receipt["directory"])
    verify_output_digests(directory, receipt["outputs"])
    return directory


def validate_preparation(context):
    """Resolve provenance and reconcile the pre-quarantine population before admitting image processing."""
    from medbenchagent.agents.data_processing import _run
    from medbenchagent.data.population import verify_population
    spec = _spec(context)
    verify_source(context["source_manifest"])
    candidate = _candidate(context)
    validation = _run(context, Path(context["operation_directory"]) / "validation",
                      request={"operation": "validate_extraction", "spec": spec.model_dump()},
                      annotation=str(candidate / "annotation.jsonl"), provenance=str(candidate / "provenance.jsonl"))
    count = validation["count"]
    if spec.expected_images is not None and count != spec.expected_images:
        raise ValueError("Candidate count differs from source-declared expected_images")
    answer_reference = None
    if spec.population_mode == "independent":
        value = _population(context, spec.population_reference)
        population_path = value["population_path"]
        assurance = "independently_reconciled" if not _read(context["execution_plan_path"])["plan"]["identity_import"] else "input_annotation_only"
    else:
        population_path = str(candidate / "population.jsonl")
        verify_population(context["source_root"], population_path, candidate / "annotation.jsonl", context["source_scope"])
        from medbenchagent.runtime.clarifications import new_issue
        question = (f"Review population selection for source {context['source_scope']}: {count} images. "
                    f"Rule: {spec.population.meaning}. Complete list: {population_path}. "
                    f"Source facts: {context['fingerprint_path']}; specification: {context['semantic_spec_path']}. "
                    "Approve or reject this specific population selection. Its completeness has not been independently proven.")
        issue = new_issue("DataProcessingAgent", context, "population_selection", question)
        issue = context.get("issues", {}).get(issue["issue_id"], issue)
        answer = issue.get("population_answer", {})
        if issue["status"] == "resolved" and answer.get("decision") == "reject":
            return result("failed", "Population selection rejected; revise the semantic spec and extraction: " + answer["reason"])
        if issue["status"] != "resolved" or answer.get("decision") != "approve":
            return result("needs_input", question if not answer else answer["reason"],
                          issues={**context.get("issues", {}), issue["issue_id"]: issue},
                          global_issue=issue, unresolved_questions=[question],
                          population_review_issue=issue["issue_id"],
                          population_review_path=population_path)
        # Validate the version-bound answer again, not just its cached status.
        from medbenchagent.runtime.inspection import evidence
        answer_reference = issue["answer_reference"]
        evidence(context, answer_reference)
        assurance = "human_reviewed_selection_completeness_unproven"
    verify_population(context["source_root"], population_path, candidate / "annotation.jsonl", context["source_scope"])
    verify_source(context["source_manifest"])
    folder = Path(context["operation_directory"])
    decision_path, validation_path = folder / "population_decision.json", folder / "validation.json"
    atomic_write_json(decision_path, {"schema_version": "population-decision.v1", "count": count,
                      "assurance": assurance, "population_digest": digest_file(population_path), "answer_reference": answer_reference})
    versions = publish(context, "PopulationDecision", {"path": str(decision_path), "population_path": population_path,
                       "answer_path": str(Path(context["run_directory"]) / ".runtime/evidence" / (answer_reference + ".json")) if answer_reference else None},
                       ("SemanticPreparationSpec", "CandidateExtraction"))
    atomic_write_json(validation_path, {"schema_version": "preparation-validation.v1", "count": count,
                      "assurance": assurance, "checks": ["source_version", "repeatability", "locator_existence", "direct_field_bindings", "finding_references", "population_identity"],
                      "semantic_limit": "Custom transformations and semantic interpretations rely on source-grounded decisions; not independently proven.",
                      "deferred": ["canonical_record_quality", "image_quality", "entity_invariants"]})
    versions = publish(context, "PreparationValidation", {"path": str(validation_path)}, ("PopulationDecision", "CandidateExtraction"))
    versions = publish(context, "Annotation", {"path": str(candidate / "annotation.jsonl")}, ("PreparationValidation",))
    return result("success", "Extraction accepted; image QA and normalization remain required",
                  prepared_annotation_path=str(candidate / "annotation.jsonl"), wrapped_annotation=True,
                  population_decision_path=str(decision_path), preparation_validation_path=str(validation_path),
                  accepted_population_path=population_path, population_assurance=assurance,
                  population_review_issue=None, global_issue=None, unresolved_questions=[], **versions)


def prepare_annotation_import(context):
    """Create an identity specification for existing canonical annotation; do not reinterpret labels."""
    from medbenchagent.agents.data_processing import _run, _inventory
    if context["active_source"]["kind"] != "annotation":
        raise ValueError("Identity import requires kind=annotation")
    bound(context, "Source")
    paths = [context["active_source"]["annotation"]]
    if (Path(context["source_root"]) / "image_manifest.jsonl").is_file():
        paths.append("image_manifest.jsonl")
    track_files(context["source_manifest"], paths)
    folder = Path(context["operation_directory"]) / "inspection"
    value = _run(context, folder, request={"operation": "identity_import", "source": context["active_source"]})
    value.update(population_path=str(folder / "population.jsonl"), population_digest=digest_file(folder / "population.jsonl"))
    outcome = _inventory(context, "identity_import", value, operation="identity_import_population", arguments={"path": paths[0]})
    context.update(outcome["updates"])
    ref = json.loads(outcome["summary"])["inspection_reference"]
    context.update(save_source_fingerprint(context, inspection_references=[ref])["updates"])
    def decision(meaning):
        return {"meaning": meaning, "inspection_references": [ref], "interpretation": "preserve"}
    spec = {"population": decision("All records in the supplied canonical annotation; earlier raw completeness is unknown"),
            "population_reference": ref, "expected_images": value["unique_images"],
            "identities": {e: decision("Preserve supplied ID, null remains null") for e in ("patient", "case", "study", "image")},
            "missing": decision("Preserve all supplied values without inferring negatives"),
            "findings": decision("Preserve array objects and their pairing"),
            "spatial": decision("Canonical absolute pixel xyxy; checked against image dimensions during normalization")}
    context.update(save_semantic_spec(context, spec=spec)["updates"])
    context.update(save_execution_plan(context, plan={"executor": "built_in", "identity_import": True})["updates"])
    return result("success", "Canonical identity import planned; execute and validate it before image processing",
                  **{k: context[k] for k in ("fingerprint_path", "semantic_spec_path", "execution_plan_path", "artifact_versions", "evidence_index_path", "inventory_path")})
