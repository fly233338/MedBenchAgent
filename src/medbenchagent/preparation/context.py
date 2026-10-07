"""One bounded context per attempt; copy only accepted semantic inputs."""
from pathlib import Path
import json
import shutil

from .contracts import Stage, StageBudget, AcceptedHandoff, STAGE_OUTPUTS, digest_file, digest_payload


class ContextBudgetError(ValueError):
    pass


def resolve_prompt_budget(state, stage):
    return StageBudget(**state.get("preparation_stage_budgets", {}).get(Stage(stage).value, {}))


def stage_input_digest(stage, state):
    return digest_payload({key: state.get(key) for key in (
        "source_snapshot_digest", "bundle_root", "source_roots", "accepted_handoff", "text_artifacts", "user_decisions",
        "validation_feedback", "previous_attempt_root", "revision_basis_handoff", "execution_runtime", "model", "base_url",
    )} | {"stage": str(stage)})


def verify_previous_reference(workspace):
    path = Path(workspace) / "request.json"
    if not path.exists():
        return
    context = json.loads(path.read_text(encoding="utf-8")).get("context", {})
    for source, digest in context.get("reference_digests", {}).items():
        if not Path(source).is_file() or digest_file(source) != digest:
            raise ValueError("Previous preparation reference changed")


def build_stage_context(stage, state, workspace):
    stage, workspace = Stage(stage), Path(workspace)
    inputs = workspace / "work/inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    docker = state.get("execution_runtime", {}).get("backend") == "docker"
    context = {
        "stage": stage.value, "dataset_id": state["dataset_id"],
        "bundle_root": state["bundle_root"], "locator_roots": state["source_roots"],
        "source_roots": {key: f"/workspace/sources/{key}" if docker else value for key, value in state["source_roots"].items()},
        "source_manifests": {key: f"/workspace/source_manifests/{key}" if docker else value
                             for key, value in state["source_manifests"].items()},
        "user_request": state.get("user_request", ""), "decisions": state.get("user_decisions", []),
        "feedback": state.get("validation_feedback", {}), "inputs": {}, "reference_digests": {},
        "outputs": list(STAGE_OUTPUTS[stage]),
    }
    handoff = state.get("accepted_handoff") or state.get("revision_basis_handoff")
    if handoff:
        handoff = AcceptedHandoff.model_validate(handoff, strict=False)
        for artifact in (handoff.mapping, handoff.image_conversion_plan):
            if digest_file(artifact.path) != artifact.digest:
                raise ValueError(f"Accepted input changed: {artifact.name}")
            shutil.copy2(artifact.path, inputs / artifact.name)
            context["inputs"][artifact.name] = f"work/inputs/{artifact.name}"
            context["reference_digests"][artifact.path] = artifact.digest
    if stage is Stage.PREPARE_IMAGE:
        for name, artifact in state["text_artifacts"].items():
            if digest_file(artifact["path"]) != artifact["digest"]:
                raise ValueError("Accepted Text output changed")
            shutil.copy2(artifact["path"], inputs / name)
            context["inputs"][name] = f"work/inputs/{name}"
            context["reference_digests"][artifact["path"]] = artifact["digest"]
    if previous := state.get("previous_attempt_root"):
        root = Path(previous).resolve()
        if not root.is_relative_to(Path(state["preparation_root"]).resolve()):
            raise ValueError("Previous attempt is outside preparation runtime")
        context["previous"] = "/workspace/previous" if docker else str(root)
        context["reference_digests"].update({str(p): digest_file(p) for p in (root / "work").rglob("*") if p.is_file()})
    prompts = Path(__file__).parents[1] / "prompts"
    visible = {key: value for key, value in context.items() if key != "reference_digests"}
    prompt = "\n\n".join((
        (prompts / "data_preparation_agent.md").read_text(encoding="utf-8"),
        (prompts / f"data_preparation_{stage.value}.md").read_text(encoding="utf-8"),
        json.dumps(visible, ensure_ascii=False, indent=2),
        "Inspect Python model schemas via medbenchagent.preparation.cli schema. "
        "Inspect enough to understand the source and generate the required candidate outputs programmatically. "
        "Perform only necessary construction checks. Once all required output files are written, immediately finish. "
        "Host checks required files and validates their contents after the worker ends. "
        "No success receipt or submit command is required. Do not run preparation.cli validate. "
        "Do not perform additional inspection, validation, spot-checks, or tool calls after outputs are ready. "
        "For semantic clarification, write a JSON array of scoped question records (schema question), "
        "then run python -m medbenchagent.preparation.cli signal --outcome needs_input --questions-file work/questions.json. "
        "For blocking failures use signal --outcome blocked --summary 'diagnostic'; "
        "add --diagnostic-code mapping_revision_required only for demonstrated semantic conflicts. "
        "Only exceptional control uses work/control.json; finish after reporting it.",
    ))
    budget = resolve_prompt_budget(state, stage)
    tokens = (len(prompt.encode()) + 3) // 4
    limit = budget.initial_message_tokens
    if budget.model_context_tokens:
        limit = min(limit, budget.model_context_tokens - budget.output_reserve_tokens - budget.conversation_reserve_tokens - budget.safety_margin_tokens)
    if tokens > limit:
        raise ContextBudgetError("required_context_too_large")
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
    return context, prompt, tokens, digest_payload([stage_input_digest(stage, state), prompt, state.get("model")])
