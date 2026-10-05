"""Single OpenHands worker protocol adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medbenchagent.data.sources import digest_file

from .contracts import Diagnostic, StageRequest, StageResult, Submission


class OpenHandsWorker:
    """Invoke the run-frozen OpenHands deployment for one stage attempt.

    The class contains no stage-specific model behaviour.  Stage differences
    are expressed by the request's prompt, context and output contract.
    """

    def __init__(self, runtime: Any | None = None):
        self.runtime = runtime

    @staticmethod
    def _submission(request: StageRequest) -> tuple[Submission | None, str]:
        path = Path(request.workspace) / "outputs" / "submission.json"
        if not path.is_file():
            return None, "Worker did not write outputs/submission.json"
        try:
            submission = Submission.model_validate_json(path.read_text(encoding="utf-8"))
            output_root = Path(request.workspace) / "outputs"
            missing = [name for name in submission.outputs if not (output_root / name).is_file()]
            if missing:
                return None, f"Submission references missing outputs: {missing[:10]}"
            if submission.requested_outcome == "submitted":
                required = [name for name in request.required_outputs if name != "submission.json"]
                absent = [name for name in required if not (output_root / name).is_file()]
                if absent:
                    return None, f"Submitted outcome is missing required outputs: {absent}"
            return submission, ""
        except Exception as exc:
            return None, f"Invalid outputs/submission.json: {exc}"

    @staticmethod
    def _request_task(request: StageRequest):
        from medbenchagent.runtime.dataset_runtime import DatasetAgentTask

        if not request.source_roots:
            raise ValueError("Stage request requires at least one source root")
        primary_id = sorted(request.source_roots)[0]
        manifest = request.source_manifests.get(primary_id, "")
        if not manifest:
            raise ValueError(f"Stage request is missing the manifest for source {primary_id}")
        return DatasetAgentTask(
            task_id=f"{request.run_id}:{request.dataset_id}:{request.stage.value}",
            kind="preparation",
            source_root=request.source_roots[primary_id],
            source_manifest=manifest,
            source_roots=request.source_roots,
            workspace=request.workspace,
            objective=request.stage.value,
            required_outputs=request.required_outputs,
            feedback="",
            attempt=request.attempt,
            max_iterations=request.budget.max_iterations,
            timeout_seconds=request.budget.timeout_seconds,
            conversation_id=request.conversation_id,
            model=request.model,
            base_url=request.base_url,
            reasoning_effort=request.reasoning_effort,
            construction_events_path=request.construction_events_path,
            run_id=request.run_id,
            dataset_id=request.dataset_id,
            source_id=primary_id,
            workflow_step=request.stage.value,
            source_digest=request.input_digest,
            stage=request.stage.value,
            context_path=request.context_path,
            prompt_path=request.prompt_path,
            allowed_outputs=request.required_outputs + request.optional_outputs,
            input_digest=request.input_digest,
            prompt_token_count=request.prompt_token_count,
            execution_key=request.execution_key,
            resume_existing=request.resume_existing,
            progress_guard={"no_progress_limit": request.budget.no_progress_limit, "tool_error_limit": request.budget.tool_error_limit},
        )

    def run(self, request: StageRequest) -> StageResult:
        from medbenchagent.runtime.dataset_runtime import dataset_runtime

        task = self._request_task(request)
        runtime = self.runtime or dataset_runtime({"execution_runtime": None})
        try:
            result = runtime.run(task)
        except Exception as exc:
            diagnostic = Diagnostic(stage=request.stage, code="worker_execution_failed", category="environment", blocking=True, message=str(exc))
            stage_result = StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
                                       execution_key=request.execution_key, status="failed", diagnostics=[diagnostic],
                                       diagnostic=str(exc), termination_reason="worker_error", input_digest=request.input_digest)
            self._write_result(request, stage_result)
            return stage_result

        submission, submission_error = self._submission(request)
        status = {"complete": "submitted", "submitted": "submitted", "needs_input": "needs_input", "blocked": "blocked", "failed": "failed"}.get(result.status, "failed")
        diagnostics = []
        if result.diagnostic:
            diagnostics.append(Diagnostic(stage=request.stage, code=result.error_category or "worker", category="environment" if result.error_category == "environment_unavailable" else "unknown", blocking=status != "submitted", message=result.diagnostic))
        if submission_error:
            diagnostics.append(Diagnostic(stage=request.stage, code="submission_invalid", category="contract", blocking=True, message=submission_error))
            if status == "submitted":
                status = "failed"
        elif submission and submission.requested_outcome != "submitted" and status == "submitted":
            status = submission.requested_outcome
        if submission:
            for code in submission.diagnostic_codes:
                diagnostics.append(Diagnostic(stage=request.stage, code=code, category="mapping" if code == "mapping_revision_required" else "unknown",
                                              blocking=True, message=f"Worker reported diagnostic code: {code}"))
        outputs = {name: str(Path(path).resolve()) for name, path in result.outputs.items()}
        prompt_digest = digest_file(request.prompt_path) if Path(request.prompt_path).is_file() else ""
        context_digest = digest_file(request.context_path) if Path(request.context_path).is_file() else ""
        stage_result = StageResult(
            stage=request.stage,
            attempt=request.attempt,
            conversation_id=request.conversation_id,
            execution_key=request.execution_key,
            status=status,
            outputs=outputs,
            question_refs=submission.question_refs if submission else [],
            diagnostics=diagnostics,
            diagnostic=(submission.summary if submission else result.diagnostic),
            termination_reason=result.termination_reason or ("completed" if status == "submitted" else "worker_error"),
            progress=result.progress,
            submission_path=str((Path(request.workspace) / "outputs" / "submission.json").resolve()) if submission else None,
            input_digest=request.input_digest,
            prompt_digest=prompt_digest,
            context_digest=context_digest,
            token_count=request.prompt_token_count,
        )
        self._write_result(request, stage_result)
        return stage_result

    @staticmethod
    def _write_result(request: StageRequest, result: StageResult) -> None:
        path = Path(request.workspace) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
