"""Fixed preparation stage dispatcher."""
from __future__ import annotations

from pathlib import Path

from .preparation_gates import read_proposal
from .preparation_prompts import render_stage_prompt

STAGES = ("inventory", "profile", "map", "plan", "implement", "sample_validate",
          "full_execute", "final_validate", "publish")
AGENT_STAGES = frozenset(STAGES[:6])


class PreparationController:
    def __init__(self, *, runtime=None):
        self.runtime = runtime

    def step(self, session: dict, *, stop_requested=None) -> dict:
        stage = session["stage"]
        if stage not in STAGES:
            raise ValueError(f"Unknown preparation stage: {stage}")
        if stop_requested and stop_requested():
            return {"status": "cancelled", "stage": stage}
        workspace = Path(session["workspace"])
        workspace.mkdir(parents=True, exist_ok=True)
        turn_id = f"turn-{session.get('turn_counter', 0) + 1:04d}"
        session["turn_counter"] = session.get("turn_counter", 0) + 1
        if stage in AGENT_STAGES:
            if self.runtime is None:
                return {"status": "blocked", "stage": stage, "diagnostic": "Preparation runtime is unavailable"}
            proposal = workspace / "state" / "proposals" / f"{turn_id}.json"
            proposal.parent.mkdir(parents=True, exist_ok=True)
            task = {
                "task_id": f"{session['dataset_id']}:{session['revision']}:{turn_id}",
                "turn_id": turn_id, "conversation_id": session["conversation_id"],
                "dataset_id": session["dataset_id"], "revision": session["revision"],
                "stage": stage, "workspace": str(workspace), "proposal_path": str(proposal),
                "prompt": render_stage_prompt(dataset_id=session["dataset_id"], revision=session["revision"],
                    conversation_id=session["conversation_id"], stage=stage, sources_ref=session["sources"],
                    goal=session.get("goal", ""), artifact_index=session.get("artifacts", {}),
                    decisions_ref=session.get("decisions_ref", ""), feedback_ref=session.get("feedback_ref", ""),
                    required_artifacts=session.get("required_artifacts", []),
                    acceptance_criteria=session.get("acceptance_criteria", ""), proposal_path=str(proposal)),
            }
            result = self.runtime.run(task)
            if isinstance(result, dict):
                result = dict(result)
            else:
                result = result.model_dump()
            if result.get("status") == "submitted":
                proposal_data = read_proposal(proposal, turn_id=turn_id, stage=stage)
                result.update(proposal=proposal_data)
            return {**result, "stage": stage, "turn_id": turn_id}
        return {"status": "blocked", "stage": stage, "diagnostic": f"Host handler for {stage} is not implemented"}

    @staticmethod
    def advance(session: dict, result: dict) -> dict:
        status = result.get("status")
        if status != "submitted":
            return {**session, "status": status or "failed", "last_result": result}
        index = STAGES.index(session["stage"])
        if index + 1 == len(STAGES):
            return {**session, "status": "complete", "last_result": result}
        return {**session, "stage": STAGES[index + 1], "status": "running", "last_result": result}
