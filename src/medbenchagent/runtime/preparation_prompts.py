"""Single prompt renderer for preparation coding turns."""
from __future__ import annotations


def render_stage_prompt(*, dataset_id, revision, conversation_id, stage, sources_ref,
                        goal, artifact_index, decisions_ref, feedback_ref, required_artifacts,
                        acceptance_criteria, proposal_path):
    return (f"You are the MedBench data preparation coding agent. dataset={dataset_id} "
            f"revision={revision} conversation={conversation_id}. Work only on stage={stage}.\n"
            f"Sources: {sources_ref}\nGoal: {goal}\nVerified inputs: {artifact_index}\n"
            f"Decisions: {decisions_ref}\nFeedback: {feedback_ref}\n"
            f"Required artifacts: {required_artifacts}\nAcceptance: {acceptance_criteria}\n"
            f"Submit a JSON proposal to {proposal_path}; do not claim publication.")
