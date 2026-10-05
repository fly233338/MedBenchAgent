from medbenchagent.runtime.preparation_state import (
    MemoryRecord,
    PreparationPhase,
    PreparationState,
    StageBudget,
    StageContext,
)


def test_stage_sessions_are_fresh_and_deterministic():
    state = PreparationState(run_id="run", dataset_id="dataset")
    first = state.next_session(PreparationPhase.INVENTORY)
    retry = first.next_session(PreparationPhase.INVENTORY, 2)
    assert first.session_id != retry.session_id
    assert retry.session_id == first.next_session(PreparationPhase.INVENTORY, 2).session_id
    assert retry.progress.iterations == 0


def test_memory_records_are_digest_bound_and_context_is_bounded():
    record = MemoryRecord(id="fact-1", value={"count": 390}, producer="host", scope="dataset")
    context = StageContext(
        run_id="run", dataset_id="dataset", phase=PreparationPhase.SEMANTIC,
        attempt=1, objective="Map fields", memory={"verified_facts": [record]},
        budget=StageBudget(max_iterations=16, timeout_seconds=900), workspace="worker",
    )
    assert record.digest
    assert context.phase == PreparationPhase.SEMANTIC
    assert context.budget.no_progress_limit == 8
