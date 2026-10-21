import pytest
from pydantic import ValidationError
from medbenchagent.workflow.graph import BindingArguments
from medbenchagent.planning.dataset_selection import bind_dataset
from medbenchagent.planning.bir import BIRStore
from medbenchagent.data.dataset_catalog import register_dataset
from medbenchagent.preparation.graph import compile_preparation_graph
from test_preparation_graph import Worker, state


def test_binding_has_one_dataset_and_rejects_old_plural_arguments():
    assert BindingArguments(dataset_id="d", reason="Requested").dataset_id == "d"
    with pytest.raises(ValidationError):
        BindingArguments(dataset_ids=["d"], selection_reason={"d": "Requested"})


@pytest.fixture
def binding_context(tmp_path, monkeypatch):
    initial = state(tmp_path)
    publication = compile_preparation_graph(worker=Worker()).invoke(initial)["publication"]
    catalog = tmp_path / "catalog.json"
    register_dataset("d", publication["bundle"], catalog_path=catalog)
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    store.update("benchmark_info.json", {"user_requirement": "Compile d", "dataset_mentions": ["d"]}, run_id="r", actor="test")
    context = {"bir_path": str(store.root), "user_requirement": "Compile d", "run_id": "r",
               "operation_directory": str(tmp_path / "binding"), "registration_catalog_path": str(catalog)}
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.prepared_directory_entries", lambda _context: ([], {}))
    return context


def test_binding_freezes_one_verified_access_and_cannot_switch(binding_context):
    context = binding_context
    store = BIRStore(context["bir_path"])
    bind_dataset(context, dataset_id="d", reason="Requested")
    binding = store.load("dataset_binding.json")
    assert binding["access"]["dataset_id"] == "d"
    assert "source_access" not in binding["access"] and "datasets" not in binding
    with pytest.raises(ValueError, match="cannot switch"):
        bind_dataset(context, dataset_id="other", reason="Other")


def test_brief_revision_keeps_binding_and_invalidates_plan(binding_context):
    context = binding_context
    store = BIRStore(context["bir_path"])
    bind_dataset(context, dataset_id="d", reason="Requested")
    from medbenchagent.planning.bir import business_dependencies
    store.update("task_plan.json", {"phase": "complete"}, run_id="r", actor="test",
                 dependencies=business_dependencies(context, "benchmark_info.json", "dataset_binding.json"))
    binding_digest = store.digest("dataset_binding.json")
    store.update("benchmark_info.json", {"user_requirement": "Compile d in Chinese"}, run_id="r", actor="test")
    assert store.current("dataset_binding.json")["access"]["dataset_id"] == "d"
    assert store.digest("dataset_binding.json") == binding_digest
    assert store.current("task_plan.json") is None
    with pytest.raises(ValueError, match="cannot switch"):
        bind_dataset(context, dataset_id="other", reason="Other")


@pytest.mark.parametrize("issue", [None, {"issue_id": "selection", "purpose": "dataset_selection", "status": "open"},
    {"issue_id": "selection", "purpose": "dataset_selection", "status": "answered",
     "input_versions": {"benchmark_info.json": "outdated"}, "answer_reference": "answer"}])
def test_invalid_selection_question_does_not_bind(binding_context, issue):
    context = {**binding_context, "global_issue": issue}
    store = BIRStore(context["bir_path"])
    with pytest.raises(ValueError, match="selection question|inputs changed"):
        bind_dataset(context, dataset_id="d", reason="Requested", selection_issue_id="selection")
    assert store.current("dataset_binding.json") is None


def test_catalog_rejects_dataset_identity_alias(binding_context):
    from pathlib import Path
    from medbenchagent.data.dataset_catalog import load_dataset_catalog, load_prepared
    catalog = Path(binding_context["registration_catalog_path"])
    entry = load_dataset_catalog(catalog)[0]
    before = catalog.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        register_dataset("other", Path(entry["cases_path"]).parent, catalog_path=catalog)
    assert catalog.read_bytes() == before
    with pytest.raises(ValueError, match="identity"):
        load_prepared({**entry, "dataset_id": "other"})
