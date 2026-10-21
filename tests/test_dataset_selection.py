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


def test_binding_freezes_one_verified_access_and_cannot_switch(tmp_path, monkeypatch):
    initial = state(tmp_path)
    publication = compile_preparation_graph(worker=Worker()).invoke(initial)["publication"]
    catalog = tmp_path / "catalog.json"
    register_dataset("d", publication["bundle"], catalog_path=catalog)
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    store.update("benchmark_info.json", {"user_requirement": "Compile d", "dataset_mentions": ["d"]}, run_id="r", actor="test")
    context = {"bir_path": str(store.root), "user_requirement": "Compile d", "run_id": "r",
               "operation_directory": str(tmp_path / "binding"), "registration_catalog_path": str(catalog)}
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.prepared_directory_entries", lambda _context: ([], {}))
    bind_dataset(context, dataset_id="d", reason="Requested")
    binding = store.load("dataset_binding.json")
    assert binding["access"]["dataset_id"] == "d"
    assert "source_access" not in binding["access"] and "datasets" not in binding
    with pytest.raises(ValueError, match="cannot switch"):
        bind_dataset(context, dataset_id="other", reason="Other")
