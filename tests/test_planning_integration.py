"""End-to-end construction using real data paths and deterministic model doubles."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from langgraph.types import Command
from PIL import Image
from langgraph.checkpoint.memory import MemorySaver

from medbenchagent import settings
from medbenchagent.workflow.graph import compile_benchmark_graph
from test_generation import _record, _task


@pytest.mark.parametrize(
    "ask_global,revise_after_plan", [(False, False), (True, False), (False, True)]
)
def test_complete_build_with_explicit_planning_shortfall(
    tmp_path, monkeypatch, ask_global, revise_after_plan
):
    source = tmp_path / "source"
    source.mkdir()
    rows, manifest = [], []
    for index, label in enumerate(["benign", "malignant"]):
        row = _record(index, {"status": label})
        row["image"]["image_path"] = f"{index}.png"
        rows.append(row)
        image = Image.new("L", (4, 4))
        image.putpixel((0, 0), 255)
        image.save(source / f"{index}.png")
        manifest.append(
            {"image_id": f"img-{index}", "png_path": f"{index}.png", "width": 4, "height": 4}
        )
    annotation = source / "annotation.jsonl"
    annotation.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (source / "image_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest), encoding="utf-8"
    )
    blueprint_template = tmp_path / "core.json"
    blueprint_template.write_text(
        json.dumps(
            {
                "concepts": {
                    "status": {
                        "origin": "core",
                        "category": "clinical",
                        "description": "Status",
                        "require": "Explicit annotation",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    configured = settings.get_settings()
    configured = replace(
        configured,
        paths=replace(
            configured.paths,
            blueprint_template=blueprint_template,
            dataset_catalog=tmp_path / "catalog.json",
            materials_root=tmp_path / "materials",
            runs_root=tmp_path / "runs",
        ),
    )
    monkeypatch.setattr(settings, "_SETTINGS", configured)
    info = {
        "disease": "Breast disease",
        "modality": "Mammography",
        "language": "en",
        "description": "Classify status",
        "sample_count": 4,
    }
    updated_info = {**info, "sample_count": 1, "language": "zh"}
    briefs = iter([info, updated_info])
    monkeypatch.setattr(
        "medbenchagent.planning.benchmark_info.complete_json",
        lambda *a: next(briefs) if revise_after_plan else info,
    )
    supervisor = iter(
        (["specify", "plan", "ask_user"] if ask_global or revise_after_plan else [])
        + ["specify", "plan", "generate", "audit", "finish"]
    )
    leaf = _task("letter", ["image.annotation.status"])["protocol"]
    planning_actions = (
        [("ask_user", {"question": "Confirm goal", "scope": "global"})] if ask_global else []
    ) + [
        ("select_dataset", {"dataset_id": "test", "annotation_path": str(annotation)}),
        ("profile_dataset", {}),
        ("select_materials", {"files": []}),
        ("read_node", {"node_id": "status"}),
        ("read_field_values", {"field": "image.annotation.status"}),
        (
            "save_leaf",
            {
                "leaf_id": "arbitrary.identity",
                "concept_id": "status",
                "description": "Status",
                "protocol": leaf,
            },
        ),
        (
            "complete_concept",
            {"concept_id": "status", "status": "complete", "reason": "Direct data"},
        ),
        ("coverage_gap", {}),
        ("finish_discovery", {}),
        ("review_tree", {}),
        (
            "allocate",
            {"weights": {"arbitrary.identity": {"weight": 1.0, "reason": "Requested ability"}}},
        ),
        ("finish", {}),
    ]
    if revise_after_plan:
        planning_actions += [
            ("select_dataset", {"dataset_id": "updated-test", "annotation_path": str(annotation)}),
            ("profile_dataset", {}),
            (
                "complete_concept",
                {
                    "concept_id": "status",
                    "status": "complete",
                    "reason": "Revalidated current data",
                },
            ),
            ("coverage_gap", {}),
            ("finish_discovery", {}),
            ("review_tree", {}),
            (
                "allocate",
                {"weights": {"arbitrary.identity": {"weight": 1.0, "reason": "Updated request"}}},
            ),
            ("finish", {}),
        ]
    actions = iter(planning_actions)
    delegated = []

    def action_model(instructions, prompt):
        if instructions.startswith("Build a two-level"):
            name, args = next(actions)
        else:
            name, args = next(supervisor), {}
            if name == "ask_user":
                args = {"question": "Confirm goal"}
            delegated.append(name)
        return {"tool": name, "arguments": args, "reason": "Required next action"}

    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", action_model)
    monkeypatch.setattr(
        "medbenchagent.agents.task_planning.complete_json", lambda *a: {"issues": []}
    )

    def template_model(system, prompt, **kwargs):
        assert json.loads(prompt)["specification"]["language"] == (
            "zh" if revise_after_plan else "en"
        )
        if system.startswith("Review"):
            return {"approved": True}
        if revise_after_plan:
            return {
                "templates": ["图像显示哪种状态？", "请选择图中的状态。", "判断图像对应的状态。"]
            }
        return {
            "templates": ["What status is shown?", "Which status applies?", "Identify the status."]
        }

    monkeypatch.setattr("medbenchagent.generation.templates.complete_json", template_model)
    monkeypatch.setattr(
        "medbenchagent.audit.semantic_audit.complete_json",
        lambda system, prompt: {
            "items": [
                {"item_id": i["item_id"], "decision": "pass", "violations": []}
                for i in json.loads(prompt)["items"]
            ]
        },
    )
    graph = compile_benchmark_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "integration"}, "recursion_limit": 250}
    output = graph.invoke(
        {"user_requirement": "Build four English mammography status questions"}, config=config
    )
    if ask_global or revise_after_plan:
        assert output["__interrupt__"]
        output = graph.invoke(
            Command(
                resume="Use one Chinese question and the updated Dataset"
                if revise_after_plan
                else "Use the stated status classification goal"
            ),
            config=config,
        )
    assert output["status"] == "complete"
    assert delegated.count("plan") == (2 if ask_global or revise_after_plan else 1)
    assert output["item_count"] == (1 if revise_after_plan else 2)
    assert (
        output["planning_shortfall"] == output["total_shortfall"] == (0 if revise_after_plan else 2)
    )
    assert output["generation_shortfalls"] == {"arbitrary.identity": 0}
    blueprint = json.loads(Path(output["working_blueprint_path"]).read_text(encoding="utf-8"))
    assert "question_templates" not in blueprint["tasks"]["arbitrary.identity"]
    assert (Path(output["run_directory"]) / "templates.json").is_file()
    manifest = json.loads(Path(output["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["tasks"]["arbitrary.identity"]["concept_id"] == "status"
    assert manifest["allocation"]["planning_shortfall"] == (0 if revise_after_plan else 2)
    expected_info = updated_info if revise_after_plan else info
    assert blueprint["benchmark_info"] == manifest["benchmark_info"] == expected_info
    assert manifest["dataset"]["dataset_id"] == ("updated-test" if revise_after_plan else "test")
    assert manifest["allocation"]["requested_count"] == expected_info["sample_count"]
    if revise_after_plan:
        index = json.loads(
            (Path(output["run_directory"]) / ".runtime/artifacts.json").read_text(encoding="utf-8")
        )
        versions = [entry for entry in index["versions"].values() if entry["kind"] == "Plan"]
        assert len(versions) == 2
        original = json.loads(
            Path(versions[0]["references"]["blueprint"]).read_text(encoding="utf-8")
        )
        assert original["benchmark_info"] == info
        assert original["dataset"]["dataset_id"] == "test"
        assert set(original["tasks"]) == set(blueprint["tasks"]) == {"arbitrary.identity"}
