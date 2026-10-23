import json
import os
from pathlib import Path
import subprocess
import pytest
from medbenchagent.agents.data_processing.coding.skills import freeze_skills, verify_skills
from medbenchagent.settings import load_coding_settings


def test_frozen_skills_bind_all_resources_and_detect_changes(tmp_path, monkeypatch):
    import medbenchagent.agents.data_processing.coding.skills as skills
    snapshot = freeze_skills(tmp_path, ["medical-image"])
    monkeypatch.setattr(skills, "files", lambda *_: pytest.fail("Read live skills"))
    root = verify_skills(tmp_path, snapshot)
    reference = root / "medical-image/references/coordinates.md"
    assert reference.is_file()
    reference.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_skills(tmp_path, snapshot)


def test_native_sdk_loads_and_invokes_frozen_skill_offline(tmp_path):
    settings = load_coding_settings()
    if not Path(settings.worker_python).is_file():
        pytest.skip("Provision the independent Coding environment for the SDK contract check")
    snapshot = freeze_skills(tmp_path, settings.skill_names)
    code = '''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from medbenchagent.agents.data_processing.coding.skills import FrozenSkills,load_native_skills
from openhands.sdk.context import AgentContext
from openhands.sdk.tool.builtins.invoke_skill import InvokeSkillAction,InvokeSkillExecutor
directory, raw = json.load(sys.stdin)
skills = load_native_skills(Path(directory), FrozenSkills.model_validate(raw))
ctx = AgentContext(skills=skills,load_user_skills=False,load_public_skills=False,load_project_skills=False,load_memory=False)
conversation = SimpleNamespace(state=SimpleNamespace(agent=SimpleNamespace(agent_context=ctx),workspace=SimpleNamespace(working_dir=directory),invoked_skills=[]))
result = InvokeSkillExecutor()(InvokeSkillAction(name="tabular-inspection"), conversation)
assert not result.is_error and "inspect_table" in result.text
assert conversation.state.invoked_skills == ["tabular-inspection"]
print(json.dumps([skill.name for skill in skills]))
'''
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "PYTHONUTF8": "1"}
    result = subprocess.run([settings.worker_python, "-c", code],
                            input=json.dumps([str(tmp_path), snapshot.model_dump()]), capture_output=True,
                            text=True, encoding="utf-8", env=environment, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == settings.skill_names
