import asyncio
import pytest
from medbenchagent.runtime.sessions import create_experiment
from medbenchagent.tui import MBAApp
from textual.widgets import Select, TextArea


def test_new_rejects_existing_name_before_configuration(tmp_path):
    create_experiment("example", root=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        create_experiment("example", root=tmp_path)


def test_plain_text_status_and_incremental_run_rows(tmp_path):
    create_experiment("example", root=tmp_path)
    async def check():
        app = MBAApp(root=tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            app.query_one("#experiment", Select).value = "example"
            await pilot.pause()
            assert "[llm]" in app.query_one("#config", TextArea).text
            assert app.experiment == "example"
    asyncio.run(check())
