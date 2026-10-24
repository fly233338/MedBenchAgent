from textual import events
from textual.message import Message
from textual.widgets import TextArea, Static
from rich.text import Text


class Composer(TextArea):
    class CommandKey(Message):
        def __init__(self, key):
            super().__init__()
            self.key = key

    class Submitted(Message):
        def __init__(self, value):
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key):
        if self.text.startswith("/") and event.key in {"tab", "up", "down"}:
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandKey(event.key))
            return
        if event.key in {"enter", "shift+enter", "ctrl+j"}:
            event.prevent_default()
            event.stop()
            if event.key == "enter":
                self.post_message(self.Submitted(self.text))
            else:
                self.insert("\n")
            return
        await super()._on_key(event)


class ChatMessage(Static):
    def __init__(self, role, text):
        content = Text()
        content.append(role + "\n", style="bold")
        content.append(text)
        super().__init__(content, classes="chat-message")
