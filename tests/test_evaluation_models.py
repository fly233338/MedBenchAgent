import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from medbenchagent.evaluation.contracts import ImageEntry, InferenceSample, TaskSpec
from medbenchagent.evaluation.models import load_model_adapter
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


def _sample(tmp_path: Path) -> InferenceSample:
    image_path = tmp_path / "image.png"
    Image.new("L", (20, 10), 100).save(image_path)
    return InferenceSample(
        0,
        {
            "id": "one",
            "task_id": "view",
            "dataset": "dataset",
            "image_id": "image",
            "image_path": "opaque",
            "question": "Which view?",
            "options": ["CC", "MLO"],
            "answer": "A",
        },
        TaskSpec("view", "acquisition", "letter", "accuracy"),
        ImageEntry("image", image_path, "image.png", 20, 10, "a" * 64),
    )


def test_unknown_model_adapter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model adapter"):
        load_model_adapter("missing", "model")


def test_qwen_adapter_batches_and_reports_processed_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=torch.tensor([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, _device: torch.device) -> "FakeInputs":
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeProcessor":
            assert kwargs["min_pixels"] < kwargs["max_pixels"]
            return cls()

        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages[1]["content"][1]["type"] == "text"
            assert messages[1]["content"][0]["min_pixels"] == 256 * 28 * 28
            return "prompt"

        def __call__(self, **kwargs: object) -> FakeInputs:
            assert kwargs["text"] == ["prompt"]
            return FakeInputs()

        def batch_decode(self, _tokens: list[torch.Tensor], **_kwargs: object) -> list[str]:
            return ['{"answer":"A"}']

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeModel":
            assert kwargs["attn_implementation"] == "sdpa"
            return cls()

        def to(self, _device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> torch.Tensor:
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    processed = Image.new("RGB", (56, 28))
    qwen_utils = types.ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = lambda _messages: ([processed], [])
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FakeProcessor
    transformers.Qwen2_5_VLForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = Qwen25VLAdapter("model", device=torch.device("cpu"))
    outputs = adapter.generate([_sample(tmp_path)], max_new_tokens=16)

    assert outputs[0].raw_text == '{"answer":"A"}'
    assert outputs[0].coordinate_size == (56, 28)
