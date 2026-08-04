import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from medbenchagent.evaluation.contracts import ImageEntry, InferenceSample, TaskSpec
from medbenchagent.evaluation.models import load_model_adapter
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


def _sample(tmp_path: Path, response_format: str = "letter") -> InferenceSample:
    image_path = tmp_path / "image.png"
    Image.new("L", (20, 10), 100).save(image_path)
    options = ["CC", "MLO"] if response_format == "letter" else None
    answer = {"letter": "A", "free_text": "CC", "bbox": [1, 1, 10, 8]}[response_format]
    metric = {"letter": "accuracy", "free_text": "bleu", "bbox": "miou"}[response_format]
    return InferenceSample(
        0,
        {
            "id": "one",
            "task_id": "view",
            "dataset": "dataset",
            "image_id": "image",
            "image_path": "opaque",
            "question": "Which view?",
            "options": options,
            "answer": answer,
        },
        TaskSpec("view", "acquisition", response_format, metric),
        ImageEntry("image", image_path, "image.png", 20, 10, "a" * 64),
    )


def test_unknown_model_adapter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model adapter"):
        load_model_adapter("missing", "model")


@pytest.mark.parametrize(
    ("response_format", "expected_system", "expected_text", "raw_output"),
    [
        (
            "letter",
            "Answer with the correct single letter answer.",
            "Which view?\n\nA. CC\nB. MLO",
            "A",
        ),
        ("free_text", None, "Which view?", "CC"),
        (
            "bbox",
            'Answer with only a JSON object containing the single bounding box under the "bbox" '
            "key as four absolute pixel coordinates in x1, y1, x2, y2 order.",
            "Which view?",
            '{"bbox":[1,1,10,8]}',
        ),
    ],
)
def test_qwen_adapter_builds_task_messages_and_reports_processed_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_format: str,
    expected_system: str | None,
    expected_text: str,
    raw_output: str,
) -> None:
    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=torch.tensor([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, _device: torch.device) -> "FakeInputs":
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = types.SimpleNamespace(padding_side="right")

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeProcessor":
            assert kwargs["min_pixels"] < kwargs["max_pixels"]
            return cls()

        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            if expected_system is None:
                assert len(messages) == 1
                user_message = messages[0]
            else:
                assert messages[0] == {"role": "system", "content": expected_system}
                user_message = messages[1]
            assert user_message["role"] == "user"
            assert user_message["content"] == [
                {
                    "type": "image",
                    "image": user_message["content"][0]["image"],
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": expected_text},
            ]
            return "prompt"

        def __call__(self, **kwargs: object) -> FakeInputs:
            assert kwargs["text"] == ["prompt"]
            return FakeInputs()

        def batch_decode(self, _tokens: list[torch.Tensor], **_kwargs: object) -> list[str]:
            return [raw_output]

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
    assert adapter.processor.tokenizer.padding_side == "left"
    outputs = adapter.generate([_sample(tmp_path, response_format)], max_new_tokens=16)

    assert outputs[0].raw_text == raw_output
    assert outputs[0].coordinate_size == (56, 28)
