import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from medbenchagent.evaluation.contracts import ImageEntry, InferenceSample, TaskSpec
from medbenchagent.evaluation.models import ADAPTERS
from medbenchagent.evaluation.models.base import system_prompt, user_prompt
from medbenchagent.evaluation.models.chiron_o1_8b import ChironO1_8BAdapter
from medbenchagent.evaluation.models.hulu_med_32b import HuluMed32BAdapter
from medbenchagent.evaluation.models.hulu_med_7b import HuluMed7BAdapter
from medbenchagent.evaluation.models.internvl import InternVLAdapter, dynamic_preprocess
from medbenchagent.evaluation.models.internvl3_5_8b import InternVL3_5_8BAdapter
from medbenchagent.evaluation.models.lingshu_32b import Lingshu32BAdapter
from medbenchagent.evaluation.models.qoq_med_vl_32b import QoQMedVL32BAdapter
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter
from medbenchagent.evaluation.models.qwen2_5_vl_32b_instruct import (
    Qwen25VL32BInstructAdapter,
)
from medbenchagent.evaluation.models.qwen3_vl import Qwen3VLAdapter
from medbenchagent.evaluation.models.qwen3_vl_32b_instruct import Qwen3VL32BInstructAdapter
from medbenchagent.evaluation.models.qwen3_vl_8b_thinking import Qwen3VL8BThinkingAdapter


EXPECTED_ADAPTERS = {
    "qwen2_5_vl",
    "qwen2_5_vl_7b_instruct",
    "qwen2_5_vl_32b_instruct",
    "internvl3_8b",
    "internvl3_5_8b",
    "qwen3_vl_8b_instruct",
    "qwen3_vl_8b_thinking",
    "qwen3_vl_32b_instruct",
    "lingshu_7b",
    "lingshu_32b",
    "hulu_med_7b",
    "hulu_med_32b",
    "qoq_med_vl_7b",
    "qoq_med_vl_32b",
    "chiron_o1_8b",
}


def _sample(tmp_path: Path, response_format: str, index: int = 0) -> InferenceSample:
    image_path = tmp_path / f"image-{index}.png"
    Image.new("L", (20, 10), 100).save(image_path)
    options = ["CC", "MLO"] if response_format == "letter" else None
    answer = {"letter": "A", "free_text": "CC", "bbox": [1, 1, 10, 8]}[response_format]
    metric = {"letter": "accuracy", "free_text": "bleu", "bbox": "miou"}[response_format]
    return InferenceSample(
        index,
        {
            "id": f"item-{index}",
            "task_id": "view",
            "dataset": "dataset",
            "image_id": f"image-{index}",
            "image_path": "opaque",
            "question": "Which view?",
            "options": options,
            "answer": answer,
        },
        TaskSpec("view", "acquisition", response_format, metric),
        ImageEntry(f"image-{index}", image_path, image_path.name, 20, 10, "a" * 64),
    )


def test_registry_contains_all_model_keys() -> None:
    assert set(ADAPTERS) == EXPECTED_ADAPTERS


def test_new_checkpoint_sizes_reuse_their_family_implementations() -> None:
    assert issubclass(Qwen25VL32BInstructAdapter, Qwen25VLAdapter)
    assert issubclass(Qwen3VL32BInstructAdapter, Qwen3VLAdapter)
    assert issubclass(Lingshu32BAdapter, Qwen25VLAdapter)
    assert issubclass(HuluMed32BAdapter, HuluMed7BAdapter)
    assert issubclass(InternVL3_5_8BAdapter, InternVLAdapter)
    assert issubclass(QoQMedVL32BAdapter, Qwen25VLAdapter)


def test_batch_script_uses_server_model_paths_and_flash_attention() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "evaluate_models.sh"
    ).read_text(encoding="utf-8")

    assert "ATTN_IMPLEMENTATION=flash_attention_2" in script
    assert (
        "QWEN2_5_VL_32B_INSTRUCT_PATH=/public/home/rxa-jj-wzk/models/"
        "Qwen2.5-VL/Qwen2.5-VL-32B-Instruct"
    ) in script
    assert (
        "INTERNVL3_5_8B_PATH=/public/home/rxa-jj-wzk/models/"
        "InternVL3.5/InternVL3_5-8B"
    ) in script
    assert "INTERNVL3_5_8B" in script
    assert (
        "QOQ_MED_VL_32B_PATH=/public/home/rxa-jj-wzk/models/"
        "QoQ-Med-VL/QoQ-Med-VL-32B"
    ) in script


def test_public_prompts_remain_qwen25_prompts(tmp_path: Path) -> None:
    letter = _sample(tmp_path, "letter")
    free_text = _sample(tmp_path, "free_text", 1)
    bbox = _sample(tmp_path, "bbox", 2)

    assert system_prompt(letter) == "Answer with the correct single letter answer."
    assert user_prompt(letter) == "Which view?\n\nA. CC\nB. MLO"
    assert system_prompt(free_text) is None
    assert user_prompt(free_text) == "Which view?"
    assert system_prompt(bbox) == (
        'Answer with only a JSON object containing the single bounding box under the "bbox" '
        "key as four absolute pixel coordinates in x1, y1, x2, y2 order."
    )


@pytest.mark.parametrize(
    ("adapter_class", "raw_text", "expected"),
    [
        (Qwen3VL8BThinkingAdapter, "reasoning</think>A", "A"),
        (Qwen3VL8BThinkingAdapter, "reasoning</think>", ""),
        (Qwen3VL8BThinkingAdapter, "A", "A"),
        (ChironO1_8BAdapter, "reasoning### The final answer is:A", "A"),
        (ChironO1_8BAdapter, "reasoning### The final answer is:", ""),
        (ChironO1_8BAdapter, "A", "A"),
    ],
)
def test_reasoning_adapters_extract_only_official_final_answer(
    adapter_class: type,
    raw_text: str,
    expected: str,
) -> None:
    adapter = object.__new__(adapter_class)
    assert adapter.extract_answer(raw_text) == expected


def test_qwen3_uses_patch_size_16_without_second_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

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
            calls["processor_load"] = kwargs
            return cls()

        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
            calls["messages"] = messages
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "prompt"

        def __call__(self, **kwargs: object) -> FakeInputs:
            calls["processor_call"] = kwargs
            return FakeInputs()

        def batch_decode(self, _tokens: list[torch.Tensor], **_kwargs: object) -> list[str]:
            return ["reasoning</think>A"]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeModel":
            calls["model_load"] = kwargs
            return cls()

        def to(self, _device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> torch.Tensor:
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    processed = Image.new("RGB", (64, 32))

    def fake_process(messages: list[dict], **kwargs: object) -> tuple[list[Image.Image], list]:
        assert kwargs == {"image_patch_size": 16}
        calls["vision_messages"] = messages
        return [processed], []

    qwen_utils = types.ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = fake_process
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FakeProcessor
    transformers.Qwen3VLForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = Qwen3VL8BThinkingAdapter("model", device=torch.device("cpu"))
    output = adapter.generate([_sample(tmp_path, "letter")], max_new_tokens=16)[0]

    assert adapter.processor.tokenizer.padding_side == "left"
    assert calls["processor_call"]["do_resize"] is False
    assert calls["processor_call"]["padding"] is True
    assert calls["messages"] == calls["vision_messages"]
    assert output.raw_text == "reasoning</think>A"
    assert output.answer_text == "A"
    assert output.coordinate_size == (64, 32)


def test_internvl_dynamic_preprocess_adds_thumbnail_for_multiple_tiles() -> None:
    tiles = dynamic_preprocess(Image.new("RGB", (896, 448)))

    assert len(tiles) == 3
    assert all(tile.size == (448, 448) for tile in tiles)


def test_internvl_groups_by_system_prompt_and_restores_sample_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"batches": []}

    class FakeModel:
        dtype = torch.float32
        system_message = "checkpoint default"

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeModel":
            calls["model_load"] = kwargs
            return cls()

        def to(self, _device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def batch_chat(self, _tokenizer: object, pixels: torch.Tensor, **kwargs: object) -> list[str]:
            calls["batches"].append(
                {
                    "system": self.system_message,
                    "pixels": pixels.shape[0],
                    **kwargs,
                }
            )
            return [f"response:{question}" for question in kwargs["questions"]]

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeTokenizer":
            calls["tokenizer_load"] = kwargs
            return cls()

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    patch_counts = iter([2, 1, 3])
    monkeypatch.setattr(
        "medbenchagent.evaluation.models.internvl.load_image",
        lambda *_args, **_kwargs: torch.zeros((next(patch_counts), 3, 448, 448)),
    )
    samples = [
        _sample(tmp_path, "letter", 0),
        _sample(tmp_path, "free_text", 1),
        _sample(tmp_path, "bbox", 2),
    ]

    adapter = InternVLAdapter("model", device=torch.device("cpu"))
    outputs = adapter.generate(samples, max_new_tokens=24)

    assert calls["model_load"] == {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "use_flash_attn": False,
        "trust_remote_code": True,
    }
    assert calls["tokenizer_load"] == {"trust_remote_code": True, "use_fast": False}
    assert [batch["system"] for batch in calls["batches"]] == [
        system_prompt(samples[0]),
        "",
        system_prompt(samples[2]),
    ]
    assert [batch["num_patches_list"] for batch in calls["batches"]] == [[2], [1], [3]]
    assert [output.raw_text for output in outputs] == [
        f"response:<image>\n{user_prompt(sample)}" for sample in samples
    ]
    assert all(output.coordinate_size == (1000, 1000) for output in outputs)
    assert adapter.model.system_message == "checkpoint default"


def test_hulu_processes_each_conversation_and_decodes_same_ids_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"conversations": [], "decode": [], "generate": []}

    class FakeModel:
        dtype = torch.float32

        def __init__(self) -> None:
            self.count = 0

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeModel":
            calls["model_load"] = kwargs
            return cls()

        def to(self, _device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> torch.Tensor:
            self.count += 1
            calls["generate"].append(kwargs)
            return torch.tensor([[self.count]])

    class FakeProcessor:
        tokenizer = types.SimpleNamespace()

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "FakeProcessor":
            calls["processor_load"] = kwargs
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            calls["conversations"].append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float64),
            }

        def batch_decode(self, output_ids: torch.Tensor, **kwargs: object) -> list[str]:
            item = int(output_ids.item())
            calls["decode"].append((item, kwargs))
            if kwargs["use_think"]:
                return [f"reasoning-{item}"]
            return ["A" if item == 1 else "report"]

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeModel
    transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    samples = [_sample(tmp_path, "letter", 0), _sample(tmp_path, "free_text", 1)]

    adapter = HuluMed7BAdapter("model", device=torch.device("cpu"))
    outputs = adapter.generate(samples, max_new_tokens=32)

    assert len(calls["conversations"]) == 2
    first_conversation = calls["conversations"][0]
    assert first_conversation["add_system_prompt"] is False
    assert first_conversation["add_generation_prompt"] is True
    assert first_conversation["conversation"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt(samples[0])}],
    }
    assert calls["conversations"][1]["conversation"][0]["role"] == "user"
    assert all(call["do_sample"] is False for call in calls["generate"])
    assert all(call["pixel_values"].dtype == torch.float32 for call in calls["generate"])
    assert [item for item, _kwargs in calls["decode"]] == [1, 1, 2, 2]
    assert [(output.raw_text, output.answer_text) for output in outputs] == [
        ("reasoning-1", "A"),
        ("reasoning-2", "report"),
    ]
    assert all(output.coordinate_size == (20, 10) for output in outputs)
