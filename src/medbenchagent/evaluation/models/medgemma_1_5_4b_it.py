"""Transformers adapter for google/medgemma-1.5-4b-it."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from PIL import Image

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


_FINAL_ANSWER_MARKER = "FINAL_ANSWER:"
_BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE)
_EXPLICIT_LETTER = re.compile(
    r"\b(?:the\s+)?(?:(?:final|correct)\s+)?answer\s+is\s*[:\-]?\s*"
    r"(?:\*{1,2}|\$\s*)*(?:\\boxed\{\s*)?([A-Z])",
    re.IGNORECASE,
)
_CATEGORY_LETTER = re.compile(
    r"\b(?:the\s+)?(?:most\s+)?(?:appropriate\s+|best\s+|correct\s+)?"
    r"(?:BI-RADS\s+)?(?:assessment\s+)?category\s+is\s*"
    r"(?:\*{1,2}|\$\s*)*([A-Z])\s*\.",
    re.IGNORECASE,
)


def _image_coordinate_size(processor: object) -> tuple[int, int]:
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        width, height = size.get("width"), size.get("height")
        if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
            return width, height
    return 896, 896


class MedGemma15_4BITAdapter:
    """MedGemma 1.5 4B multimodal inference using its native Gemma chat template."""

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        del min_pixels, max_pixels
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        self.coordinate_size = _image_coordinate_size(self.processor)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()
        self.compute_dtype = self.model.dtype if torch_dtype == "auto" else torch_dtype

    def prompt_for(self, sample: InferenceSample) -> str:
        if sample.task.response_format == "letter":
            options = sample.item["options"]
            assert isinstance(options, list)
            final_format = (
                "one valid option letter. The valid letters are A through "
                f"{chr(ord('A') + len(options) - 1)}"
            )
        elif sample.task.response_format == "free_text":
            final_format = "only the concise final answer"
        else:
            final_format = (
                'only a JSON object containing the single bounding box under the "bbox" key '
                f"as four absolute xyxy coordinates in the processed {self.coordinate_size[0]} by "
                f"{self.coordinate_size[1]} image space"
            )
        return (
            f"{user_prompt(sample)}\n\n"
            "You may reason before answering. End the response with exactly one line beginning "
            f"with {_FINAL_ANSWER_MARKER} followed by {final_format}. Do not write anything after "
            "that line."
        )

    def extract_answer(self, sample: InferenceSample, text: str) -> str:
        if _FINAL_ANSWER_MARKER in text:
            return text.rpartition(_FINAL_ANSWER_MARKER)[2].strip()
        if sample.task.response_format != "letter":
            return text

        options = sample.item["options"]
        assert isinstance(options, list)
        valid_letters = {chr(ord("A") + index) for index in range(len(options))}
        candidates = {
            match.group(1).upper()
            for pattern in (_BOXED_LETTER, _EXPLICIT_LETTER, _CATEGORY_LETTER)
            for match in pattern.finditer(text)
            if match.group(1).upper() in valid_letters
        }
        if len(candidates) == 1:
            return candidates.pop()
        if len(candidates) > 1:
            return ""

        option_matches = {
            letter
            for index, option in enumerate(options)
            if isinstance(option, str)
            for letter in [chr(ord("A") + index)]
            if re.search(
                rf"\b{letter}\.\s*(?:\*{{1,2}})?{re.escape(option)}(?:\*{{1,2}})?",
                text,
                re.IGNORECASE,
            )
        }
        if len(option_matches) == 1:
            return option_matches.pop()
        if len(option_matches) > 1:
            return ""
        return text

    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        conversations = []
        for sample in samples:
            with Image.open(sample.image.png_path) as source:
                image = source.convert("RGB")
            messages = []
            instruction = system_prompt(
                sample,
                bbox_coordinate_space=(
                    f"in the processed {self.coordinate_size[0]} by "
                    f"{self.coordinate_size[1]} image space"
                ),
            )
            if instruction is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": instruction}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt_for(sample)},
                    ],
                }
            )
            conversations.append(messages)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=self.compute_dtype)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        decoded = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != len(samples):
            raise RuntimeError("MedGemma returned an unexpected number of outputs.")
        return [
            ModelOutput(
                raw_text=text.strip(),
                coordinate_size=self.coordinate_size,
                answer_text=self.extract_answer(sample, text.strip()),
            )
            for sample, text in zip(samples, decoded)
        ]
