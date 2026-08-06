"""Transformers adapter for google/medgemma-1.5-4b-it."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


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
            instruction = system_prompt(sample)
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
                        {"type": "text", "text": user_prompt(sample)},
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
            ModelOutput(raw_text=text.strip(), coordinate_size=self.coordinate_size)
            for text in decoded
        ]
