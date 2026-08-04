"""Transformers adapter for Qwen2.5-VL checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


class Qwen25VLAdapter:
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
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()

    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        conversations = []
        images = []
        coordinate_sizes = []
        for sample in samples:
            with Image.open(sample.image.png_path) as source:
                image = source.convert("RGB")
            messages = []
            instruction = system_prompt(sample)
            if instruction is not None:
                messages.append({"role": "system", "content": instruction})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": user_prompt(sample)},
                    ],
                }
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            if video_inputs or len(image_inputs) != 1:
                raise ValueError("Qwen2.5-VL evaluation requires exactly one image per item.")
            processed_image = image_inputs[0]
            conversations.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(processed_image)
            coordinate_sizes.append(processed_image.size)

        inputs = self.processor(
            text=conversations,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != len(samples):
            raise RuntimeError("Qwen2.5-VL returned an unexpected number of outputs.")
        return [
            ModelOutput(raw_text=text, coordinate_size=size)
            for text, size in zip(decoded, coordinate_sizes)
        ]
