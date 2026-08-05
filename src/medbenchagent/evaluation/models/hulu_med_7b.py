"""Transformers remote-code adapter for Hulu-Med-7B."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


class HuluMed7BAdapter:
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
        from transformers import AutoModelForCausalLM, AutoProcessor

        del min_pixels, max_pixels
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.compute_dtype = self.model.dtype if torch_dtype == "auto" else torch_dtype

    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        outputs = []
        for sample in samples:
            conversation = []
            instruction = system_prompt(sample)
            if instruction is not None:
                conversation.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": instruction}],
                    }
                )
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": {"image_path": str(sample.image.png_path)},
                        },
                        {"type": "text", "text": user_prompt(sample)},
                    ],
                }
            )
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.compute_dtype)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            raw_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=True,
            )[0].strip()
            answer_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=False,
            )[0].strip()
            outputs.append(
                ModelOutput(
                    raw_text=raw_text,
                    coordinate_size=(sample.image.width, sample.image.height),
                    answer_text=answer_text,
                )
            )
        return outputs
