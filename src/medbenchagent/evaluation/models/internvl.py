"""Shared Transformers remote-code implementation for InternVL-family checkpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_difference = float("inf")
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference and (
            area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Apply InternVL's official aspect-ratio tiling policy."""
    width, height = image.size
    target_ratios = sorted(
        {
            (columns, rows)
            for count in range(min_num, max_num + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if min_num <= columns * rows <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = _closest_aspect_ratio(
        width / height,
        target_ratios,
        width,
        height,
        image_size,
    )
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(path: str, *, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    transform = _build_transform(input_size)
    tiles = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    return torch.stack([transform(tile) for tile in tiles])


class InternVLAdapter:
    coordinate_size = (1000, 1000)

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
        from transformers import AutoModel, AutoTokenizer

        del min_pixels, max_pixels
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=attn_implementation == "flash_attention_2",
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.compute_dtype = self.model.dtype if torch_dtype == "auto" else torch_dtype

    def extract_answer(self, _text: str) -> str | None:
        return None

    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        pixel_values = [
            load_image(str(sample.image.png_path), max_num=12).to(
                device=self.device,
                dtype=self.compute_dtype,
            )
            for sample in samples
        ]
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[system_prompt(sample) or ""].append(index)

        decoded: list[str | None] = [None] * len(samples)
        original_system_message = self.model.system_message
        try:
            for instruction, indices in grouped.items():
                group_pixels = torch.cat([pixel_values[index] for index in indices])
                patch_counts = [pixel_values[index].size(0) for index in indices]
                questions = [f"<image>\n{user_prompt(samples[index])}" for index in indices]
                self.model.system_message = instruction
                responses = self.model.batch_chat(
                    self.tokenizer,
                    group_pixels,
                    num_patches_list=patch_counts,
                    questions=questions,
                    generation_config={
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                    },
                )
                if len(responses) != len(indices):
                    raise RuntimeError("InternVL returned an unexpected number of outputs.")
                for index, response in zip(indices, responses):
                    decoded[index] = response
        finally:
            self.model.system_message = original_system_message

        if any(text is None for text in decoded):
            raise RuntimeError("InternVL did not return an output for every sample.")
        return [
            ModelOutput(
                raw_text=text,
                coordinate_size=self.coordinate_size,
                answer_text=self.extract_answer(text),
            )
            for text in decoded
            if text is not None
        ]
