"""MS-SWIFT adapter for the BreastGPT-8B checkpoint."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput
from medbenchagent.evaluation.models.base import system_prompt, user_prompt


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LEGACY_BOX = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.DOTALL)
_PLUGIN_PACKAGE = "model"
_loaded_plugin_root: Path | None = None


def _load_checkpoint_plugin(model_path: str) -> None:
    """Load the bundled plugin under the package name expected by its imports."""
    global _loaded_plugin_root

    plugin_root = (Path(model_path).resolve() / "plugin").resolve()
    plugin_init = plugin_root / "__init__.py"
    if not plugin_init.is_file():
        raise FileNotFoundError(
            "BreastGPT requires a local ModelScope checkpoint containing plugin/__init__.py: "
            f"{plugin_init}"
        )
    if _loaded_plugin_root == plugin_root:
        return
    if _loaded_plugin_root is not None:
        raise RuntimeError(
            "A different BreastGPT checkpoint plugin is already loaded in this process: "
            f"{_loaded_plugin_root}"
        )
    if _PLUGIN_PACKAGE in sys.modules:
        raise RuntimeError(
            "BreastGPT cannot load because the checkpoint plugin package name 'model' is already "
            "in use. Run this adapter in a fresh evaluation process."
        )

    spec = importlib.util.spec_from_file_location(
        _PLUGIN_PACKAGE,
        plugin_init,
        submodule_search_locations=[str(plugin_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the BreastGPT checkpoint plugin: {plugin_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PLUGIN_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for module_name in tuple(sys.modules):
            if module_name == _PLUGIN_PACKAGE or module_name.startswith(f"{_PLUGIN_PACKAGE}."):
                sys.modules.pop(module_name, None)
        raise
    _loaded_plugin_root = plugin_root


def _legacy_bbox_answer(text: str) -> str | None:
    candidates = set()
    for payload in _LEGACY_BOX.findall(text):
        values = re.findall(_NUMBER, payload)
        if len(values) == 4:
            candidates.add(tuple(float(value) for value in values))
    if len(candidates) != 1:
        return None
    return json.dumps({"bbox": list(candidates.pop())})


class BreastGPT8BAdapter:
    """BreastGPT-8B through its bundled model and template registrations."""

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
        del min_pixels, max_pixels
        try:
            from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        except ModuleNotFoundError as exc:
            if exc.name == "swift":
                raise RuntimeError(
                    "BreastGPT evaluation requires ms-swift from requirements/evaluation.txt."
                ) from exc
            raise

        _load_checkpoint_plugin(model_path)
        torch_dtype = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.infer_request_class = InferRequest
        self.request_config_class = RequestConfig
        self.engine = TransformersEngine(
            str(Path(model_path).resolve()),
            model_type="breastgpt",
            template_type="breastgpt",
            max_batch_size=0,
            torch_dtype=torch_dtype,
            attn_impl=attn_implementation,
            device_map=str(device),
        )

    @staticmethod
    def extract_answer(sample: InferenceSample, text: str) -> str | None:
        if sample.task.response_format == "bbox":
            return _legacy_bbox_answer(text)
        return None

    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        requests = []
        for sample in samples:
            messages: list[dict[str, Any]] = []
            instruction = system_prompt(
                sample,
                bbox_coordinate_space=(
                    "normalized to the fixed [0, 1000] range on each image axis"
                ),
            )
            if instruction is not None:
                messages.append({"role": "system", "content": instruction})
            messages.append(
                {
                    "role": "user",
                    "content": f"<image>\n{user_prompt(sample)}",
                }
            )
            requests.append(
                self.infer_request_class(
                    messages=messages,
                    images=[str(sample.image.png_path)],
                )
            )

        responses = self.engine.infer(
            requests,
            self.request_config_class(max_tokens=max_new_tokens, temperature=0),
            use_tqdm=False,
        )
        if len(responses) != len(samples):
            raise RuntimeError("BreastGPT returned an unexpected number of outputs.")

        outputs = []
        for sample, response in zip(samples, responses):
            text = response.choices[0].message.content
            if not isinstance(text, str):
                raise RuntimeError("BreastGPT returned a non-text response.")
            text = text.strip()
            outputs.append(
                ModelOutput(
                    raw_text=text,
                    coordinate_size=(1000, 1000),
                    answer_text=self.extract_answer(sample, text),
                )
            )
        return outputs
