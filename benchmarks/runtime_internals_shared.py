"""Shared helpers for runtime-internals benchmarks."""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn

import mirin as ti
from mirin.hooks import _extract
from mirin.output import Output

from .model_api import _config_value


def extract_logits(result: Any) -> torch.Tensor:
    model_output = result._model_output if isinstance(result, Output) else result
    logits = getattr(model_output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), torch.Tensor):
        return cast(torch.Tensor, model_output["logits"])
    if isinstance(model_output, torch.Tensor):
        return model_output
    raise TypeError(f"Cannot extract logits from {type(model_output).__name__}.")


def extract_last_token_logits(result: Any) -> torch.Tensor:
    logits = extract_logits(result)
    while logits.ndim > 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Expected logits with 2 or 3 dims, got shape {tuple(logits.shape)}.")


def filter_supported_kwargs(
    wrapped: torch.nn.Module,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(wrapped.forward)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def eos_token_ids(wrapped: torch.nn.Module) -> set[int]:
    config = getattr(wrapped, "config", None)
    if config is None:
        return set()
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, (list, tuple, set)):
        return {int(token_id) for token_id in eos}
    return set()


def contains_eos(token_ids: torch.Tensor, eos_ids: set[int]) -> bool:
    if not eos_ids:
        return False
    return int(token_ids.view(-1)[0].item()) in eos_ids


def _requests_from_batch(batch: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    return [
        {
            "input_ids": input_ids[idx],
            "attention_mask": attention_mask[idx],
        }
        for idx in range(input_ids.shape[0])
    ]


def _prepare_generation_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any | None,
) -> dict[str, Any]:
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if not callable(prepare):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": True,
        }
    prepare_kwargs = {
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "use_cache": True,
    }
    cache_position = _cache_position(cache, input_ids)
    if cache_position is not None:
        prepare_kwargs["cache_position"] = cache_position
    prepared = cast(Mapping[str, Any], prepare(input_ids, **prepare_kwargs))
    return {**prepared, "use_cache": True}


def _cache_position(cache: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    if cache is None:
        start = 0
    elif callable(getattr(cache, "get_seq_length", None)):
        start = int(cache.get_seq_length())
    else:
        return None
    return torch.arange(start, start + input_ids.shape[-1], device=input_ids.device)


class _ManualHookLoop:
    def __init__(self, model: nn.Module, site_path: str) -> None:
        self.model = model
        self.module = _resolve_module(model, site_path)
        self.captured: dict[str, torch.Tensor] = {}
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        self.captured["act"] = _extract(output).detach().cpu()

    def run(self, dataset: list[dict[str, torch.Tensor]]) -> float:
        total = 0.0
        with torch.no_grad():
            for batch in dataset:
                self.captured.clear()
                _ = self.model(**batch, use_cache=False)
                total += float(self.captured["act"].float().sum().item())
        return total

    def close(self) -> None:
        self.handle.remove()


def _manual_hook_once(
    model: nn.Module,
    site_path: str,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        captured["act"] = _extract(output).detach().cpu()

    handle = _resolve_module(model, site_path).register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**batch, use_cache=False)
    finally:
        handle.remove()
    return captured["act"]


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return cast(nn.Module, current)


def _resolve_proxy(model: ti.Model, path: str) -> Any:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _open_remote_model(sock_path: str) -> Any:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if os.path.exists(sock_path):
            try:
                return ti.Model(f"unix://{sock_path}")
            except OSError:
                time.sleep(0.02)
                continue
        time.sleep(0.02)
    raise RuntimeError(f"Remote benchmark server did not open {sock_path}.")


def _supports_static_cache(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        config = get_text(decoder=True)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return True
    supported = {"full_attention", "sliding_attention", "chunked_attention"}
    return all(layer_type in supported for layer_type in layer_types)


def _load_model(
    config: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Server benchmarks require `transformers`.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
    )
    return model.to(device=device).eval()


def _make_dataset(
    *,
    batch_size: int,
    batches: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    outputs: list[dict[str, torch.Tensor]] = []
    for _ in range(batches):
        input_ids = torch.randint(
            low=3,
            high=max(vocab_size, 4),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        outputs.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return outputs


def _site_path(model_name: str, model: nn.Module) -> str:
    del model_name
    wrapper = ti.Model(model, rename=ti.renames.llm)
    site = ti.find(wrapper.layers[0], "linear_attn")
    if site is None:
        site = ti.find(wrapper.layers[0], "self_attn")
    if site is None:
        site = ti.find(wrapper.layers[0], "attn")
    if site is None:
        raise RuntimeError("Could not infer a benchmark site path.")
    return site.path


def _vocab_size(model: nn.Module) -> int:
    config = getattr(model, "config", SimpleNamespace(vocab_size=256))
    return int(_config_value(config, "vocab_size") or 256)


def _reshape_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    d_head = tensor.shape[-1] // n_heads
    return tensor.view(*tensor.shape[:-1], n_heads, d_head)


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pad_batch_sequences(sequences: list[torch.Tensor], *, pad_token_id: int) -> torch.Tensor:
    max_len = max(sequence.shape[1] for sequence in sequences)
    padded: list[torch.Tensor] = []
    for sequence in sequences:
        if sequence.shape[1] == max_len:
            padded.append(sequence)
            continue
        pad = torch.full(
            (sequence.shape[0], max_len - sequence.shape[1]),
            pad_token_id,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        padded.append(torch.cat([sequence, pad], dim=-1))
    return torch.cat(padded, dim=0)


def _normalize_generated_batch(value: Any, *, pad_token_id: int) -> torch.Tensor:
    if isinstance(value, ti.GenerateOutput):
        return cast(torch.Tensor, value.sequences)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, ti.GenerateOutput) for item in value)
    ):
        return _pad_batch_sequences(
            [cast(torch.Tensor, item.sequences) for item in cast(list[ti.GenerateOutput], value)],
            pad_token_id=pad_token_id,
        )
    if not isinstance(value, list) or not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(
            f"Expected GenerateOutput or list[GenerateOutput], got {type(value).__name__}."
        )
    return _pad_batch_sequences(cast(list[torch.Tensor], value), pad_token_id=pad_token_id)


def _pad_token_id(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return 0
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(config, "eos_token_id", 0)
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
        return int(eos_token_id[0])
    return 0
