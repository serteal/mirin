"""Shared runtime helpers for the inference server engines."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, cast

import torch

from ..output import Output


def move_tensors_to(value: Any, device: torch.device) -> Any:
    """Move nested tensors onto one device, short-circuiting when possible."""

    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device=device)
    if isinstance(value, tuple):
        return tuple(move_tensors_to(item, device) for item in value)
    if isinstance(value, list):
        return [move_tensors_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_tensors_to(item, device) for key, item in value.items()}
    return value


def default_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    like: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is not None:
        return cast(torch.Tensor, move_tensors_to(attention_mask, device))
    return torch.ones(like.shape[:2], dtype=torch.long, device=device)


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


def to_cpu(value: torch.Tensor, *, enabled: bool, pin_memory: bool = False) -> torch.Tensor:
    if not enabled:
        return value
    tensor = value.detach()
    if tensor.device.type == "cpu":
        return tensor.clone()
    if not pin_memory:
        return tensor.cpu()
    staging = torch.empty_like(tensor, device="cpu", pin_memory=True)
    staging.copy_(tensor, non_blocking=True)
    if tensor.device.type == "cuda":
        torch.cuda.current_stream(tensor.device).synchronize()
    return staging


def to_cpu_dict(
    values: Mapping[str, Any],
    *,
    enabled: bool,
    pin_memory: bool = False,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            output[key] = to_cpu(value, enabled=enabled, pin_memory=pin_memory)
        else:
            output[key] = value
    return output


def split_batch_tensor(tensor: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [tensor[idx : idx + 1] for idx in range(batch_size)]


def split_activation_dict(
    activations: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = [{} for _ in range(batch_size)]
    for path, value in activations.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == batch_size:
            for idx in range(batch_size):
                outputs[idx][path] = value[idx : idx + 1]
        else:
            for idx in range(batch_size):
                outputs[idx][path] = value
    return outputs


def batch_size_from_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    return 1


def prompt_tokens_from_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
        return int(input_ids.shape[0] * input_ids.shape[1])
    return 0


def model_dtype(wrapped: torch.nn.Module) -> torch.dtype:
    try:
        return next(wrapped.parameters()).dtype
    except StopIteration:
        return torch.float32


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


def gpu_stats(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"gpu_memory_allocated_mb": 0.0, "gpu_memory_reserved_mb": 0.0}
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return {
        "gpu_memory_allocated_mb": allocated,
        "gpu_memory_reserved_mb": reserved,
    }


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


def is_static_cache(cache: Any) -> bool:
    return type(cache).__name__.lower() == "staticcache"


def supports_static_cache_model(wrapped: torch.nn.Module) -> bool:
    config = getattr(wrapped, "config", None)
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
