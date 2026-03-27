"""Session state and cache helpers for the inference server."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from .plans import CompiledPlan


@dataclass(slots=True)
class SamplingConfig:
    """Sampling settings carried by an open session."""

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None


@dataclass(slots=True)
class Session:
    """Persistent generation state owned by the server."""

    id: str
    plan: CompiledPlan
    cache_mode: str
    sampling: SamplingConfig
    use_hf_cache: bool
    current_length: int = 0
    max_total_tokens: int | None = None
    max_new_tokens_hint: int | None = None
    decode_bucket_len: int | None = None
    prompt_length: int = 0
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    pending_input_ids: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None
    cache: Any | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    family_key: tuple[str, ...] | None = None
    slot_index: int | None = None
    history_cpu: list[int] = field(default_factory=list)
    generated_cpu: list[int] = field(default_factory=list)
    finished: bool = False

    @property
    def compatibility_key(self) -> tuple[str, str, int]:
        bucket = self.decode_bucket_len or self.current_length
        return (self.plan.fingerprint, self.cache_mode, bucket)


def create_cache(
    wrapped: torch.nn.Module,
    cache_mode: str,
    *,
    max_cache_len: int | None = None,
) -> Any | None:
    """Create a HuggingFace cache object when the wrapped model supports it."""

    if cache_mode == "none":
        return None
    if not hasattr(wrapped, "prepare_inputs_for_generation"):
        return None
    config = getattr(wrapped, "config", None)
    if cache_mode == "dynamic":
        try:
            from transformers import DynamicCache
        except ImportError:
            return None
        if config is None:
            return DynamicCache()
        return DynamicCache(config=config)
    if cache_mode == "static":
        if max_cache_len is None or max_cache_len <= 0:
            raise ValueError("Static cache requires a positive max_cache_len.")
        if config is None:
            raise ValueError("Static cache requires a HuggingFace config.")
        try:
            from transformers import StaticCache
        except ImportError:
            return None
        return StaticCache(config=config, max_cache_len=max_cache_len)
    raise ValueError(f"Unsupported cache mode {cache_mode!r}.")


def merge_caches(caches: list[Any], wrapped: torch.nn.Module) -> Any:
    """Merge per-session caches into one batch cache for decode microbatching."""

    if not caches:
        raise ValueError("merge_caches() requires at least one cache.")
    if len(caches) == 1:
        return caches[0]
    try:
        from transformers import DynamicCache
    except ImportError as exc:
        raise RuntimeError("transformers is required for batched cache decode.") from exc
    if not all(isinstance(cache, DynamicCache) for cache in caches):
        raise RuntimeError("merge_caches() currently supports DynamicCache only.")

    ddp_cache_data: list[tuple[torch.Tensor, ...]] = []
    n_layers = len(caches[0].layers)
    for layer_idx in range(n_layers):
        layer_keys: list[torch.Tensor] = []
        layer_values: list[torch.Tensor] = []
        first_layer = caches[0].layers[layer_idx]
        for cache in caches:
            layer = cache.layers[layer_idx]
            layer_keys.append(cast(torch.Tensor, layer.keys))
            layer_values.append(cast(torch.Tensor, layer.values))
        merged: tuple[torch.Tensor, ...]
        if hasattr(first_layer, "_sliding_window_tensor"):
            merged = (
                torch.cat(layer_keys, dim=0),
                torch.cat(layer_values, dim=0),
                cast(torch.Tensor, first_layer._sliding_window_tensor),
            )
        else:
            merged = (torch.cat(layer_keys, dim=0), torch.cat(layer_values, dim=0))
        ddp_cache_data.append(merged)
    return DynamicCache(ddp_cache_data=ddp_cache_data, config=getattr(wrapped, "config", None))


def split_cache(cache: Any, wrapped: torch.nn.Module, batch_size: int) -> list[Any]:
    """Split a batch cache back into one cache object per session."""

    if batch_size == 1:
        return [cache]
    try:
        from transformers import DynamicCache
    except ImportError as exc:
        raise RuntimeError("transformers is required for batched cache decode.") from exc

    outputs: list[Any] = []
    n_layers = len(cache.layers)
    config = getattr(wrapped, "config", None)
    for batch_idx in range(batch_size):
        ddp_cache_data: list[tuple[torch.Tensor, ...]] = []
        for layer_idx in range(n_layers):
            layer = cache.layers[layer_idx]
            entry: tuple[torch.Tensor, ...]
            if hasattr(layer, "_sliding_window_tensor"):
                entry = (
                    cast(torch.Tensor, layer.keys)[batch_idx : batch_idx + 1].clone(),
                    cast(torch.Tensor, layer.values)[batch_idx : batch_idx + 1].clone(),
                    cast(torch.Tensor, layer._sliding_window_tensor),
                )
            else:
                entry = (
                    cast(torch.Tensor, layer.keys)[batch_idx : batch_idx + 1].clone(),
                    cast(torch.Tensor, layer.values)[batch_idx : batch_idx + 1].clone(),
                )
            ddp_cache_data.append(entry)
        outputs.append(DynamicCache(ddp_cache_data=ddp_cache_data, config=config))
    return outputs


def sample_next_token(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    """Sample one next token from a batch of logits."""

    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, vocab] logits, got shape {tuple(logits.shape)}.")
    if not sampling.do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    distribution = logits
    if sampling.temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True.")
    distribution = distribution / sampling.temperature
    if sampling.top_k is not None and sampling.top_k > 0:
        k = min(sampling.top_k, distribution.shape[-1])
        top_values, top_indices = torch.topk(distribution, k=k, dim=-1)
        probs = torch.softmax(top_values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(-1, sampled)
    probs = torch.softmax(distribution, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def merge_batch_tensors(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-session prepared kwargs into one batched kwargs dict."""

    merged: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in items))
    for key in keys:
        values = [item.get(key) for item in items]
        sample = next((value for value in values if value is not None), None)
        if sample is None:
            continue
        if isinstance(sample, torch.Tensor):
            merged[key] = torch.cat(
                [value for value in values if isinstance(value, torch.Tensor)],
                dim=0,
            )
            continue
        merged[key] = sample
    return merged
