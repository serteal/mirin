"""Cache adapters for server-side decode families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch


class CacheAdapter(Protocol):
    name: str

    def supports_cache(self, cache: Any) -> bool: ...

    def supports_batched_decode(self) -> bool: ...

    def append_cache(
        self,
        existing: Any | None,
        new_cache: Any,
        wrapped: torch.nn.Module,
    ) -> Any: ...

    def compact_cache(
        self,
        cache: Any | None,
        keep_indices: Sequence[int],
        wrapped: torch.nn.Module,
    ) -> Any | None: ...


@dataclass(slots=True)
class FallbackCacheAdapter:
    """Conservative adapter used when no batched cache strategy is available."""

    name: str = "fallback"

    def supports_cache(self, cache: Any) -> bool:
        return cache is None

    def supports_batched_decode(self) -> bool:
        return False

    def append_cache(self, existing: Any | None, new_cache: Any, wrapped: torch.nn.Module) -> Any:
        del existing, wrapped
        return new_cache

    def compact_cache(
        self,
        cache: Any | None,
        keep_indices: Sequence[int],
        wrapped: torch.nn.Module,
    ) -> Any | None:
        del keep_indices, wrapped
        return cache


@dataclass(slots=True)
class DynamicBatchAdapter:
    """Persistent-batch adapter for HuggingFace DynamicCache."""

    name: str = "dynamic_batch"

    def supports_cache(self, cache: Any) -> bool:
        try:
            from transformers import DynamicCache
        except ImportError:
            return False
        return isinstance(cache, DynamicCache)

    def supports_batched_decode(self) -> bool:
        return True

    def append_cache(self, existing: Any | None, new_cache: Any, wrapped: torch.nn.Module) -> Any:
        if existing is None:
            return new_cache
        try:
            from transformers import DynamicCache
        except ImportError as exc:
            raise RuntimeError("transformers is required for batched cache decode.") from exc
        if not isinstance(existing, DynamicCache) or not isinstance(new_cache, DynamicCache):
            raise RuntimeError("DynamicBatchAdapter requires DynamicCache.")
        ddp_cache_data: list[tuple[torch.Tensor, ...]] = []
        for old_layer, new_layer in zip(existing.layers, new_cache.layers, strict=True):
            if hasattr(old_layer, "_sliding_window_tensor"):
                ddp_cache_data.append(
                    (
                        torch.cat(
                            [
                                cast(torch.Tensor, old_layer.keys),
                                cast(torch.Tensor, new_layer.keys),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                cast(torch.Tensor, old_layer.values),
                                cast(torch.Tensor, new_layer.values),
                            ],
                            dim=0,
                        ),
                        cast(torch.Tensor, old_layer._sliding_window_tensor),
                    )
                )
            else:
                ddp_cache_data.append(
                    (
                        torch.cat(
                            [
                                cast(torch.Tensor, old_layer.keys),
                                cast(torch.Tensor, new_layer.keys),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                cast(torch.Tensor, old_layer.values),
                                cast(torch.Tensor, new_layer.values),
                            ],
                            dim=0,
                        ),
                    )
                )
        return DynamicCache(ddp_cache_data=ddp_cache_data, config=getattr(wrapped, "config", None))

    def compact_cache(
        self,
        cache: Any | None,
        keep_indices: Sequence[int],
        wrapped: torch.nn.Module,
    ) -> Any | None:
        if cache is None:
            return None
        try:
            from transformers import DynamicCache
        except ImportError as exc:
            raise RuntimeError("transformers is required for batched cache decode.") from exc
        if not isinstance(cache, DynamicCache):
            raise RuntimeError("DynamicBatchAdapter requires DynamicCache.")
        if not keep_indices:
            return None
        first_keys = cast(torch.Tensor, cache.layers[0].keys)
        index = torch.tensor(
            list(keep_indices),
            device=first_keys.device,
            dtype=torch.long,
        )
        ddp_cache_data: list[tuple[torch.Tensor, ...]] = []
        for layer in cache.layers:
            if hasattr(layer, "_sliding_window_tensor"):
                ddp_cache_data.append(
                    (
                        cast(torch.Tensor, layer.keys).index_select(0, index),
                        cast(torch.Tensor, layer.values).index_select(0, index),
                        cast(torch.Tensor, layer._sliding_window_tensor),
                    )
                )
            else:
                ddp_cache_data.append(
                    (
                        cast(torch.Tensor, layer.keys).index_select(0, index),
                        cast(torch.Tensor, layer.values).index_select(0, index),
                    )
                )
        return DynamicCache(ddp_cache_data=ddp_cache_data, config=getattr(wrapped, "config", None))


@dataclass(slots=True)
class StaticFallbackAdapter:
    """Static-cache adapter placeholder until batched static decode is benchmarked in."""

    name: str = "static_fallback"

    def supports_cache(self, cache: Any) -> bool:
        return type(cache).__name__.lower() == "staticcache"

    def supports_batched_decode(self) -> bool:
        return False

    def append_cache(self, existing: Any | None, new_cache: Any, wrapped: torch.nn.Module) -> Any:
        del existing, wrapped
        return new_cache

    def compact_cache(
        self,
        cache: Any | None,
        keep_indices: Sequence[int],
        wrapped: torch.nn.Module,
    ) -> Any | None:
        del keep_indices, wrapped
        return cache


@dataclass(slots=True)
class QwenHybridAdapter:
    """Persistent-batch adapter for hybrid HF caches with both KV and linear state."""

    name: str = "hybrid_state_batch"

    def supports_cache(self, cache: Any) -> bool:
        if type(cache).__name__ == "Qwen3_5DynamicCache":
            return True
        required = ("key_cache", "value_cache", "conv_states", "recurrent_states", "layer_types")
        return all(hasattr(cache, attr) for attr in required)

    def supports_batched_decode(self) -> bool:
        return True

    def append_cache(self, existing: Any | None, new_cache: Any, wrapped: torch.nn.Module) -> Any:
        if existing is None:
            return new_cache
        if not self.supports_cache(existing) or not self.supports_cache(new_cache):
            raise RuntimeError("QwenHybridAdapter requires a supported hybrid cache type.")
        merged = _new_hybrid_cache_like(new_cache, wrapped)
        for attr in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            merged_values: list[torch.Tensor | None] = []
            old_values = cast(list[torch.Tensor | None], getattr(existing, attr))
            new_values = cast(list[torch.Tensor | None], getattr(new_cache, attr))
            for old_item, new_item in zip(old_values, new_values, strict=True):
                merged_values.append(_concat_optional_batch(old_item, new_item))
            setattr(merged, attr, merged_values)
        return merged

    def compact_cache(
        self,
        cache: Any | None,
        keep_indices: Sequence[int],
        wrapped: torch.nn.Module,
    ) -> Any | None:
        if cache is None:
            return None
        if not keep_indices:
            return None
        if not self.supports_cache(cache):
            raise RuntimeError("QwenHybridAdapter requires a supported hybrid cache type.")
        first_tensor = _first_hybrid_state(cache)
        if first_tensor is None:
            return cache
        index = torch.tensor(list(keep_indices), device=first_tensor.device, dtype=torch.long)
        compacted = _new_hybrid_cache_like(cache, wrapped)
        for attr in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            compacted_values: list[torch.Tensor | None] = []
            values = cast(list[torch.Tensor | None], getattr(cache, attr))
            for value in values:
                compacted_values.append(
                    None if value is None else value.index_select(0, index)
                )
            setattr(compacted, attr, compacted_values)
        return compacted


def _concat_optional_batch(
    existing: torch.Tensor | None,
    new_value: torch.Tensor | None,
) -> torch.Tensor | None:
    if existing is None:
        return new_value
    if new_value is None:
        return existing
    return torch.cat([existing, new_value], dim=0)


def _first_hybrid_state(cache: Any) -> torch.Tensor | None:
    for attr in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        for value in cast(list[torch.Tensor | None], getattr(cache, attr)):
            if isinstance(value, torch.Tensor):
                return value
    return None


def _resolve_cache_config(wrapped: torch.nn.Module) -> Any:
    config = getattr(wrapped, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config(decoder=True)
        except TypeError:
            return get_text_config()
    return config


def _new_hybrid_cache_like(cache: Any, wrapped: torch.nn.Module) -> Any:
    cache_type = type(cache)
    config = _resolve_cache_config(wrapped)
    try:
        return cache_type(config=config)
    except TypeError:
        if config is not None:
            try:
                return cache_type(config)
            except TypeError as exc:
                raise RuntimeError(
                    f"Could not construct hybrid cache type {cache_type.__name__}."
                ) from exc
        raise RuntimeError(
            f"Could not construct hybrid cache type {cache_type.__name__}."
        ) from None


def select_cache_adapter(cache: Any | None, cache_mode: str) -> CacheAdapter:
    """Choose the narrowest adapter that can serve one session family."""

    if cache_mode == "dynamic" and cache is None:
        return DynamicBatchAdapter()
    if cache_mode == "static":
        return StaticFallbackAdapter()
    for adapter in (DynamicBatchAdapter(), QwenHybridAdapter()):
        if cache is not None and adapter.supports_cache(cache):
            return adapter
    return FallbackCacheAdapter()
