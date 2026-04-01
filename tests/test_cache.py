"""Unit tests for cache adapter validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import transformers

from mirin.server.cache import DynamicBatchAdapter


class _FakeDynamicCache:
    def __init__(
        self,
        ddp_cache_data: list[tuple[torch.Tensor, ...]] | None = None,
        config: object | None = None,
    ) -> None:
        del config
        self.layers = []
        for keys, values, *rest in ddp_cache_data or []:
            layer = SimpleNamespace(keys=keys, values=values)
            if rest:
                layer._sliding_window_tensor = rest[0]
            self.layers.append(layer)


def test_dynamic_batch_adapter_validates_keep_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformers, "DynamicCache", _FakeDynamicCache)
    adapter = DynamicBatchAdapter()
    cache = _FakeDynamicCache(
        ddp_cache_data=[
            (
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 4),
            )
        ]
    )

    with pytest.raises(ValueError, match="sorted"):
        adapter.compact_cache(cache, [1, 0], torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="unique"):
        adapter.compact_cache(cache, [0, 0], torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="within"):
        adapter.compact_cache(cache, [2], torch.nn.Linear(1, 1))
