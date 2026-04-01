"""Unit tests for executor internals."""

from __future__ import annotations

import pytest

import mirin as ti
from mirin.executors import _map_cache_key

from .helpers import FakeDecoderModel


def test_local_executor_layers_rejects_non_modulelist_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ti.Model(FakeDecoderModel())

    def bad_wrap_proxy(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr("mirin.model._wrap_proxy", bad_wrap_proxy)
    model._layers_proxy = None

    with pytest.raises(TypeError, match="ModuleList proxy"):
        _ = model.layers


def test_map_cache_key_uses_callable_identity_not_reused_ids() -> None:
    def fn(x: object) -> object:
        return x

    same_a = _map_cache_key(fn)
    same_b = _map_cache_key(fn)
    other = _map_cache_key(lambda x: x)

    assert same_a == same_b
    assert same_a != other
