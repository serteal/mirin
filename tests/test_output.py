"""Unit tests for output helpers and activation lookup."""

from __future__ import annotations

import torch

import mirin as ti
from mirin.output import _lengths_from_field, output_from_path_activations

from .helpers import FakeDecoderModel


def test_local_output_supports_string_path_and_foreign_proxy_lookup() -> None:
    model = ti.Model(FakeDecoderModel())
    other = ti.Model(FakeDecoderModel())
    site = model.layers[0]
    foreign_site = other.layers[0]

    output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), get=[site])

    assert torch.allclose(output["transformer.h.0"], output[site])
    assert torch.allclose(output[foreign_site], output[site])


def test_lengths_from_field_rejects_non_integral_values() -> None:
    try:
        _lengths_from_field([1.9], 1, name="generated_length")
    except TypeError as exc:
        assert "integers" in str(exc)
    else:
        raise AssertionError("Expected non-integral lengths to raise TypeError.")


def test_activation_view_items_remain_lazy_until_iteration() -> None:
    calls: list[str] = []

    class _LazyValue:
        def resolve(self) -> torch.Tensor:
            calls.append("resolve")
            return torch.ones(1, 2, 3)

    output = output_from_path_activations({}, {"site": _LazyValue()})

    items = output.activations.items()
    values = output.activations.values()

    assert calls == []
    resolved_items = list(items)
    assert len(resolved_items) == 1
    assert resolved_items[0][0] == "site"
    assert torch.equal(resolved_items[0][1], torch.ones(1, 2, 3))
    assert calls == ["resolve"]
    resolved_values = list(values)
    assert len(resolved_values) == 1
    assert torch.equal(resolved_values[0], torch.ones(1, 2, 3))
    assert calls == ["resolve", "resolve"]
