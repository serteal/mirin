"""Core tests for the proxy-based tinyinterp API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import tinyinterp as ti
from tinyinterp.hooks import _extract, _replace

from .helpers import FakeGpt2Model, FakeLlamaModel, get_module, get_proxy


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


@pytest.mark.parametrize("factory", [FakeGpt2Model, FakeLlamaModel])
def test_passthrough_matches_wrapped_model(factory: Callable[[], nn.Module]) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    model = ti.Model(wrapped)

    with torch.no_grad():
        expected = wrapped(_input_ids())
        actual = model(_input_ids())

    assert type(actual) is type(expected)
    assert torch.allclose(actual.logits, expected.logits)


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (FakeGpt2Model, "transformer.h.1.attn"),
        (FakeLlamaModel, "model.layers.1.self_attn"),
    ],
)
def test_get_matches_manual_hook(
    factory: Callable[[], nn.Module],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        captured["act"] = _extract(output).detach()

    handle = get_module(wrapped, path).register_forward_hook(capture)
    try:
        wrapped(_input_ids())
    finally:
        handle.remove()

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    output = model(_input_ids(), get=[proxy])

    assert torch.allclose(output[proxy], captured["act"])


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (FakeGpt2Model, "transformer.h.1.attn"),
        (FakeLlamaModel, "model.layers.1.self_attn"),
    ],
)
def test_map_matches_manual_hook(
    factory: Callable[[], nn.Module],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped = factory()

    def manual_zero(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        return _replace(output, torch.zeros_like(_extract(output)))

    handle = get_module(wrapped, path).register_forward_hook(manual_zero)
    try:
        with torch.no_grad():
            expected = wrapped(_input_ids()).logits
    finally:
        handle.remove()

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    with torch.no_grad():
        actual = model(_input_ids(), map={proxy: ti.zero()}).logits

    assert torch.allclose(actual, expected)


def test_call_cleans_up_after_exception() -> None:
    wrapped = FakeGpt2Model()
    block0 = cast(Any, wrapped.transformer.h[0])
    block0.fail = True
    model = ti.Model(wrapped)

    with pytest.raises(RuntimeError, match="boom"):
        _ = model(_input_ids(), get=[model.transformer.h[0]])

    block0.fail = False
    output = model(_input_ids(), get=[model.transformer.h[0]])

    assert output[model.transformer.h[0]].shape[1] == _input_ids().shape[1]


@pytest.mark.parametrize(
    ("factory", "expected_prefix"),
    [
        (FakeGpt2Model, "transformer.h"),
        (FakeLlamaModel, "model.layers"),
    ],
)
def test_layers_finds_biggest_modulelist(
    factory: Callable[[], nn.Module],
    expected_prefix: str,
) -> None:
    model = ti.Model(factory())

    assert len(model.layers) == 2
    assert model.layers[1].path == f"{expected_prefix}.1"


def test_find_and_children_explore_real_tree() -> None:
    model = ti.Model(FakeGpt2Model())

    found = ti.find(model.layers[0], "attn")
    assert found == model.transformer.h[0].attn

    listed = dict(ti.children(model.layers[0]))
    assert listed["attn"] == "FakeGpt2Attention"
    assert listed["mlp"] == "FakeGpt2Mlp"


def test_rename_pack_exposes_canonical_aliases() -> None:
    model = ti.Model(FakeGpt2Model(), rename=ti.renames.llm)

    canonical = model.model.layers[0].self_attn
    real = model.transformer.h[0].attn

    assert canonical == real
    assert canonical.path == "transformer.h.0.attn"


def test_counters_debug_and_graph(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ti.Counters.reset()
    model = ti.Model(FakeGpt2Model())
    graph_path = tmp_path / "graph.svg"

    with ti.context(debug=2, graph=graph_path):
        output = model(_input_ids(), get=[model.transformer.h[0].attn])

    assert ti.Counters.calls == 1
    assert ti.Counters.forward_passes == 1
    assert ti.Counters.activations_captured == 1
    assert (
        ti.Counters.activations_bytes
        == output[model.transformer.h[0].attn].numel()
        * output[model.transformer.h[0].attn].element_size()
    )
    assert graph_path.exists()

    stdout = capsys.readouterr().out
    assert "[ti] call:" in stdout
    assert "transformer.h.0.attn" in stdout


def test_map_head_targets_only_one_slice() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    fn = ti.map_head(1, ti.zero(), n_heads=3)
    out = fn(x)

    assert torch.equal(out[..., :4], x[..., :4])
    assert torch.equal(out[..., 4:8], torch.zeros_like(out[..., 4:8]))
    assert torch.equal(out[..., 8:], x[..., 8:])
