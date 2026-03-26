"""Phase 3 tests for batching and streaming."""

from __future__ import annotations

from pathlib import Path

import torch

import tinyinterp as ti

from .helpers import FakeGpt2Model


def _input_ids(batch: int = 1) -> torch.Tensor:
    rows = [[1, 2, 3, 4] for _ in range(batch)]
    return torch.tensor(rows, dtype=torch.long)


def test_batch_fuses_compatible_calls() -> None:
    torch.manual_seed(0)
    model = ti.Model(FakeGpt2Model())
    proxy = model.transformer.h[0].attn
    inputs = _input_ids()

    with torch.no_grad():
        expected_zero = model(inputs, map={proxy: ti.zero()}).logits
        expected_shift = model(inputs, map={proxy: ti.add(1.0)}).logits

    ti.Counters.reset()
    with ti.batch():
        out_zero = model(inputs, map={proxy: ti.zero()})
        out_shift = model(inputs, map={proxy: ti.add(1.0)})

    assert torch.allclose(out_zero.logits, expected_zero)
    assert torch.allclose(out_shift.logits, expected_shift)
    assert ti.Counters.calls == 2
    assert ti.Counters.forward_passes == 1
    assert ti.Counters.batch_groups == 1
    assert ti.Counters.batch_fusions == 1


def test_stream_chunks_batches_and_moves_activations_to_cpu() -> None:
    torch.manual_seed(0)
    model = ti.Model(FakeGpt2Model())
    proxy = model.transformer.h[0].attn
    dataloader = [{"input_ids": _input_ids(batch=4)}]

    outputs = list(model.stream(dataloader, get=[proxy], batch_size=2))

    assert len(outputs) == 2
    assert outputs[0][proxy].device.type == "cpu"
    assert outputs[0][proxy].shape[0] == 2
    assert outputs[1][proxy].shape[0] == 2

    expected0 = model(_input_ids(batch=2), get=[proxy])
    expected1 = model({"input_ids": _input_ids(batch=4)}["input_ids"][2:4], get=[proxy])
    assert torch.allclose(outputs[0][proxy], expected0[proxy].cpu())
    assert torch.allclose(outputs[1][proxy], expected1[proxy].cpu())


def test_stream_supports_direct_path_with_graph_context(
    tmp_path: Path,
) -> None:
    model = ti.Model(FakeGpt2Model())
    proxy = model.transformer.h[1]
    graph_path = tmp_path / "stream-graph.svg"
    batches = [{"input_ids": _input_ids(batch=2)}]

    with ti.context(graph=graph_path):
        outputs = list(model.stream(batches, get=[proxy], batch_size=1))

    assert len(outputs) == 2
    assert outputs[0][proxy].shape[0] == 1
