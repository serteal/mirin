"""Batching tests for tinyinterp."""

from __future__ import annotations

import pytest
import torch

import tinyinterp as ti
from tinyinterp.batch import _stack_values

from .helpers import FakeDecoderModel


def _input_ids(batch: int = 1) -> torch.Tensor:
    rows = [[1, 2, 3, 4] for _ in range(batch)]
    return torch.tensor(rows, dtype=torch.long)


def test_batch_fuses_compatible_calls() -> None:
    torch.manual_seed(0)
    model = ti.Model(FakeDecoderModel())
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


def test_batch_fuses_nonadjacent_compatible_calls() -> None:
    torch.manual_seed(0)
    model = ti.Model(FakeDecoderModel())
    first_proxy = model.transformer.h[0].attn
    second_proxy = model.transformer.h[1].attn
    inputs = _input_ids()

    with torch.no_grad():
        expected_zero = model(inputs, map={first_proxy: ti.zero()}).logits
        expected_mid = model(inputs, map={second_proxy: ti.zero()}).logits
        expected_shift = model(inputs, map={first_proxy: ti.add(1.0)}).logits

    ti.Counters.reset()
    with ti.batch():
        out_zero = model(inputs, map={first_proxy: ti.zero()})
        out_mid = model(inputs, map={second_proxy: ti.zero()})
        out_shift = model(inputs, map={first_proxy: ti.add(1.0)})

    assert torch.allclose(out_zero.logits, expected_zero)
    assert torch.allclose(out_mid.logits, expected_mid)
    assert torch.allclose(out_shift.logits, expected_shift)
    assert ti.Counters.calls == 3
    assert ti.Counters.forward_passes == 2
    assert ti.Counters.batch_groups == 2
    assert ti.Counters.batch_fusions == 1


def test_model_has_no_stream_api() -> None:
    model = ti.Model(FakeDecoderModel())

    with pytest.raises(AttributeError, match="stream"):
        _ = model.stream


def test_stop_at_last_get_is_rejected_inside_batch() -> None:
    model = ti.Model(FakeDecoderModel())
    proxy = model.transformer.h[0]
    inputs = _input_ids()

    with ti.batch():
        with pytest.raises(ValueError, match="not supported inside ti.batch"):
            _ = model(inputs, get=[proxy], stop_at_last_get=True)


def test_single_passthrough_inside_batch_preserves_wrapped_return_type() -> None:
    model = ti.Model(FakeDecoderModel())

    with ti.batch():
        deferred = model(_input_ids())

    resolved = deferred.resolve()

    assert not isinstance(resolved, ti.Output)
    assert hasattr(resolved, "logits")


def test_stack_values_rejects_mismatched_non_tensor_literals() -> None:
    with pytest.raises(ValueError, match="mismatched non-tensor values"):
        _stack_values([5, 6])
