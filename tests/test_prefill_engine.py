"""Unit tests for prefill-engine helpers."""

from __future__ import annotations

import torch

from mirin.server.prefill_engine import _apply_reducer, _merge_chunk_result
from mirin.server.results import PlanResult


def test_apply_reducer_handles_token_index_and_last_token() -> None:
    activations = {"site": torch.arange(12, dtype=torch.float32).view(1, 4, 3)}
    reduced = _apply_reducer(activations, reducer="last_token", token_index=None)
    assert torch.equal(reduced["site"], activations["site"][:, -1])

    indexed = _apply_reducer(activations, reducer=None, token_index=-1)
    assert torch.equal(indexed["site"], activations["site"][:, -1:])


def test_merge_chunk_result_concatenates_activation_time_dimension() -> None:
    first = PlanResult(
        activations={"site": torch.ones(1, 2, 3)},
        completed_forward=False,
    )
    second = PlanResult(
        activations={"site": torch.zeros(1, 1, 3)},
        completed_forward=True,
        metadata={"k": 1},
    )

    merged = _merge_chunk_result(first, second)

    assert torch.equal(
        merged.activations["site"],
        torch.cat([torch.ones(1, 2, 3), torch.zeros(1, 1, 3)], dim=1),
    )
    assert merged.completed_forward is True
    assert merged.metadata == {"k": 1}
