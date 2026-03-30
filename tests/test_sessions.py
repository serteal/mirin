"""Unit tests for session helpers."""

from __future__ import annotations

import pytest
import torch

from tinyinterp.server.sessions import SamplingConfig, merge_batch_tensors, sample_next_token


def test_sample_next_token_argmax_and_temperature_validation() -> None:
    logits = torch.tensor([[0.1, 0.5, 0.2]], dtype=torch.float32)

    greedy = sample_next_token(logits, SamplingConfig(do_sample=False))
    assert torch.equal(greedy, torch.tensor([[1]]))

    with pytest.raises(ValueError, match="temperature must be > 0"):
        sample_next_token(logits, SamplingConfig(do_sample=True, temperature=0.0))


def test_sample_next_token_respects_top_k() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 0.9, 0.2, 0.3]], dtype=torch.float32)
    sampled = sample_next_token(logits, SamplingConfig(do_sample=True, temperature=1.0, top_k=1))
    assert torch.equal(sampled, torch.tensor([[1]]))


def test_merge_batch_tensors_concatenates_tensor_values() -> None:
    merged = merge_batch_tensors(
        [
            {"input_ids": torch.tensor([[1, 2]]), "flag": "x"},
            {"input_ids": torch.tensor([[3, 4]]), "flag": "x"},
        ]
    )
    assert torch.equal(merged["input_ids"], torch.tensor([[1, 2], [3, 4]]))
    assert merged["flag"] == "x"
