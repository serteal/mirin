"""High-level model/runtime API tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import mirin as ti

from ..helpers import FakeDecoderModel


def _row(length: int, start: int) -> dict[str, torch.Tensor]:
    tokens = (torch.arange(start, start + length, dtype=torch.long) % 16).unsqueeze(0)
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }


def _batch(batch_size: int, seq_len: int, start: int = 0) -> dict[str, torch.Tensor]:
    tokens = (torch.arange(start, start + (batch_size * seq_len), dtype=torch.long) % 16).view(
        batch_size,
        seq_len,
    )
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }


def _activation_tensor(output: Any, site: Any) -> torch.Tensor:
    value = output[site]
    assert isinstance(value, torch.Tensor)
    return value


def test_model_exposes_local_capacity_and_stats() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        capacity = model.capacity
        assert capacity["device_type"] in {"cpu", "cuda"}
        assert capacity["collect_token_budget"] is None or capacity["collect_token_budget"] >= 1

        initial = model.stats()
        assert initial["requests_served"] == 0

        site = model.layers[0]
        outputs = model.collect([_row(5, 0), _row(3, 8)], get=[site], max_tokens=4)

        assert len(outputs) == 2
        assert all(output.partial for output in outputs)
        assert _activation_tensor(outputs[0], site).shape[0] == 1
        for output in outputs:
            output.release()

        stats = model.stats()
        assert stats["requests_served"] >= 1
        assert stats["queues"]["collect_batch"]["total_batches"] >= 1
    finally:
        model.close()


def test_model_collect_accepts_batched_inputs_and_iterables() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]
        outputs = model.collect(_batch(3, 5), get=[site], max_tokens=4)
        assert len(outputs) == 3
        assert all(output.partial for output in outputs)
        for output in outputs:
            assert _activation_tensor(output, site).shape[0] == 1
            output.release()

        dataset = [_batch(2, 4, start=32), [_row(4, 64), _row(2, 80)]]
        for step in model.collect(dataset, get=[site], out="cpu", max_tokens=4):
            assert len(step.indices) >= 1
            assert isinstance(step.batch["attention_mask"], torch.Tensor)
            value = step[site]
            assert isinstance(value, torch.Tensor)
            assert value.device.type == "cpu"
            step.release()

        stats = model.stats()
        assert stats["queues"]["collect_batch"]["total_batches"] >= 2
    finally:
        model.close()


def test_model_collect_processes_each_step_locally() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]

        def process(step: Any) -> torch.Tensor:
            value = step[site]
            assert isinstance(value, torch.Tensor)
            attention_mask = step.batch["attention_mask"]
            assert isinstance(attention_mask, torch.Tensor)
            valid_counts = attention_mask.sum(dim=1).to(torch.long)
            batch_idx = torch.arange(int(value.shape[0]), device=value.device)
            return value[batch_idx, valid_counts - 1].detach().cpu()

        processed = list(
            model.collect(
                [_row(5, 0), _row(3, 8)],
                get=[site],
                process=process,
                max_tokens=4,
            )
        )
        assert len(processed) == 2
        assert all(isinstance(value, torch.Tensor) for value in processed)
        assert all(value.ndim == 2 for value in processed)
    finally:
        model.close()


def test_model_collect_rejects_removed_public_args() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]
        with pytest.raises(TypeError, match="Use process="):
            model.collect([_row(5, 0)], get=[site], reduce="last")
        with pytest.raises(TypeError, match="Use out="):
            model.collect([_row(5, 0)], get=[site], activation_output="cpu")
        with pytest.raises(TypeError, match="Use max_tokens="):
            model.collect([_row(5, 0)], get=[site], token_budget=8)
    finally:
        model.close()
