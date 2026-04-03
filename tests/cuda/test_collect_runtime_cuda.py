"""CUDA validation for shared-runtime local collection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mirin as ti

from ..helpers import FakeDecoderModel

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")


def _batch(batch_size: int, seq_len: int, start: int = 0) -> dict[str, torch.Tensor]:
    tokens = (torch.arange(start, start + (batch_size * seq_len), dtype=torch.long) % 16).view(
        batch_size,
        seq_len,
    )
    batch = {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }
    return {key: value.to("cuda") for key, value in batch.items()}


def test_cuda_local_collect_large_dataset_stays_within_runtime_capacity(tmp_path: Path) -> None:
    _require_cuda()

    model = ti.Model(FakeDecoderModel().to("cuda").eval())
    try:
        site = model.layers[0]
        dataset = [_batch(4, 24, start=idx * 256) for idx in range(4)]
        manifest = model.collect(
            dataset,
            get=[site],
            out=tmp_path / "acts",
            max_tokens=48,
        )
        assert manifest.rows == 4 * 4
        assert manifest.files[site.path]

        stats = model.stats()
        queue = stats["queues"]["collect_batch"]
        resources = stats["resources"]
        assert queue["total_batches"] > len(dataset)
        assert resources["reserved_gpu_bytes"] == 0
        gpu_capacity = resources["gpu_capacity_bytes"]
        if isinstance(gpu_capacity, int):
            assert isinstance(resources["peak_gpu_bytes"], int)
            assert resources["peak_gpu_bytes"] <= gpu_capacity
    finally:
        model.close()
