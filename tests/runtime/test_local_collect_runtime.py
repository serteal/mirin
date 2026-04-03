"""Shared-runtime collection stress tests for local ``mirin.Model``."""

from __future__ import annotations

from pathlib import Path

import torch

import mirin as ti

from ..helpers import FakeDecoderModel


def _batch(batch_size: int, seq_len: int, start: int = 0) -> dict[str, torch.Tensor]:
    tokens = (torch.arange(start, start + (batch_size * seq_len), dtype=torch.long) % 16).view(
        batch_size,
        seq_len,
    )
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }


def _assert_bounded_resources(stats: dict[str, object]) -> None:
    resources = stats["resources"]
    assert isinstance(resources, dict)
    assert resources["reserved_gpu_bytes"] == 0
    assert resources["reserved_cpu_bytes"] == 0
    gpu_capacity = resources["gpu_capacity_bytes"]
    cpu_capacity = resources["cpu_capacity_bytes"]
    if isinstance(gpu_capacity, int):
        assert isinstance(resources["peak_gpu_bytes"], int)
        assert resources["peak_gpu_bytes"] <= gpu_capacity
    if isinstance(cpu_capacity, int):
        assert isinstance(resources["peak_cpu_bytes"], int)
        assert resources["peak_cpu_bytes"] <= cpu_capacity


def test_local_collector_large_cpu_dataset_splits_without_leaking() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]
        dataset = [_batch(6, 12, start=idx * 128) for idx in range(4)]
        total_rows = 0
        for step in model.collect(
            dataset,
            get=[site],
            out="cpu",
            max_tokens=24,
        ):
            total_rows += len(step.indices)
            assert isinstance(step.batch["attention_mask"], torch.Tensor)
            value = step[site]
            assert isinstance(value, torch.Tensor)
            assert value.device.type == "cpu"
            step.release()

        assert total_rows == 6 * 4
        stats = model.stats()
        queue = stats["queues"]["collect_batch"]
        assert queue["total_batches"] > len(dataset)
        assert queue["max_batch_sessions"] <= 2
        _assert_bounded_resources(stats)
    finally:
        model.close()


def test_local_collect_process_runs_on_chunked_steps() -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]
        dataset = [_batch(6, 12, start=idx * 128) for idx in range(2)]

        def process(step: ti.CollectStep) -> torch.Tensor:
            value = step[site]
            assert isinstance(value, torch.Tensor)
            attention_mask = step.batch["attention_mask"]
            assert isinstance(attention_mask, torch.Tensor)
            valid_counts = attention_mask.sum(dim=1).to(torch.long)
            batch_idx = torch.arange(int(value.shape[0]), device=value.device)
            return value[batch_idx, valid_counts - 1].detach().cpu()

        processed = list(model.collect(dataset, get=[site], process=process, max_tokens=24))
        assert len(processed) >= len(dataset)
        assert all(isinstance(value, torch.Tensor) for value in processed)
        stats = model.stats()
        _assert_bounded_resources(stats)
    finally:
        model.close()


def test_local_collect_large_mmap_dataset_returns_manifest(
    tmp_path: Path,
) -> None:
    model = ti.Model(FakeDecoderModel())
    try:
        site = model.layers[0]
        dataset = [_batch(5, 12, start=idx * 160) for idx in range(4)]
        mmap_root = tmp_path / "acts"
        manifest = model.collect(
            dataset,
            get=[site],
            out=mmap_root,
            max_tokens=24,
        )
        assert manifest.rows == 5 * 4
        assert manifest.root == str(mmap_root)
        assert manifest.files[site.path]
        assert all(Path(filename).exists() for filename in manifest.files[site.path])
        stats = model.stats()
        queue = stats["queues"]["collect_batch"]
        assert queue["total_batches"] > len(dataset)
        _assert_bounded_resources(stats)
    finally:
        model.close()
