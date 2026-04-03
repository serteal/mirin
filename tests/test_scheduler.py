"""Unit tests for scheduler helpers."""

from __future__ import annotations

import torch

import mirin as ti
from mirin.runtime.plans import compile_plan
from mirin.runtime.scheduler import (
    QueueMetrics,
    bucket_length,
    chunk_sessions_by_decode_budget,
    estimate_activation_bytes,
    estimate_kv_cache_bytes,
)

from .helpers import FakeLlamaModel


def test_bucket_length_rounds_up_and_handles_zero() -> None:
    assert bucket_length(0, 64) == 0
    assert bucket_length(65, 64) == 128
    assert bucket_length(63, 1) == 63


def test_chunk_sessions_by_decode_budget_splits_by_token_budget() -> None:
    sessions = [1, 2, 3, 4, 5]
    chunks = chunk_sessions_by_decode_budget(
        sessions,
        total_tokens_per_session=8,
        max_batch_tokens=16,
    )
    assert chunks == [[1, 2], [3, 4], [5]]


def test_queue_metrics_snapshot_reports_means() -> None:
    metrics = QueueMetrics(
        started=2,
        completed=2,
        total_queue_wait_ns=4_000_000,
        total_service_ns=10_000_000,
    )
    snapshot = metrics.snapshot()
    assert snapshot["mean_queue_wait_ms"] == 2.0
    assert snapshot["mean_service_ms"] == 5.0


def test_estimate_bytes_helpers_use_model_config() -> None:
    model = FakeLlamaModel()
    plan = compile_plan(model=ti.Model(model), get=["model.layers.0"], mapping=None)
    kv = estimate_kv_cache_bytes(model, dtype=torch.float32, batch_size=2, total_tokens=16)
    acts = estimate_activation_bytes(
        model,
        plan=plan,
        dtype=torch.float32,
        batch_size=2,
        seq_len=16,
    )
    assert kv > 0
    assert acts > 0
