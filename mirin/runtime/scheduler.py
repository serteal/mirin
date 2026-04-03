"""Scheduler helpers for queue accounting and admission control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from .plans import CompiledPlan


@dataclass(slots=True)
class SchedulerConfig:
    """Runtime scheduling and admission-control settings."""

    decode_bucket_multiple: int = 64
    decode_max_batch_tokens: int | None = None
    prefill_token_budget: int | None = None
    collect_token_budget: int | None = None
    max_kv_cache_bytes: int | None = None
    max_activation_capture_bytes: int | None = None


@dataclass(slots=True)
class QueueMetrics:
    """Per-queue counters tracked by the local runtime."""

    enqueued: int = 0
    started: int = 0
    completed: int = 0
    rejected: int = 0
    current_depth: int = 0
    peak_depth: int = 0
    total_queue_wait_ns: int = 0
    total_service_ns: int = 0
    physical_tokens: int = 0
    physical_context_tokens: int = 0
    physical_batches: int = 0
    physical_sessions: int = 0
    max_batch_sessions: int = 0
    split_batches: int = 0
    split_items: int = 0
    split_reasons: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def record_physical_batch(
        self,
        *,
        batch_size: int,
        batch_tokens: int,
        context_tokens: int | None = None,
    ) -> None:
        self.physical_batches += 1
        self.physical_tokens += batch_tokens
        self.physical_sessions += batch_size
        self.max_batch_sessions = max(self.max_batch_sessions, batch_size)
        if context_tokens is not None:
            self.physical_context_tokens += context_tokens

    def record_split(
        self,
        *,
        reason: str,
        original_items: int,
        produced_chunks: int,
    ) -> None:
        if produced_chunks <= 1:
            return
        self.split_batches += produced_chunks - 1
        self.split_items += original_items
        self.split_reasons[reason] = self.split_reasons.get(reason, 0) + 1

    def record_reject(self, reason: str | None) -> None:
        self.rejected += 1
        if reason is not None:
            self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        mean_wait_ms = 0.0 if self.started == 0 else (self.total_queue_wait_ns / self.started) / 1e6
        mean_service_ms = (
            0.0 if self.completed == 0 else (self.total_service_ns / self.completed) / 1e6
        )
        return {
            "enqueued": self.enqueued,
            "started": self.started,
            "completed": self.completed,
            "rejected": self.rejected,
            "current_depth": self.current_depth,
            "peak_depth": self.peak_depth,
            "mean_queue_wait_ms": mean_wait_ms,
            "mean_service_ms": mean_service_ms,
            "total_tokens": self.physical_tokens,
            "total_context_tokens": self.physical_context_tokens,
            "total_batches": self.physical_batches,
            "total_sessions": self.physical_sessions,
            "max_batch_sessions": self.max_batch_sessions,
            "split_batches": self.split_batches,
            "split_items": self.split_items,
            "split_reasons": dict(self.split_reasons),
            "reject_reasons": dict(self.reject_reasons),
        }


@dataclass(slots=True)
class AdmissionEstimate:
    """Estimated resource footprint for a request before execution."""

    queue: str
    prompt_tokens: int
    projected_decode_tokens: int
    batch_size: int
    bucket_tokens: int
    kv_cache_bytes: int
    activation_bytes: int
    admitted: bool
    reason: str | None = None

    def snapshot(self) -> dict[str, int | bool | str | None]:
        return {
            "queue": self.queue,
            "prompt_tokens": self.prompt_tokens,
            "projected_decode_tokens": self.projected_decode_tokens,
            "batch_size": self.batch_size,
            "bucket_tokens": self.bucket_tokens,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activation_bytes": self.activation_bytes,
            "admitted": self.admitted,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ResourceLedger:
    """Track in-flight resource reservations across concurrent local work."""

    reserved_gpu_bytes: int = 0
    reserved_cpu_bytes: int = 0
    reserved_kv_cache_bytes: int = 0
    reserved_activation_bytes: int = 0
    peak_gpu_bytes: int = 0
    peak_cpu_bytes: int = 0
    peak_kv_cache_bytes: int = 0
    peak_activation_bytes: int = 0

    def snapshot(
        self,
        *,
        gpu_capacity_bytes: int | None,
        cpu_capacity_bytes: int | None,
        kv_capacity_bytes: int | None,
        activation_capacity_bytes: int | None,
    ) -> dict[str, int | None]:
        return {
            "reserved_gpu_bytes": self.reserved_gpu_bytes,
            "reserved_cpu_bytes": self.reserved_cpu_bytes,
            "reserved_kv_cache_bytes": self.reserved_kv_cache_bytes,
            "reserved_activation_bytes": self.reserved_activation_bytes,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "peak_cpu_bytes": self.peak_cpu_bytes,
            "peak_kv_cache_bytes": self.peak_kv_cache_bytes,
            "peak_activation_bytes": self.peak_activation_bytes,
            "gpu_capacity_bytes": gpu_capacity_bytes,
            "cpu_capacity_bytes": cpu_capacity_bytes,
            "kv_capacity_bytes": kv_capacity_bytes,
            "activation_capacity_bytes": activation_capacity_bytes,
        }

    def try_reserve(
        self,
        *,
        kv_bytes: int,
        activation_bytes: int,
        gpu_bytes: int,
        cpu_bytes: int,
        gpu_capacity_bytes: int | None,
        cpu_capacity_bytes: int | None,
        kv_capacity_bytes: int | None,
        activation_capacity_bytes: int | None,
    ) -> str | None:
        if kv_capacity_bytes is not None and kv_bytes > kv_capacity_bytes:
            return "kv_cache_budget"
        if (
            activation_capacity_bytes is not None
            and activation_bytes > activation_capacity_bytes
        ):
            return "activation_budget"
        if gpu_capacity_bytes is not None and gpu_bytes > gpu_capacity_bytes:
            return "gpu_budget"
        if cpu_capacity_bytes is not None and cpu_bytes > cpu_capacity_bytes:
            return "cpu_budget"
        if (
            kv_capacity_bytes is not None
            and self.reserved_kv_cache_bytes + kv_bytes > kv_capacity_bytes
        ):
            return "kv_cache_budget"
        if (
            activation_capacity_bytes is not None
            and self.reserved_activation_bytes + activation_bytes > activation_capacity_bytes
        ):
            return "activation_budget"
        if (
            gpu_capacity_bytes is not None
            and self.reserved_gpu_bytes + gpu_bytes > gpu_capacity_bytes
        ):
            return "gpu_budget"
        if (
            cpu_capacity_bytes is not None
            and self.reserved_cpu_bytes + cpu_bytes > cpu_capacity_bytes
        ):
            return "cpu_budget"
        self.reserved_gpu_bytes += gpu_bytes
        self.reserved_cpu_bytes += cpu_bytes
        self.reserved_kv_cache_bytes += kv_bytes
        self.reserved_activation_bytes += activation_bytes
        self.peak_gpu_bytes = max(self.peak_gpu_bytes, self.reserved_gpu_bytes)
        self.peak_cpu_bytes = max(self.peak_cpu_bytes, self.reserved_cpu_bytes)
        self.peak_kv_cache_bytes = max(self.peak_kv_cache_bytes, self.reserved_kv_cache_bytes)
        self.peak_activation_bytes = max(
            self.peak_activation_bytes,
            self.reserved_activation_bytes,
        )
        return None

    def release(
        self,
        *,
        kv_bytes: int,
        activation_bytes: int,
        gpu_bytes: int,
        cpu_bytes: int,
    ) -> None:
        self.reserved_gpu_bytes = max(self.reserved_gpu_bytes - gpu_bytes, 0)
        self.reserved_cpu_bytes = max(self.reserved_cpu_bytes - cpu_bytes, 0)
        self.reserved_kv_cache_bytes = max(self.reserved_kv_cache_bytes - kv_bytes, 0)
        self.reserved_activation_bytes = max(
            self.reserved_activation_bytes - activation_bytes,
            0,
        )


def bucket_length(length: int, multiple: int) -> int:
    """Round a positive length up to the next scheduler bucket."""

    if length <= 0:
        return 0
    if multiple <= 1:
        return length
    return int(math.ceil(length / multiple) * multiple)


def chunk_sessions_by_decode_budget(
    sessions: list[T],
    *,
    total_tokens_per_session: int,
    max_batch_tokens: int | None,
) -> list[list[T]]:
    """Split a decode-compatible session group by a simple token budget."""

    if max_batch_tokens is None or len(sessions) <= 1:
        return [sessions]
    max_sessions = max(max_batch_tokens // max(total_tokens_per_session, 1), 1)
    if len(sessions) <= max_sessions:
        return [sessions]
    return [
        sessions[start : start + max_sessions] for start in range(0, len(sessions), max_sessions)
    ]


def estimate_admission(
    *,
    queue: str,
    wrapped: torch.nn.Module,
    plan: CompiledPlan,
    dtype: torch.dtype,
    batch_size: int,
    prompt_tokens: int,
    projected_decode_tokens: int,
    bucket_multiple: int,
    max_kv_cache_bytes: int | None,
    max_activation_capture_bytes: int | None,
) -> AdmissionEstimate:
    """Produce a conservative request-size estimate for early rejection."""

    total_tokens = prompt_tokens + projected_decode_tokens
    bucket_tokens = bucket_length(total_tokens, bucket_multiple)
    kv_cache_bytes = estimate_kv_cache_bytes(
        wrapped,
        dtype=dtype,
        batch_size=batch_size,
        total_tokens=bucket_tokens,
    )
    activation_bytes = estimate_activation_bytes(
        wrapped,
        plan=plan,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=max(prompt_tokens, 1),
    )

    admitted = True
    reason: str | None = None
    if max_kv_cache_bytes is not None and kv_cache_bytes > max_kv_cache_bytes:
        admitted = False
        reason = "kv_cache_budget"
    elif (
        max_activation_capture_bytes is not None and activation_bytes > max_activation_capture_bytes
    ):
        admitted = False
        reason = "activation_budget"

    return AdmissionEstimate(
        queue=queue,
        prompt_tokens=prompt_tokens,
        projected_decode_tokens=projected_decode_tokens,
        batch_size=batch_size,
        bucket_tokens=bucket_tokens,
        kv_cache_bytes=kv_cache_bytes,
        activation_bytes=activation_bytes,
        admitted=admitted,
        reason=reason,
    )


def estimate_kv_cache_bytes(
    wrapped: torch.nn.Module,
    *,
    dtype: torch.dtype,
    batch_size: int,
    total_tokens: int,
) -> int:
    """Estimate KV-cache footprint for one causal-LM decode workload."""

    config = _text_config(getattr(wrapped, "config", None))
    if config is None:
        return 0
    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    kv_heads = int(
        getattr(
            config,
            "num_key_value_heads",
            getattr(config, "num_attention_heads", getattr(config, "n_head", 0)),
        )
    )
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    n_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", kv_heads or 1)))
    head_dim = int(getattr(config, "head_dim", hidden // max(n_heads, 1))) if hidden else 0
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0 or total_tokens <= 0:
        return 0
    element_size = torch.empty((), dtype=dtype).element_size()
    return batch_size * total_tokens * layers * kv_heads * head_dim * 2 * element_size


def estimate_activation_bytes(
    wrapped: torch.nn.Module,
    *,
    plan: CompiledPlan,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
) -> int:
    """Estimate requested activation payload size for a plan."""

    if not plan.output.activations or not plan.get_paths:
        return 0
    config = _text_config(getattr(wrapped, "config", None))
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0))) if config else 0
    if hidden <= 0:
        return 0
    element_size = torch.empty((), dtype=dtype).element_size()
    return batch_size * seq_len * hidden * max(len(plan.get_paths), 1) * element_size


def _text_config(config: Any) -> Any | None:
    if config is None:
        return None
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        return get_text(decoder=True)
    return config


T = TypeVar("T")
