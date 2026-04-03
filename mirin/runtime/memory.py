"""Runtime capacity detection, calibration, and auto-chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .scheduler import bucket_length, estimate_activation_bytes, estimate_kv_cache_bytes
from .util import filter_supported_kwargs, model_dtype


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip()
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _proc_meminfo_bytes(field: str) -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return 0
    return 0


def _cpu_memory_bytes() -> int:
    """Detect the effective CPU memory limit, respecting cgroups."""

    limit = _read_int("/sys/fs/cgroup/memory.max")
    if limit is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None:
        return limit
    total = _proc_meminfo_bytes("MemTotal")
    return total if total > 0 else 0


def _cpu_available_bytes() -> int:
    """Detect the currently available CPU memory inside the container."""

    limit = _read_int("/sys/fs/cgroup/memory.max")
    usage = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        usage = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    available = _proc_meminfo_bytes("MemAvailable")
    if limit is not None and usage is not None:
        available = min(available or limit, max(limit - usage, 0))
    if available > 0:
        return available
    total = _cpu_memory_bytes()
    return total if total > 0 else 0


def _gpu_memory_bytes(device: torch.device) -> tuple[int, int]:
    """Return ``(free, total)`` GPU memory in bytes."""

    if device.type != "cuda":
        return (0, 0)
    return torch.cuda.mem_get_info(device)


def _model_memory_bytes(wrapped: torch.nn.Module, device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    total = 0
    for tensor in list(wrapped.parameters()) + list(wrapped.buffers()):
        total += int(tensor.numel() * tensor.element_size())
    return total


def _text_config(config: Any) -> Any | None:
    if config is None:
        return None
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        return get_text(decoder=True)
    return config


def _fallback_forward_bytes_per_token(wrapped: torch.nn.Module) -> int:
    config = _text_config(getattr(wrapped, "config", None))
    if config is None:
        return 0
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    if hidden <= 0 or layers <= 0:
        return 0
    element_size = torch.empty((), dtype=model_dtype(wrapped)).element_size()
    return hidden * layers * element_size * 4


def _measure_forward_bytes_per_token(
    wrapped: torch.nn.Module,
    *,
    device: torch.device,
    seq_len: int = 32,
) -> int:
    if device.type != "cuda":
        return _fallback_forward_bytes_per_token(wrapped)

    before = int(torch.cuda.memory_allocated(device))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(dummy)
    kwargs = filter_supported_kwargs(
        wrapped,
        {
            "input_ids": dummy,
            "attention_mask": attention_mask,
            "use_cache": False,
        },
    )
    try:
        with torch.inference_mode():
            wrapped(**kwargs)
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        measured = peak - before
        if measured > 0:
            return max(measured // max(seq_len, 1), 1)
    except Exception:
        pass
    finally:
        del dummy
        del attention_mask
        torch.cuda.empty_cache()
    return _fallback_forward_bytes_per_token(wrapped)


def _capacity_as_tokens(capacity_bytes: int, per_token_bytes: int) -> int | None:
    if capacity_bytes <= 0 or per_token_bytes <= 0:
        return None
    return max(capacity_bytes // per_token_bytes, 1)


@dataclass(slots=True)
class RuntimeCapacity:
    """Startup-derived capacity model for the local runtime."""

    wrapped: torch.nn.Module
    device: torch.device
    gpu_fraction: float
    cpu_fraction: float
    gpu_free_bytes: int
    gpu_total_bytes: int
    cpu_total_bytes: int
    cpu_available_bytes: int
    gpu_capacity_bytes: int
    cpu_capacity_bytes: int
    device_capacity_bytes: int
    model_bytes: int
    kv_bytes_per_token: int
    forward_bytes_per_token: int
    kv_cache_bytes: int | None
    activation_capture_bytes: int | None
    spill_bytes: int | None
    max_prefill_tokens: int | None
    max_decode_batch_tokens: int | None
    collect_token_budget: int | None
    max_running_requests: int | None

    @classmethod
    def detect(
        cls,
        wrapped: torch.nn.Module,
        device: torch.device,
        *,
        gpu_fraction: float,
        cpu_fraction: float,
        max_kv_cache_bytes: int | None,
        max_activation_capture_bytes: int | None,
        prefill_token_budget: int | None,
        decode_max_batch_tokens: int | None,
        collect_token_budget: int | None,
    ) -> RuntimeCapacity:
        if not 0.0 < gpu_fraction <= 1.0:
            raise ValueError("gpu_fraction must be in (0, 1].")
        if not 0.0 < cpu_fraction <= 1.0:
            raise ValueError("cpu_fraction must be in (0, 1].")

        gpu_free_bytes, gpu_total_bytes = _gpu_memory_bytes(device)
        cpu_total_bytes = _cpu_memory_bytes()
        cpu_available_bytes = _cpu_available_bytes()
        gpu_capacity_bytes = int(gpu_free_bytes * gpu_fraction) if gpu_free_bytes > 0 else 0
        cpu_capacity_bytes = int(cpu_available_bytes * cpu_fraction) if cpu_available_bytes > 0 else 0
        device_capacity_bytes = (
            gpu_capacity_bytes if device.type == "cuda" else cpu_capacity_bytes
        )
        model_bytes = _model_memory_bytes(wrapped, device)
        dtype = model_dtype(wrapped)
        kv_bytes_per_token = estimate_kv_cache_bytes(
            wrapped,
            dtype=dtype,
            batch_size=1,
            total_tokens=1,
        )
        forward_bytes_per_token = _measure_forward_bytes_per_token(
            wrapped,
            device=device,
        )
        default_cap = device_capacity_bytes if device_capacity_bytes > 0 else None
        kv_cache_bytes = max_kv_cache_bytes if max_kv_cache_bytes is not None else default_cap
        activation_capture_bytes = (
            max_activation_capture_bytes
            if max_activation_capture_bytes is not None
            else default_cap
        )
        spill_bytes = cpu_capacity_bytes if cpu_capacity_bytes > 0 else None
        max_prefill_tokens = (
            prefill_token_budget
            if prefill_token_budget is not None
            else _capacity_as_tokens(device_capacity_bytes, forward_bytes_per_token)
        )
        max_decode_batch_tokens = (
            decode_max_batch_tokens
            if decode_max_batch_tokens is not None
            else _capacity_as_tokens(device_capacity_bytes, kv_bytes_per_token)
        )
        collect_budget = (
            collect_token_budget if collect_token_budget is not None else max_prefill_tokens
        )
        max_running_requests = (
            None
            if max_decode_batch_tokens is None
            else max(max_decode_batch_tokens, 1)
        )
        capacity = cls(
            wrapped=wrapped,
            device=device,
            gpu_fraction=gpu_fraction,
            cpu_fraction=cpu_fraction,
            gpu_free_bytes=gpu_free_bytes,
            gpu_total_bytes=gpu_total_bytes,
            cpu_total_bytes=cpu_total_bytes,
            cpu_available_bytes=cpu_available_bytes,
            gpu_capacity_bytes=gpu_capacity_bytes,
            cpu_capacity_bytes=cpu_capacity_bytes,
            device_capacity_bytes=device_capacity_bytes,
            model_bytes=model_bytes,
            kv_bytes_per_token=kv_bytes_per_token,
            forward_bytes_per_token=forward_bytes_per_token,
            kv_cache_bytes=kv_cache_bytes,
            activation_capture_bytes=activation_capture_bytes,
            spill_bytes=spill_bytes,
            max_prefill_tokens=max_prefill_tokens,
            max_decode_batch_tokens=max_decode_batch_tokens,
            collect_token_budget=collect_budget,
            max_running_requests=max_running_requests,
        )
        capacity.validate()
        return capacity

    @property
    def gpu_budget(self) -> int:
        return self.gpu_capacity_bytes

    @gpu_budget.setter
    def gpu_budget(self, value: int) -> None:
        self.gpu_capacity_bytes = max(int(value), 0)
        if self.device.type == "cuda":
            self.device_capacity_bytes = self.gpu_capacity_bytes

    @property
    def cpu_budget(self) -> int:
        return self.cpu_capacity_bytes

    @cpu_budget.setter
    def cpu_budget(self, value: int) -> None:
        self.cpu_capacity_bytes = max(int(value), 0)
        if self.device.type != "cuda":
            self.device_capacity_bytes = self.cpu_capacity_bytes
        self.spill_bytes = self.cpu_capacity_bytes if self.cpu_capacity_bytes > 0 else None

    def validate(self) -> None:
        if self.device.type == "cuda" and self.gpu_free_bytes > 0 and self.gpu_capacity_bytes <= 0:
            raise RuntimeError("mirin runtime GPU capacity is zero after applying gpu_fraction.")
        if self.device_capacity_bytes <= 0 and self.device.type != "meta":
            raise RuntimeError(
                f"mirin runtime could not derive a positive {self.device.type} capacity."
            )
        for name, value in (
            ("max_kv_cache_bytes", self.kv_cache_bytes),
            ("max_activation_capture_bytes", self.activation_capture_bytes),
        ):
            if value is None:
                continue
            if value <= 0:
                raise RuntimeError(f"{name} must be positive once capacity is derived.")
            if self.device_capacity_bytes > 0 and value > self.device_capacity_bytes:
                raise RuntimeError(
                    f"{name} ({value}) exceeds the derived {self.device.type} capacity "
                    f"({self.device_capacity_bytes})."
                )
        if self.kv_bytes_per_token > 0 and self.kv_cache_bytes is not None:
            if self.kv_cache_bytes < self.kv_bytes_per_token:
                raise RuntimeError(
                    "mirin runtime KV capacity is too small to hold a single cached token."
                )
        if self.max_prefill_tokens is not None and self.max_prefill_tokens < 1:
            raise RuntimeError("mirin runtime prefill token budget must be at least 1.")
        if self.max_decode_batch_tokens is not None and self.max_decode_batch_tokens < 1:
            raise RuntimeError("mirin runtime decode token budget must be at least 1.")
        if self.collect_token_budget is not None and self.collect_token_budget < 1:
            raise RuntimeError("mirin runtime collect token budget must be at least 1.")

    def snapshot(self) -> dict[str, int | str | None]:
        return {
            "device_type": self.device.type,
            "model_bytes": self.model_bytes,
            "gpu_free_bytes": self.gpu_free_bytes or None,
            "gpu_total_bytes": self.gpu_total_bytes or None,
            "gpu_capacity_bytes": self.gpu_capacity_bytes or None,
            "cpu_total_bytes": self.cpu_total_bytes or None,
            "cpu_available_bytes": self.cpu_available_bytes or None,
            "cpu_capacity_bytes": self.cpu_capacity_bytes or None,
            "device_capacity_bytes": self.device_capacity_bytes or None,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activation_capture_bytes": self.activation_capture_bytes,
            "spill_bytes": self.spill_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token or None,
            "forward_bytes_per_token": self.forward_bytes_per_token or None,
            "max_prefill_tokens": self.max_prefill_tokens,
            "max_decode_batch_tokens": self.max_decode_batch_tokens,
            "collect_token_budget": self.collect_token_budget,
            "max_running_requests": self.max_running_requests,
        }

    def max_batch_size(
        self,
        plan: Any,
        seq_len: int,
        *,
        bucket_multiple: int = 64,
    ) -> int:
        """Max batch size that fits within the derived device capacity."""

        if self.device_capacity_bytes <= 0:
            return 1
        dtype = model_dtype(self.wrapped)
        forward_bytes = self.forward_bytes_per_token * max(seq_len, 1)
        bucket_tokens = bucket_length(max(seq_len, 1), bucket_multiple)
        capture_bytes = estimate_activation_bytes(
            self.wrapped,
            plan=plan,
            dtype=dtype,
            batch_size=1,
            seq_len=max(seq_len, 1),
        )
        kv_bytes = estimate_kv_cache_bytes(
            self.wrapped,
            dtype=dtype,
            batch_size=1,
            total_tokens=bucket_tokens,
        )
        per_example = max(forward_bytes, capture_bytes + kv_bytes, 1)
        return max(self.device_capacity_bytes // per_example, 1)

    def estimate_cpu_bytes(self, plan: Any, batch_size: int, seq_len: int) -> int:
        """Estimate host bytes for a captured activation payload."""

        dtype = model_dtype(self.wrapped)
        return estimate_activation_bytes(
            self.wrapped,
            plan=plan,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=seq_len,
        )


def auto_chunk(
    input_ids: torch.Tensor,
    max_batch: int,
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Split a batch into chunks that fit within ``max_batch`` rows."""

    batch_size = input_ids.shape[0]
    if batch_size <= max_batch:
        chunk: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if extra_tensors:
            chunk.update(extra_tensors)
        return [chunk]
    chunks: list[dict[str, torch.Tensor]] = []
    for start in range(0, batch_size, max_batch):
        end = min(start + max_batch, batch_size)
        chunk = {"input_ids": input_ids[start:end]}
        if extra_tensors:
            for key, value in extra_tensors.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    chunk[key] = value[start:end]
                else:
                    chunk[key] = value
        chunks.append(chunk)
    return chunks
