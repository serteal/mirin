"""Memory budget detection and auto-chunking for the inference server."""

from __future__ import annotations

from typing import Any

import torch

from .runtime import model_dtype
from .scheduler import estimate_activation_bytes, estimate_kv_cache_bytes

# --- real memory detection ---


def _cpu_memory_bytes() -> int:
    """Detect actual CPU memory limit, respecting cgroups."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as fh:
            v = int(fh.read().strip())
        if v < (1 << 62):
            return v
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    # fallback: /proc/meminfo
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError):
        pass
    return 0


def _gpu_memory_bytes(device: torch.device) -> tuple[int, int]:
    """Return (free, total) GPU memory in bytes."""
    if device.type != "cuda":
        return (0, 0)
    return torch.cuda.mem_get_info(device)


# --- budget ---


class MemoryBudget:
    """Track GPU and CPU memory budgets for safe auto-batching."""

    __slots__ = ("gpu_budget", "cpu_budget", "device", "_wrapped", "_per_token_gpu")

    def __init__(
        self,
        wrapped: torch.nn.Module,
        device: torch.device,
        *,
        gpu_fraction: float = 0.9,
        cpu_fraction: float = 0.8,
    ) -> None:
        self._wrapped = wrapped
        self.device = device
        self._per_token_gpu: int | None = None

        # GPU budget: fraction of free memory after model load
        gpu_free, gpu_total = _gpu_memory_bytes(device)
        self.gpu_budget = int(gpu_free * gpu_fraction) if gpu_free > 0 else 0

        # CPU budget: fraction of real memory limit
        cpu_total = _cpu_memory_bytes()
        self.cpu_budget = int(cpu_total * cpu_fraction) if cpu_total > 0 else 0

    def calibrate(self, seq_len: int = 32) -> None:
        """Run a calibration forward to measure actual per-token GPU cost."""
        if self.device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        dummy = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self._wrapped(input_ids=dummy, use_cache=False)
        peak = torch.cuda.max_memory_allocated(self.device)
        del dummy
        torch.cuda.empty_cache()
        self._per_token_gpu = max((peak - baseline) // max(seq_len, 1), 1)

    def max_batch_size(self, plan: Any, seq_len: int) -> int:
        """Max batch size that fits in GPU memory for this plan and seq_len."""
        if self.gpu_budget <= 0:
            return 1
        dtype = model_dtype(self._wrapped)
        # Use calibrated per-token cost if available
        if self._per_token_gpu is not None:
            gpu_per_example = self._per_token_gpu * seq_len
        else:
            # Formula estimate: KV cache + activations for batch=1
            kv = estimate_kv_cache_bytes(
                self._wrapped, dtype=dtype, batch_size=1, total_tokens=seq_len
            )
            act = estimate_activation_bytes(
                self._wrapped, plan=plan, dtype=dtype, batch_size=1, seq_len=seq_len
            )
            gpu_per_example = max(kv + act, 1)
        return max(self.gpu_budget // gpu_per_example, 1)

    def estimate_cpu_bytes(self, plan: Any, batch_size: int, seq_len: int) -> int:
        """Estimate CPU bytes for captured activations."""
        dtype = model_dtype(self._wrapped)
        return estimate_activation_bytes(
            self._wrapped,
            plan=plan,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=seq_len,
        )


# --- auto-chunker ---


def auto_chunk(
    input_ids: torch.Tensor,
    max_batch: int,
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Split a batch into chunks that fit within max_batch."""
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
            for k, v in extra_tensors.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                    chunk[k] = v[start:end]
                else:
                    chunk[k] = v
        chunks.append(chunk)
    return chunks
