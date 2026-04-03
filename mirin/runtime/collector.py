"""Collector handle for batched activation extraction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .plans import CompiledPlan
from .results import PlanResult

if TYPE_CHECKING:
    from .core import _RuntimeCore


@dataclass(slots=True)
class Collector:
    """Runtime-owned collector configuration.

    ``token_budget`` applies to collector-side request splitting for dataset-like
    iteration. This is separate from the scheduler's prefill/decode admission
    budgets.
    """

    id: str
    plan: CompiledPlan
    use_cache: bool
    stop_at_last_get: bool
    token_budget: int | None
    activation_budget_bytes: int | None
    activation_output: Literal["gpu", "cpu", "mmap"]
    pin_memory: bool
    mmap_path: str | None
    runtime: _RuntimeCore
    mmap_state: dict[str, Any] = field(default_factory=dict)

    @property
    def activations_to_cpu(self) -> bool:
        return self.activation_output == "cpu"

    @property
    def uses_mmap(self) -> bool:
        return self.activation_output == "mmap"

    def collect_batch(self, batch: Mapping[str, Any]) -> PlanResult:
        return self.runtime.collect_batch(self, batch)

    def collect_many(self, requests: Iterable[Any], **kwargs: Any) -> list[PlanResult]:
        return self.runtime.collect_many(self, list(requests), **kwargs)

    def run(self, dataset: Iterable[Mapping[str, Any]]) -> Iterator[PlanResult]:
        for batch in dataset:
            for chunk in _split_batch(batch, self.token_budget):
                yield self.collect_batch(chunk)

    def close(self) -> None:
        self.runtime.close_collector(self)


@dataclass(frozen=True, slots=True)
class MmapSegment:
    filename: str
    shape: tuple[int, ...]
    dtype: str
    storage_dtype: str | None = None


@dataclass(slots=True)
class MmapActivationRef:
    """Lazy activation handle backed by one or more memmap segments."""

    path: str
    segments: tuple[MmapSegment, ...]

    def resolve(self) -> torch.Tensor:
        expected_dtype = _torch_dtype_from_name(self.segments[0].dtype)
        arrays = [
            np.asarray(
                np.memmap(
                    segment.filename,
                    mode="r",
                    dtype=np.dtype(segment.storage_dtype or segment.dtype),
                    shape=segment.shape,
                )
            )
            for segment in self.segments
        ]
        if not arrays:
            raise RuntimeError("mmap activation handle has no backing segments.")
        merged = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        tensor = torch.from_numpy(np.array(merged, copy=True))
        if tensor.dtype == expected_dtype:
            return tensor
        if tensor.dtype == torch.uint16 and expected_dtype == torch.bfloat16:
            return tensor.view(torch.bfloat16)
        return tensor.to(dtype=expected_dtype)

    def release(self) -> None:
        return None


def _split_batch(
    batch: Mapping[str, Any],
    token_budget: int | None,
) -> list[dict[str, Any]]:
    """Split one collector batch by a per-chunk token budget."""

    if token_budget is None:
        return [dict(batch)]
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        return [dict(batch)]
    batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
    max_batch = max(token_budget // max(seq_len, 1), 1)
    if batch_size <= max_batch:
        return [dict(batch)]
    outputs: list[dict[str, Any]] = []
    for start in range(0, batch_size, max_batch):
        end = min(start + max_batch, batch_size)
        chunk: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        outputs.append(chunk)
    return outputs


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    try:
        return mapping[name.removeprefix("torch.")]
    except KeyError as exc:
        raise ValueError(f"Unsupported mmap tensor dtype {name!r}.") from exc
