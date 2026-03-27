"""Collector handle for batched activation extraction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .plans import CompiledPlan
from .results import PlanResult

if TYPE_CHECKING:
    from .inference import Server


@dataclass(slots=True)
class Collector:
    """Server-owned collector configuration."""

    id: str
    plan: CompiledPlan
    use_cache: bool
    stop_at_last_get: bool
    token_budget: int | None
    activation_budget_bytes: int | None
    activations_to_cpu: bool
    pin_memory: bool
    reducer: str | None
    token_index: int | None
    mmap_path: str | None
    server: Server
    mmap_state: dict[str, Any] = field(default_factory=dict)

    def collect_batch(self, batch: Mapping[str, Any]) -> PlanResult:
        return self.server.collect_batch(self, batch)

    def collect_many(self, requests: Iterable[Any]) -> list[PlanResult]:
        return self.server.collect_many(self, list(requests))

    def run(self, dataset: Iterable[Mapping[str, Any]]) -> Iterator[PlanResult]:
        for batch in dataset:
            for chunk in _split_batch(batch, self.token_budget):
                yield self.collect_batch(chunk)

    def close(self) -> None:
        self.server.close_collector(self)


def _split_batch(
    batch: Mapping[str, Any],
    token_budget: int | None,
) -> list[dict[str, Any]]:
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
