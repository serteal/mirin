"""Dataset-style collection helpers built on top of local ``mirin.Model``."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .model import Model
from .output import _release_value, _resolve_value
from .requests import normalize_request_row, request_items


@dataclass(slots=True)
class CollectStep:
    """One collected step plus the normalized batch that produced it."""

    activations: dict[str, Any]
    batch: dict[str, Any]
    rows: list[dict[str, torch.Tensor]]
    indices: list[int]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return _resolve_value(self.activations[key])
        path = getattr(key, "path", None) or getattr(key, "_path", None)
        if isinstance(path, str):
            return _resolve_value(self.activations[path])
        return _resolve_value(self.activations[key])

    def release(self) -> None:
        _release_value(self.activations)


@dataclass(slots=True)
class CollectExport:
    """Manifest returned by ``model.collect(..., out=PATH)``."""

    root: str
    files: dict[str, list[str]]
    rows: int
    format: str = "mmap_v1"


def normalize_collect_out(out: str | os.PathLike[str] | None) -> tuple[str, str | None]:
    """Map the public ``out=`` value to runtime collector settings."""

    if out is None or out == "gpu":
        return "gpu", None
    if out == "cpu":
        return "cpu", None
    if isinstance(out, (str, os.PathLike)):
        return "mmap", os.fspath(out)
    raise ValueError("collect out= must be None, 'gpu', 'cpu', or a filesystem path.")


def is_batched_mapping(value: Any) -> bool:
    """Return ``True`` when *value* looks like a padded batch mapping."""

    if not isinstance(value, Mapping):
        return False
    input_ids = value.get("input_ids")
    return (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim == 2
        and int(input_ids.shape[0]) > 1
    )


def resolve_layer_sites(
    model: Model,
    layers: Sequence[int] | int,
    *,
    hook_point: str = "block",
) -> list[Any]:
    """Resolve layer indices into mirin proxies for common LLM hook points."""

    layer_ids = [layers] if isinstance(layers, int) else list(layers)
    if hook_point == "block":
        return [model.layers[layer] for layer in layer_ids]
    if hook_point == "layernorm":
        sites: list[Any] = []
        for layer in layer_ids:
            block = model.layers[layer]
            site = getattr(block, "input_layernorm", None)
            if site is None:
                raise ValueError(
                    "hook_point='layernorm' requires `input_layernorm` on every selected layer."
                )
            sites.append(site)
        return sites
    raise ValueError("hook_point must be 'block' or 'layernorm'.")


def stream_collect(
    model: Model,
    requests: Sequence[Any] | Any,
    *,
    get: Sequence[Any] | Any | None = None,
    map: Mapping[Any, Any] | None = None,
    batch_size: int = 32,
    batch_token_budget: int | None = None,
    sort_by_length: bool = True,
    stop_at_last_get: bool = True,
    add_generation_prompt: bool = False,
    **kwargs: Any,
) -> Iterator[CollectStep]:
    """Collect request rows in bounded batches using the local model API."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = normalize_collect_requests(
        model,
        requests,
        add_generation_prompt=add_generation_prompt,
    )
    order = list(range(len(rows)))
    if sort_by_length:
        order.sort(key=lambda idx: _request_length(rows[idx]), reverse=True)
    cursor = 0
    while cursor < len(order):
        start = cursor
        current_max = 0
        while cursor < len(order) and (cursor - start) < batch_size:
            row = rows[order[cursor]]
            seq_len = _request_length(row)
            next_max = max(current_max, seq_len)
            next_count = (cursor - start) + 1
            if (
                next_count > 1
                and batch_token_budget is not None
                and next_max * next_count > batch_token_budget
            ):
                break
            current_max = next_max
            cursor += 1
        if cursor == start:
            cursor += 1
        batch_indices = order[start:cursor]
        batch_rows = [rows[idx] for idx in batch_indices]
        iterator = model.collect(
            batch_rows,
            get=get,
            map=dict(map) if map is not None else None,
            process=lambda step: step,
            max_items=batch_size,
            max_tokens=batch_token_budget,
            sort=False,
            stop_at_last_get=stop_at_last_get,
            **kwargs,
        )
        for step in iterator:
            if not isinstance(step, CollectStep):
                raise TypeError("model.collect(..., process=...) must yield CollectStep values.")
            yield CollectStep(
                activations=step.activations,
                batch=step.batch,
                rows=step.rows,
                indices=[batch_indices[idx] for idx in step.indices],
            )


def normalize_collect_requests(
    model: Model,
    requests: Sequence[Any] | Any,
    *,
    add_generation_prompt: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Normalize one or more request items into token rows for collection."""

    items = request_items(requests)
    if items is None:
        raise TypeError("Expected one request or a sequence of requests.")
    if not items:
        raise ValueError("Expected at least one request.")
    tokenizer = getattr(model, "tokenizer", None)
    owner = type(model).__name__
    return [
        normalize_request_row(
            request,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            owner=f"{owner}.collect",
        )
        for request in items
    ]


def _request_length(row: Mapping[str, torch.Tensor]) -> int:
    attention_mask = row.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask.sum().item())
    input_ids = row.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    return 0

