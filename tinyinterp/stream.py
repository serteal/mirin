"""Dataset-scale streaming helpers for tinyinterp."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch

from .output import Output


def stream_model(
    model: Any,
    dataloader: Iterable[Any],
    *,
    get: Sequence[Any] | Any | None = None,
    map_dict: dict[Any, Any] | None = None,
    grad: bool = False,
    batch_size: int | None = None,
    to_cpu: bool = True,
    non_blocking: bool = True,
) -> Iterator[Any]:
    """Iterate over a dataset and yield per-batch tinyinterp outputs."""

    for batch in dataloader:
        for chunk in _iter_chunks(batch, batch_size):
            output = _call_model(model, chunk, get=get, map_dict=map_dict, grad=grad)
            if to_cpu:
                yield _move_to_cpu(output, non_blocking=non_blocking)
            else:
                yield output


def _call_model(
    model: Any,
    batch: Any,
    *,
    get: Sequence[Any] | Any | None,
    map_dict: dict[Any, Any] | None,
    grad: bool,
) -> Any:
    if isinstance(batch, dict):
        return model(**batch, get=get, map=map_dict, grad=grad)
    if isinstance(batch, tuple):
        return model(*batch, get=get, map=map_dict, grad=grad)
    if isinstance(batch, list):
        return model(*batch, get=get, map=map_dict, grad=grad)
    return model(batch, get=get, map=map_dict, grad=grad)


def _iter_chunks(batch: Any, batch_size: int | None) -> Iterator[Any]:
    if batch_size is None:
        yield batch
        return
    total = _infer_batch_size(batch)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        yield _slice_batch(batch, start, end)


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            return _infer_batch_size(value)
    if isinstance(batch, (tuple, list)):
        for value in batch:
            return _infer_batch_size(value)
    raise TypeError("Could not infer batch size from stream input.")


def _slice_batch(batch: Any, start: int, end: int) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch[start:end]
    if isinstance(batch, dict):
        return {key: _slice_batch(value, start, end) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_slice_batch(value, start, end) for value in batch)
    if isinstance(batch, list):
        return [_slice_batch(value, start, end) for value in batch]
    raise TypeError(f"Unsupported stream batch type: {type(batch).__name__}")


def _move_to_cpu(value: Any, *, non_blocking: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", non_blocking=non_blocking)
    if isinstance(value, Output):
        model_output = _move_to_cpu(value._model_output, non_blocking=non_blocking)
        activations = {
            sid: _move_to_cpu(tensor, non_blocking=non_blocking)
            for sid, tensor in value.activations.items()
        }
        return Output(model_output, activations, value._id_to_sid)
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [_move_to_cpu(item, non_blocking=non_blocking) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_cpu(item, non_blocking=non_blocking) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        clone = copy.copy(value)
        for name, item in vars(value).items():
            setattr(clone, name, _move_to_cpu(item, non_blocking=non_blocking))
        return clone
    return value
