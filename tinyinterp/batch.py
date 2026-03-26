"""Opt-in batching context for repeated tinyinterp calls."""

from __future__ import annotations

import copy
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .context import get_debug
from .counters import Counters
from .hooks import MapFn
from .output import Output

_MISSING = object()
_ACTIVE_BATCH: ContextVar[_BatchPlanner | None] = ContextVar("active_batch", default=None)


class batch(ContextDecorator):
    """Accumulate compatible model calls and fuse them on context exit."""

    def __init__(self) -> None:
        self._planner = _BatchPlanner()
        self._token: Token[_BatchPlanner | None] | None = None

    def __enter__(self) -> batch:
        self._token = _ACTIVE_BATCH.set(self._planner)
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        try:
            if exc_info[0] is None:
                self._planner.flush_all()
        finally:
            if self._token is not None:
                _ACTIVE_BATCH.reset(self._token)
                self._token = None
        return False


def maybe_enqueue_call(
    model: Any,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    get_proxies: list[Any],
    map_proxies: dict[Any, MapFn],
    grad: bool,
) -> _DeferredResult | None:
    """Queue a model call when a batch context is active."""

    planner = _ACTIVE_BATCH.get()
    if planner is None or grad:
        return None
    return planner.enqueue(
        _QueuedCall(
            model=model,
            args=args,
            kwargs=kwargs,
            get_proxies=get_proxies,
            map_proxies=map_proxies,
            future=_DeferredResult(planner),
        )
    )


class _DeferredResult:
    """Lazy placeholder returned for queued batch calls."""

    __slots__ = ("_planner", "_value")

    def __init__(self, planner: _BatchPlanner) -> None:
        self._planner = planner
        self._value: Any = _MISSING

    def resolve(self) -> Any:
        if self._value is _MISSING:
            self._planner.flush_all()
        return self._value

    def _set_value(self, value: Any) -> None:
        self._value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)

    def __getitem__(self, key: Any) -> Any:
        return self.resolve()[key]

    def __iter__(self) -> Any:
        return iter(self.resolve())

    def __len__(self) -> int:
        return len(self.resolve())


@dataclass(slots=True)
class _QueuedCall:
    model: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    get_proxies: list[Any]
    map_proxies: dict[Any, MapFn]
    future: _DeferredResult


class _BatchPlanner:
    """Collect queued calls and flush them as fused forward passes."""

    __slots__ = ("_queue", "_flushing")

    def __init__(self) -> None:
        self._queue: list[_QueuedCall] = []
        self._flushing = False

    def enqueue(self, call: _QueuedCall) -> _DeferredResult:
        self._queue.append(call)
        return call.future

    def flush_all(self) -> None:
        if self._flushing or not self._queue:
            return
        self._flushing = True
        try:
            queue = self._queue
            self._queue = []
            idx = 0
            while idx < len(queue):
                group = [queue[idx]]
                idx += 1
                while idx < len(queue) and _calls_are_compatible(group[0], queue[idx]):
                    group.append(queue[idx])
                    idx += 1
                self._execute_group(group)
        finally:
            self._flushing = False

    def _execute_group(self, group: list[_QueuedCall]) -> None:
        Counters.batch_groups += 1
        Counters.batch_fusions += max(0, len(group) - 1)
        if get_debug() >= 3:
            model_name = type(group[0].model.wrapped).__name__
            paths = ", ".join(proxy.path for proxy in group[0].map_proxies)
            print(
                f"[ti] batch: size={len(group)} model={model_name} "
                f"maps=[{paths}] get={len(group[0].get_proxies)}"
            )

        if len(group) == 1:
            call = group[0]
            result = call.model._execute_now(
                args=call.args,
                kwargs=call.kwargs,
                get_proxies=call.get_proxies,
                map_proxies=call.map_proxies,
                grad=False,
                requested_calls=1,
            )
            call.future._set_value(result)
            return

        batch_sizes = [_infer_batch_size(call.args, call.kwargs) for call in group]
        fused_args = _stack_values([call.args for call in group])
        fused_kwargs = _stack_values([call.kwargs for call in group])
        fused_map = _fuse_map_fns(group, batch_sizes)
        result = group[0].model._execute_now(
            args=fused_args,
            kwargs=fused_kwargs,
            get_proxies=group[0].get_proxies,
            map_proxies=fused_map,
            grad=False,
            requested_calls=len(group),
        )
        split_results = _split_result(result, batch_sizes)
        for call, split in zip(group, split_results, strict=True):
            call.future._set_value(split)


def _calls_are_compatible(left: _QueuedCall, right: _QueuedCall) -> bool:
    if left.model is not right.model:
        return False
    if [proxy.path for proxy in left.get_proxies] != [proxy.path for proxy in right.get_proxies]:
        return False
    left_map_paths = sorted(proxy.path for proxy in left.map_proxies)
    right_map_paths = sorted(proxy.path for proxy in right.map_proxies)
    if left_map_paths != right_map_paths:
        return False
    return _can_stack_values(left.args, right.args) and _can_stack_values(left.kwargs, right.kwargs)


def _can_stack_values(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dim() > 0
            and right.dim() > 0
            and left.dtype == right.dtype
            and left.device == right.device
            and left.shape[1:] == right.shape[1:]
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_can_stack_values(left[key], right[key]) for key in left)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _can_stack_values(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _can_stack_values(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _stack_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {key: _stack_values([value[key] for value in values]) for key in first}
    if isinstance(first, tuple):
        return tuple(_stack_values([value[idx] for value in values]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_stack_values([value[idx] for value in values]) for idx in range(len(first))]
    return first


def _fuse_map_fns(group: list[_QueuedCall], batch_sizes: list[int]) -> dict[Any, MapFn]:
    fused: dict[Any, MapFn] = {}
    proxies = list(group[0].map_proxies)
    for proxy in proxies:
        entries = [
            (size, call.map_proxies[proxy]) for size, call in zip(batch_sizes, group, strict=True)
        ]

        def batched_map(
            x: torch.Tensor,
            _entries: list[tuple[int, MapFn]] = entries,
        ) -> torch.Tensor:
            chunks: list[torch.Tensor] = []
            start = 0
            for size, fn in _entries:
                end = start + size
                chunks.append(fn(x[start:end]))
                start = end
            return torch.cat(chunks, dim=0)

        fused[proxy] = batched_map
    return fused


def _infer_batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    for value in list(args) + list(kwargs.values()):
        size = _batch_size_from_value(value)
        if size is not None:
            return size
    raise TypeError("Could not infer batch dimension for batched tinyinterp call.")


def _batch_size_from_value(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return int(value.shape[0])
    if isinstance(value, dict):
        for item in value.values():
            size = _batch_size_from_value(item)
            if size is not None:
                return size
    if isinstance(value, (tuple, list)):
        for item in value:
            size = _batch_size_from_value(item)
            if size is not None:
                return size
    return None


def _split_result(result: Any, batch_sizes: list[int]) -> list[Any]:
    if isinstance(result, Output):
        model_outputs = _split_value(result._model_output, batch_sizes)
        activation_slices = {
            sid: _split_value(tensor, batch_sizes) for sid, tensor in result.activations.items()
        }
        outputs: list[Output] = []
        for idx, model_output in enumerate(model_outputs):
            activations = {sid: chunks[idx] for sid, chunks in activation_slices.items()}
            outputs.append(Output(model_output, activations, result._id_to_sid))
        return outputs
    return _split_value(result, batch_sizes)


def _split_value(value: Any, batch_sizes: list[int]) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return list(torch.split(value, batch_sizes, dim=0))
    if isinstance(value, tuple):
        parts = [_split_value(item, batch_sizes) for item in value]
        return [tuple(part[idx] for part in parts) for idx in range(len(batch_sizes))]
    if isinstance(value, list):
        parts = [_split_value(item, batch_sizes) for item in value]
        return [[part[idx] for part in parts] for idx in range(len(batch_sizes))]
    if isinstance(value, dict):
        part_map = {key: _split_value(item, batch_sizes) for key, item in value.items()}
        return [
            {key: chunks[idx] for key, chunks in part_map.items()}
            for idx in range(len(batch_sizes))
        ]
    if hasattr(value, "__dict__"):
        outputs = [copy.copy(value) for _ in batch_sizes]
        for name, item in vars(value).items():
            slices = (
                _split_value(item, batch_sizes)
                if _is_batch_value(item)
                else [item] * len(batch_sizes)
            )
            for output, slice_value in zip(outputs, slices, strict=True):
                setattr(output, name, slice_value)
        return outputs
    return [value] * len(batch_sizes)


def _is_batch_value(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dim() > 0
    if isinstance(value, (tuple, list)):
        return any(_is_batch_value(item) for item in value)
    if isinstance(value, dict):
        return any(_is_batch_value(item) for item in value.values())
    return False
