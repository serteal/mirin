"""Internal executors for local mirin models."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch

from . import maps as maps_mod
from .batch import batch_active, maybe_enqueue_call
from .collect_options import split_collector_kwargs
from .output import output_from_path_activations
from .requests import request_items

if TYPE_CHECKING:
    from .model import Model, _ModuleListProxy


class _CallableCacheKey:
    __slots__ = ("fn",)

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __hash__(self) -> int:
        return id(self.fn)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CallableCacheKey) and self.fn is other.fn


_CALLABLE_CACHE: dict[int, _CallableCacheKey] = {}
_CALLABLE_CACHE_LOCK = threading.Lock()


class _LocalExecutor:
    def __init__(self, model: Model) -> None:
        self._model = model
        self._runtime: Any | None = None
        self._plan_cache: dict[Any, Any] = {}
        self._collector_cache: dict[Any, Any] = {}

    def get_attr(self, name: str) -> Any:
        return getattr(self._model._root, name)

    def dir(self) -> list[str]:
        return sorted(set(list(object.__dir__(self._model)) + dir(self._model._root)))

    @property
    def layers(self) -> _ModuleListProxy:
        from .model import _layers_sort_key, _ModuleListProxy, _wrap_proxy

        if self._model._layers_proxy is None:
            best: tuple[str, torch.nn.ModuleList] | None = None
            for path, module in self._model.wrapped.named_modules():
                if not isinstance(module, torch.nn.ModuleList):
                    continue
                if best is None or _layers_sort_key(path, module) > _layers_sort_key(*best):
                    best = (path, module)
            if best is None:
                raise AttributeError("No ModuleList found. Navigate directly.")
            proxy = _wrap_proxy(best[1], best[0], self._model._hooks, self._model._renames)
            if not isinstance(proxy, _ModuleListProxy):
                raise TypeError("Internal error: expected layers proxy to be a ModuleList proxy.")
            self._model._layers_proxy = proxy
        return self._model._layers_proxy

    @property
    def device(self) -> torch.device | tuple[torch.device, ...]:
        devices = {tensor.device for tensor in self._model.wrapped.parameters()}
        devices.update(tensor.device for tensor in self._model.wrapped.buffers())
        if not devices:
            return torch.device("cpu")
        if len(devices) == 1:
            return next(iter(devices))
        return tuple(sorted(devices, key=str))

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "grad": True,
            "request_tokenization": self._model.tokenizer is not None,
        }

    @property
    def capacity(self) -> dict[str, Any]:
        return dict(self._shared_runtime().capacity.snapshot())

    def stats(self) -> dict[str, Any]:
        return dict(self._shared_runtime().stats())

    def call(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        get: Any,
        mapping: Any,
        grad: bool,
        stop_at_last_get: bool,
    ) -> Any:
        get_proxies = self._model._normalize_get(get)
        map_proxies = self._model._normalize_map(mapping)
        if stop_at_last_get:
            if not get_proxies:
                raise ValueError("stop_at_last_get=True requires at least one get= site.")
            if map_proxies:
                raise ValueError("stop_at_last_get=True does not support map=.")
            if grad:
                raise ValueError("stop_at_last_get=True does not support grad=True.")
            if batch_active():
                raise ValueError("stop_at_last_get=True is not supported inside ti.batch().")
        deferred = maybe_enqueue_call(
            self._model,
            args=args,
            kwargs=kwargs,
            get_proxies=get_proxies,
            map_proxies=map_proxies,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
        )
        if deferred is not None:
            return deferred
        return self._model._execute_now(
            args=args,
            kwargs=kwargs,
            get_proxies=get_proxies,
            map_proxies=map_proxies,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
            always_output=True,
        )

    def call_requests(
        self,
        *,
        requests: list[Any],
        kwargs: Mapping[str, Any],
        get: Any,
        mapping: Any,
        grad: bool,
        stop_at_last_get: bool,
    ) -> Any:
        if not requests:
            raise ValueError("Expected at least one request.")
        if grad:
            raise ValueError(
                "model(..., grad=True) does not support text/chat/batched request forms. "
                "Pass raw tensor inputs instead."
            )
        get_proxies = self._model._normalize_get(get)
        map_proxies = self._model._normalize_map(mapping)
        if stop_at_last_get:
            if not get_proxies:
                raise ValueError("stop_at_last_get=True requires at least one get= site.")
            if map_proxies:
                raise ValueError("stop_at_last_get=True does not support map=.")
            plan = self._compiled_plan(
                get_proxies,
                {},
                output={"logits": False, "activations": True},
            )
            collector = self._collector_for(plan, stop_at_last_get=True)
            results = self._shared_runtime().collect_many(
                collector,
                requests,
                **dict(kwargs),
            )
        else:
            plan = self._compiled_plan(get_proxies, map_proxies, output=None)
            results = self._shared_runtime().call_many(
                requests,
                plan=plan,
                **dict(kwargs),
            )
        outputs = [_plan_result_output(result) for result in results]
        return outputs[0] if len(outputs) == 1 else outputs

    def generate(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        get: Any,
        mapping: Any,
        capture: str,
    ) -> Any:
        if self._can_use_runtime_generate(args, kwargs):
            return self._generate_via_runtime(
                args=args,
                kwargs=kwargs,
                get=get,
                mapping=mapping,
                capture=capture,
            )
        generate_fn = getattr(self._model.wrapped, "generate", None)
        if not callable(generate_fn):
            raise AttributeError(
                f"Wrapped model {type(self._model.wrapped).__name__} does not define generate()."
            )
        return self._model._execute_now(
            execute=lambda call_kwargs: generate_fn(*args, **call_kwargs),
            args=args,
            kwargs=kwargs,
            get_proxies=self._model._normalize_get(get),
            map_proxies=self._model._normalize_map(mapping),
            grad=False,
            stop_at_last_get=False,
            always_output=bool(get or mapping),
        )

    def generate_requests(
        self,
        *,
        requests: list[Any],
        kwargs: Mapping[str, Any],
        get: Any,
        mapping: Any,
        capture: str,
    ) -> Any:
        if not requests:
            raise ValueError("Expected at least one request.")
        outputs = self._shared_runtime().generate_many(
            requests,
            plan=self._compiled_plan(
                self._model._normalize_get(get),
                self._model._normalize_map(mapping),
                output=None,
            ),
            capture=capture,
            **dict(kwargs),
        )
        return outputs

    def collect(
        self,
        *,
        data: Any,
        get: Any,
        mapping: Any,
        out: str | os.PathLike[str] | None,
        process: Any | None,
        max_items: int | None,
        max_tokens: int | None,
        sort: bool,
        stop_at_last_get: bool,
        kwargs: Mapping[str, Any],
    ) -> Any:
        get_proxies = self._model._normalize_get(get)
        if not get_proxies:
            raise ValueError("model.collect(...) requires at least one get= site.")
        map_proxies = self._model._normalize_map(mapping)
        if stop_at_last_get and map_proxies:
            raise ValueError("stop_at_last_get=True does not support map=.")
        plan = self._compiled_plan(
            get_proxies,
            map_proxies,
            output={"logits": False, "activations": True},
        )
        collector_kwargs, call_kwargs = _resolve_public_collect_kwargs(
            out=out,
            max_tokens=max_tokens,
            kwargs=kwargs,
        )
        if process is not None and not callable(process):
            raise TypeError("collect process= must be callable.")
        source_kind, source_value = _classify_collect_source(data)
        export_root = cast(str | None, collector_kwargs.get("mmap_path"))
        if process is not None and export_root is not None:
            raise ValueError("collect process= does not support out=PATH.")
        if source_kind == "dataset" and export_root is not None:
            with self._open_runtime_collector(
                plan=plan,
                stop_at_last_get=stop_at_last_get,
                collector_kwargs=collector_kwargs,
            ) as (runtime, collector):
                return self._collect_export(
                    runtime=runtime,
                    collector=collector,
                    units=source_value,
                    call_kwargs=call_kwargs,
                    max_items=max_items,
                    max_tokens=max_tokens,
                    sort=sort,
                    root=export_root,
                )
        if source_kind == "dataset" or process is not None:

            def iterator() -> Iterator[Any]:
                with self._open_runtime_collector(
                    plan=plan,
                    stop_at_last_get=stop_at_last_get,
                    collector_kwargs=collector_kwargs,
                ) as (runtime, collector):
                    for unit_kind, unit_value in _iter_collect_source_units(
                        source_kind,
                        source_value,
                        max_items=max_items,
                    ):
                        for step in self._iter_collect_steps_for_unit(
                            runtime=runtime,
                            collector=collector,
                            unit_kind=unit_kind,
                            unit_value=unit_value,
                            call_kwargs=call_kwargs,
                            max_items=max_items,
                            max_tokens=max_tokens,
                            sort=sort,
                        ):
                            if process is None:
                                yield step
                                continue
                            try:
                                yield process(step)
                            finally:
                                step.release()

            return iterator()

        with self._open_runtime_collector(
            plan=plan,
            stop_at_last_get=stop_at_last_get,
            collector_kwargs=collector_kwargs,
        ) as (runtime, collector):
            results = self._collect_results_for_unit(
                runtime=runtime,
                collector=collector,
                unit_kind=source_kind,
                unit_value=source_value,
                call_kwargs=call_kwargs,
                max_items=max_items,
                max_tokens=max_tokens,
                sort=sort,
            )
        if export_root is not None:
            return _build_collect_export(root=export_root, results=results)
        return [_plan_result_output(result) for result in results]

    @contextmanager
    def _open_runtime_collector(
        self,
        *,
        plan: Any,
        stop_at_last_get: bool,
        collector_kwargs: Mapping[str, Any],
    ) -> Iterator[tuple[Any, Any]]:
        runtime = self._shared_runtime()
        collector = runtime.open_collector(
            plan=plan,
            stop_at_last_get=stop_at_last_get,
            **dict(collector_kwargs),
        )
        try:
            yield runtime, collector
        finally:
            runtime.close_collector(collector)

    def _collect_results_for_unit(
        self,
        *,
        runtime: Any,
        collector: Any,
        unit_kind: str,
        unit_value: Any,
        call_kwargs: Mapping[str, Any],
        max_items: int | None,
        max_tokens: int | None,
        sort: bool,
    ) -> list[Any]:
        if unit_kind == "batch":
            return self._collect_batch_results(
                runtime=runtime,
                collector=collector,
                batch=cast(Mapping[str, Any], unit_value),
                call_kwargs=call_kwargs,
                max_items=max_items,
                max_tokens=max_tokens,
            )
        return self._collect_request_results(
            runtime=runtime,
            collector=collector,
            requests=unit_value,
            call_kwargs=call_kwargs,
            max_items=max_items,
            max_tokens=max_tokens,
            sort=sort,
        )

    def _iter_collect_steps_for_unit(
        self,
        *,
        runtime: Any,
        collector: Any,
        unit_kind: str,
        unit_value: Any,
        call_kwargs: Mapping[str, Any],
        max_items: int | None,
        max_tokens: int | None,
        sort: bool,
    ) -> Iterator[Any]:
        from .collect import CollectStep, normalize_collect_requests

        if unit_kind == "batch":
            for chunk, chunk_indices in _iter_batched_mapping_chunks(
                cast(Mapping[str, Any], unit_value),
                max_items=max_items,
                max_tokens=max_tokens,
            ):
                merged = _merge_collect_batch_kwargs(chunk, call_kwargs)
                device_batch = _device_local_batch(runtime, merged)
                result = runtime.collect_batch(collector, device_batch)
                yield CollectStep(
                    activations=dict(result.activations),
                    batch=device_batch,
                    rows=_rows_from_batched_request(chunk),
                    indices=chunk_indices,
                )
            return

        rows = normalize_collect_requests(self._model, unit_value)
        for batch_rows, batch_indices in _iter_request_row_batches(
            rows,
            max_items=max_items,
            max_tokens=max_tokens,
            sort=sort,
        ):
            normalized = runtime._normalize_requests(
                batch_rows,
                add_generation_prompt=False,
                pad_side="right",
            )
            merged = runtime._merge_batch_kwargs(normalized.batch, call_kwargs)
            device_batch = _device_local_batch(runtime, merged)
            result = runtime.collect_batch(collector, device_batch)
            yield CollectStep(
                activations=dict(result.activations),
                batch=device_batch,
                rows=batch_rows,
                indices=batch_indices,
            )

    def _collect_request_results(
        self,
        *,
        runtime: Any,
        collector: Any,
        requests: Any,
        call_kwargs: Mapping[str, Any],
        max_items: int | None,
        max_tokens: int | None,
        sort: bool,
    ) -> list[Any]:
        from .collect import normalize_collect_requests

        rows = normalize_collect_requests(self._model, requests)
        outputs: list[Any | None] = [None] * len(rows)
        for batch_rows, batch_indices in _iter_request_row_batches(
            rows,
            max_items=max_items,
            max_tokens=max_tokens,
            sort=sort,
        ):
            normalized = runtime._normalize_requests(
                batch_rows,
                add_generation_prompt=False,
                pad_side="right",
            )
            merged = runtime._merge_batch_kwargs(normalized.batch, call_kwargs)
            result = runtime.collect_batch(
                collector,
                _device_local_batch(runtime, merged),
            )
            for item, idx in zip(
                runtime._split_plan_result(result, batch_size=len(batch_rows)),
                batch_indices,
                strict=True,
            ):
                outputs[idx] = item
        return cast(list[Any], outputs)

    def _collect_batch_results(
        self,
        *,
        runtime: Any,
        collector: Any,
        batch: Mapping[str, Any],
        call_kwargs: Mapping[str, Any],
        max_items: int | None,
        max_tokens: int | None,
    ) -> list[Any]:
        outputs: list[Any] = []
        for chunk, _chunk_indices in _iter_batched_mapping_chunks(
            batch,
            max_items=max_items,
            max_tokens=max_tokens,
        ):
            result = runtime.collect_batch(
                collector,
                _device_local_batch(
                    runtime,
                    _merge_collect_batch_kwargs(chunk, call_kwargs),
                ),
            )
            outputs.extend(
                runtime._split_plan_result(
                    result,
                    batch_size=_batch_size_from_batch_mapping(chunk),
                )
            )
        return outputs

    def _collect_export(
        self,
        *,
        runtime: Any,
        collector: Any,
        units: Any,
        call_kwargs: Mapping[str, Any],
        max_items: int | None,
        max_tokens: int | None,
        sort: bool,
        root: str,
    ) -> Any:
        export_results: list[Any] = []
        for unit_kind, unit_value in _iter_collect_units(units, max_items=max_items):
            export_results.extend(
                self._collect_results_for_unit(
                    runtime=runtime,
                    collector=collector,
                    unit_kind=unit_kind,
                    unit_value=unit_value,
                    call_kwargs=call_kwargs,
                    max_items=max_items,
                    max_tokens=max_tokens,
                    sort=sort,
                )
            )
        return _build_collect_export(root=root, results=export_results)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._collector_cache.clear()
        self._plan_cache.clear()

    def _shared_runtime(self) -> Any:
        if self._runtime is None:
            from .runtime.core import _RuntimeCore

            self._runtime = _RuntimeCore(self._model)
        return self._runtime

    def _can_use_runtime_generate(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> bool:
        if len(args) > 1:
            return False
        if args and not isinstance(args[0], torch.Tensor):
            return False
        if args and "input_ids" in kwargs:
            return False
        return True

    def _generate_via_runtime(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        get: Any,
        mapping: Any,
        capture: str,
    ) -> Any:
        runtime = self._shared_runtime()
        plan = self._compiled_plan(
            self._model._normalize_get(get),
            self._model._normalize_map(mapping),
            output=None,
        )
        call_kwargs = dict(kwargs)
        if args:
            call_kwargs["input_ids"] = args[0]
        return runtime.generate(plan=plan, capture=capture, **call_kwargs)

    def _compiled_plan(
        self,
        get_proxies: list[Any],
        map_proxies: Mapping[Any, Any],
        *,
        output: Mapping[str, Any] | None,
    ) -> Any:
        key = (
            tuple(proxy.path for proxy in get_proxies),
            tuple(
                (proxy.path, _map_cache_key(fn))
                for proxy, fn in sorted(map_proxies.items(), key=lambda item: item[0].path)
            ),
            None if output is None else tuple(sorted(dict(output).items())),
        )
        plan = self._plan_cache.get(key)
        if plan is not None:
            return plan
        plan = self._shared_runtime().compile(
            get=get_proxies or None,
            mapping=dict(map_proxies) or None,
            output=output,
        )
        self._plan_cache[key] = plan
        return plan

    def _collector_for(
        self,
        plan: Any,
        *,
        stop_at_last_get: bool,
    ) -> Any:
        key = (plan.id, stop_at_last_get)
        collector = self._collector_cache.get(key)
        if collector is not None:
            return collector
        collector = self._shared_runtime().open_collector(
            plan=plan,
            stop_at_last_get=stop_at_last_get,
        )
        self._collector_cache[key] = collector
        return collector


def _plan_result_output(result: Any) -> Any:
    return output_from_path_activations(
        result,
        result.activations,
        completed_forward=result.completed_forward,
    )


def _batch_size_from_batch_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    return 1


def _rows_from_batched_request(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    batch_size = _batch_size_from_batch_mapping(batch)
    rows: list[dict[str, Any]] = []
    for idx in range(batch_size):
        row: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                row[key] = value[idx : idx + 1]
            else:
                row[key] = value
        rows.append(row)
    return rows


def _map_cache_key(fn: Any) -> Any:
    if isinstance(fn, maps_mod._Zero):
        return ("zero", None)
    if isinstance(fn, maps_mod._Add):
        return ("add", _cacheable_value(fn.delta))
    if isinstance(fn, maps_mod._Scale):
        return ("scale", _cacheable_value(fn.factor))
    if isinstance(fn, maps_mod._Replace):
        return ("replace", _cacheable_value(fn.value))
    with _CALLABLE_CACHE_LOCK:
        key = _CALLABLE_CACHE.get(id(fn))
        if key is not None and key.fn is fn:
            return ("callable", key)
        key = _CallableCacheKey(fn)
        _CALLABLE_CACHE[id(fn)] = key
        return ("callable", key)


def _cacheable_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return (
            "tensor",
            str(data.dtype),
            tuple(data.shape),
            hashlib.sha1(data.numpy().tobytes()).hexdigest()[:16],
        )
    return value


def _resolve_public_collect_kwargs(
    *,
    out: str | os.PathLike[str] | None,
    max_tokens: int | None,
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .collect import normalize_collect_out

    collector_kwargs, call_kwargs = split_collector_kwargs(kwargs)
    legacy_out = "activation_output" in collector_kwargs or "mmap_path" in collector_kwargs
    legacy_tokens = "token_budget" in collector_kwargs

    if legacy_out:
        raise TypeError("collect() no longer supports activation_output=/mmap_path=. Use out=.")
    if legacy_tokens:
        raise TypeError("collect() no longer supports token_budget=. Use max_tokens=.")
    if "reduce" in call_kwargs or "reducer" in collector_kwargs or "token_index" in collector_kwargs:
        raise TypeError("collect() no longer supports reduce=/reducer=/token_index=. Use process=.")

    if out is not None:
        activation_output, mmap_path = normalize_collect_out(out)
        collector_kwargs["activation_output"] = activation_output
        collector_kwargs["mmap_path"] = mmap_path
    else:
        collector_kwargs.setdefault("activation_output", "gpu")
        collector_kwargs.setdefault("mmap_path", None)

    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("collect max_tokens= must be positive.")
        collector_kwargs["token_budget"] = max_tokens

    if collector_kwargs["activation_output"] == "mmap" and not collector_kwargs.get("mmap_path"):
        raise ValueError("collect out=PATH is required for mmap exports.")
    return collector_kwargs, call_kwargs


def _classify_collect_source(data: Any) -> tuple[str, Any]:
    from .collect import is_batched_mapping

    if is_batched_mapping(data):
        return "batch", data
    items = request_items(data)
    if items is not None:
        if _looks_like_collect_dataset(data):
            return "dataset", data
        return "requests", data
    if isinstance(data, Iterable) and not isinstance(data, (str, Mapping, torch.Tensor)):
        return "dataset", data
    raise TypeError(
        "collect() expects one request, a sequence of requests, one batched tensor mapping, "
        "or an iterable dataset of request items or batched mappings."
    )


def _looks_like_collect_dataset(data: Any) -> bool:
    from .collect import is_batched_mapping

    return (
        isinstance(data, Sequence)
        and not isinstance(data, (str, Mapping, torch.Tensor))
        and bool(data)
        and any(is_batched_mapping(item) for item in data)
    )


def _iter_collect_units(
    units: Iterable[Any],
    *,
    max_items: int | None,
) -> Iterator[tuple[str, Any]]:
    from .collect import is_batched_mapping

    pending_requests: list[Any] = []
    pending_limit = max_items or 32
    for unit in units:
        if is_batched_mapping(unit):
            if pending_requests:
                yield "requests", list(pending_requests)
                pending_requests.clear()
            yield "batch", unit
            continue
        items = request_items(unit)
        if items is None:
            raise TypeError(
                "collect dataset items must be request items, request batches, or batched mappings."
            )
        pending_requests.extend(items)
        while len(pending_requests) >= pending_limit:
            yield "requests", list(pending_requests[:pending_limit])
            del pending_requests[:pending_limit]
    if pending_requests:
        yield "requests", list(pending_requests)


def _iter_collect_source_units(
    source_kind: str,
    source_value: Any,
    *,
    max_items: int | None,
) -> Iterator[tuple[str, Any]]:
    if source_kind == "dataset":
        yield from _iter_collect_units(source_value, max_items=max_items)
        return
    yield source_kind, source_value


def _iter_request_row_batches(
    rows: Sequence[Mapping[str, torch.Tensor]],
    *,
    max_items: int | None,
    max_tokens: int | None,
    sort: bool,
) -> Iterator[tuple[list[dict[str, torch.Tensor]], list[int]]]:
    if not rows:
        return
    if max_items is not None and max_items <= 0:
        raise ValueError("collect max_items= must be positive.")
    order = list(range(len(rows)))
    if sort:
        order.sort(key=lambda idx: _request_row_length(rows[idx]), reverse=True)
    cursor = 0
    item_limit = max_items if max_items is not None else len(order)
    while cursor < len(order):
        start = cursor
        current_max = 0
        while cursor < len(order) and (cursor - start) < item_limit:
            row = rows[order[cursor]]
            seq_len = _request_row_length(row)
            next_max = max(current_max, seq_len)
            next_count = (cursor - start) + 1
            if (
                next_count > 1
                and max_tokens is not None
                and next_max * next_count > max_tokens
            ):
                break
            current_max = next_max
            cursor += 1
        if cursor == start:
            cursor += 1
        batch_indices = order[start:cursor]
        yield [dict(rows[idx]) for idx in batch_indices], batch_indices


def _iter_request_item_chunks(
    items: Sequence[Any],
    *,
    max_items: int | None,
) -> Iterator[list[Any]]:
    if not items:
        return
    if max_items is not None and max_items <= 0:
        raise ValueError("collect max_items= must be positive.")
    if max_items is None:
        yield list(items)
        return
    for start in range(0, len(items), max_items):
        yield list(items[start : start + max_items])


def _iter_batched_mapping_chunks(
    batch: Mapping[str, Any],
    *,
    max_items: int | None,
    max_tokens: int | None,
) -> Iterator[tuple[dict[str, Any], list[int]]]:
    batch_size = _batch_size_from_batch_mapping(batch)
    if max_items is not None and max_items <= 0:
        raise ValueError("collect max_items= must be positive.")
    chunk_size = batch_size
    if max_items is not None:
        chunk_size = min(chunk_size, max_items)
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2 and max_tokens is not None:
        seq_len = int(input_ids.shape[1])
        token_limited = max(max_tokens // max(seq_len, 1), 1)
        chunk_size = min(chunk_size, token_limited)
    if chunk_size >= batch_size:
        yield dict(batch), list(range(batch_size))
        return
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        yield chunk, list(range(start, end))


def _merge_collect_batch_kwargs(
    batch: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    overlap = set(batch).intersection(kwargs)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"Duplicate batch kwargs: {joined}.")
    return {**dict(batch), **dict(kwargs)}


def _device_local_batch(runtime: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    from .runtime.util import move_tensors_to

    return cast(dict[str, Any], move_tensors_to(dict(batch), runtime._primary_device()))


def _build_collect_export(root: str, results: Sequence[Any]) -> Any:
    from .collect import CollectExport

    files: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for result in results:
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        mmap_files = metadata.get("mmap_files")
        if not isinstance(mmap_files, Mapping):
            continue
        for path, filenames in mmap_files.items():
            key = str(path)
            values = filenames if isinstance(filenames, Sequence) and not isinstance(filenames, str) else [filenames]
            existing = files.setdefault(key, [])
            existing_seen = seen.setdefault(key, set())
            for filename in values:
                name = str(filename)
                if name in existing_seen:
                    continue
                existing_seen.add(name)
                existing.append(name)
    return CollectExport(root=root, files=files, rows=len(results))


def _request_row_length(row: Mapping[str, Any]) -> int:
    attention_mask = row.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask.sum().item())
    input_ids = row.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    return 0
