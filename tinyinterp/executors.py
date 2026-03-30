"""Internal executors for local and remote model backends."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from . import maps as maps_mod
from .batch import batch_active, maybe_enqueue_call
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
            "remote": False,
            "grad": True,
            "lazy_values": False,
            "request_tokenization": self._model.tokenizer is not None,
            "protocol": None,
        }

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
        requests: Any,
        get: Any,
        mapping: Any,
        stop_at_last_get: bool,
        kwargs: Mapping[str, Any],
    ) -> list[Any]:
        get_proxies = self._model._normalize_get(get)
        if not get_proxies:
            raise ValueError("model.collect(...) requires at least one get= site.")
        map_proxies = self._model._normalize_map(mapping)
        items = request_items(requests)
        if items is None:
            raise TypeError("model.collect(...) expects one request or a sequence of requests.")
        if not items:
            raise ValueError("Expected at least one request.")
        plan = self._compiled_plan(
            get_proxies,
            map_proxies,
            output={"logits": False, "activations": True},
        )
        collector = self._collector_for(
            plan,
            stop_at_last_get=stop_at_last_get,
        )
        results = self._shared_runtime().collect_many(
            collector,
            items,
            **dict(kwargs),
        )
        return [_plan_result_output(result) for result in results]

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._collector_cache.clear()
        self._plan_cache.clear()

    def _shared_runtime(self) -> Any:
        if self._runtime is None:
            from .server.inference import _RuntimeCore

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


class _RemoteExecutor:
    def __init__(self, sock_path: str) -> None:
        from .server.remote import _RemoteModel

        self._remote = _RemoteModel(sock_path)

    def get_attr(self, name: str) -> Any:
        return getattr(self._remote, name)

    def dir(self) -> list[str]:
        return sorted(set(list(object.__dir__(self._remote)) + dir(self._remote)))

    @property
    def layers(self) -> Any:
        return self._remote.layers

    @property
    def device(self) -> Any:
        return self._remote.device

    @property
    def capabilities(self) -> dict[str, Any]:
        return dict(self._remote.capabilities)

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
        return self._remote(
            *args,
            get=get,
            map=mapping,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
            **kwargs,
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
        return self._remote(
            requests,
            get=get,
            map=mapping,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
            **dict(kwargs),
        )

    def generate(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        get: Any,
        mapping: Any,
        capture: str,
    ) -> Any:
        return self._remote.generate(
            *args,
            get=get,
            map=mapping,
            capture=capture,
            **kwargs,
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
        return self._remote.generate(
            requests,
            get=get,
            map=mapping,
            capture=capture,
            **dict(kwargs),
        )

    def collect(
        self,
        *,
        requests: Any,
        get: Any,
        mapping: Any,
        stop_at_last_get: bool,
        kwargs: Mapping[str, Any],
    ) -> list[Any]:
        return self._remote.collect(
            requests,
            get=get,
            map=mapping,
            stop_at_last_get=stop_at_last_get,
            **dict(kwargs),
        )

    def close(self) -> None:
        self._remote.close()


def _plan_result_output(result: Any) -> Any:
    return output_from_path_activations(
        result,
        result.activations,
        completed_forward=result.completed_forward,
    )


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
