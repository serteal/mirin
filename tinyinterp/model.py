"""Model wrapper and module proxying for tinyinterp."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from .batch import batch_active, maybe_enqueue_call
from .context import get_debug, get_graph_path
from .counters import Counters
from .debug import log_call_start, log_model_ready, log_timing, render_intervention_graph
from .hooks import HookState, MapFn, _EarlyStop, install_hooks
from .output import Output


class _ModuleProxy:
    """Wrap any ``nn.Module``. Usable as a site in ``get=`` and ``map=``."""

    __slots__ = ("_module", "_path", "_hooks", "_renames")

    def __init__(
        self,
        module: nn.Module,
        path: str,
        hooks: HookState,
        renames: Mapping[str, str],
    ) -> None:
        self._module = module
        self._path = path
        self._hooks = hooks
        self._renames = renames

    @property
    def path(self) -> str:
        return self._path or "<root>"

    @property
    def weight(self) -> torch.Tensor:
        return torch.as_tensor(self._module.weight)

    @property
    def bias(self) -> torch.Tensor | None:
        bias = getattr(self._module, "bias", None)
        if bias is None or isinstance(bias, torch.Tensor):
            return bias
        raise AttributeError(f"{type(self._module).__name__} does not expose a tensor bias.")

    def __getattr__(self, name: str) -> Any:
        real_name = self._resolve(name)
        try:
            child = getattr(self._module, real_name)
        except AttributeError as exc:
            available = ", ".join(self.__dir__())
            raise AttributeError(
                f"{type(self._module).__name__} has no child {name!r}. Available: {available}"
            ) from exc
        if isinstance(child, nn.Module):
            return _wrap_proxy(child, _join_path(self._path, real_name), self._hooks, self._renames)
        return child

    def __getitem__(self, idx: int) -> Any:
        child = cast(Any, self._module)[idx]
        if isinstance(child, nn.Module):
            return _wrap_proxy(child, _join_path(self._path, str(idx)), self._hooks, self._renames)
        return child

    def __hash__(self) -> int:
        return id(self._module)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ModuleProxy) and self._module is other._module

    def __dir__(self) -> list[str]:
        real = [name for name, _ in self._module.named_children()]
        aliases = [tgt for src, tgt in self._renames.items() if hasattr(self._module, src)]
        return sorted(set(real + aliases))

    def __repr__(self) -> str:
        return f"Site({self.path})"

    def _resolve(self, name: str) -> str:
        for src, tgt in self._renames.items():
            if tgt == name and hasattr(self._module, src):
                return src
        return name


class _ModuleListProxy(_ModuleProxy):
    """Indexable proxy for ``nn.ModuleList`` containers."""

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[_ModuleProxy]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(cast(Any, self._module))


class Model:
    """Wrap a model for activation access without changing its call signature."""

    def __init__(
        self,
        wrapped: nn.Module | str,
        *,
        rename: Mapping[str, str] | None = None,
        tokenizer: Any | None = None,
        **load_kwargs: Any,
    ) -> None:
        if isinstance(wrapped, str):
            self.wrapped = _load_model(wrapped, **load_kwargs)
            self.tokenizer = tokenizer if tokenizer is not None else _maybe_load_tokenizer(wrapped)
        else:
            if load_kwargs:
                raise TypeError(
                    "Loading kwargs are only valid when wrapped is a string model name."
                )
            self.wrapped = wrapped
            self.tokenizer = (
                tokenizer if tokenizer is not None else getattr(wrapped, "tokenizer", None)
            )

        self._renames = dict(rename or {})
        self._hooks = install_hooks(self.wrapped)
        self._root = _wrap_proxy(self.wrapped, "", self._hooks, self._renames)
        self._layers_proxy: _ModuleListProxy | None = None

        if get_debug() >= 1:
            log_model_ready(type(self.wrapped).__name__, self._hooks.n_modules)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._root, name)

    def __dir__(self) -> list[str]:
        return sorted(set(list(object.__dir__(self)) + dir(self._root)))

    @property
    def layers(self) -> _ModuleListProxy:
        """Shortcut to the biggest ``ModuleList`` in the model."""

        if self._layers_proxy is None:
            best: tuple[str, nn.ModuleList] | None = None
            for path, module in self.wrapped.named_modules():
                if not isinstance(module, nn.ModuleList):
                    continue
                if best is None or _layers_sort_key(path, module) > _layers_sort_key(*best):
                    best = (path, module)
            if best is None:
                raise AttributeError("No ModuleList found. Navigate directly.")
            proxy = _wrap_proxy(best[1], best[0], self._hooks, self._renames)
            assert isinstance(proxy, _ModuleListProxy)
            self._layers_proxy = proxy
        return self._layers_proxy

    @property
    def device(self) -> torch.device | tuple[torch.device, ...]:
        devices = {tensor.device for tensor in self.wrapped.parameters()}
        devices.update(tensor.device for tensor in self.wrapped.buffers())
        if not devices:
            return torch.device("cpu")
        if len(devices) == 1:
            return next(iter(devices))
        return tuple(sorted(devices, key=str))

    def __call__(
        self,
        *args: Any,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None = None,
        map: dict[_ModuleProxy, MapFn] | None = None,
        grad: bool = False,
        stop_at_last_get: bool = False,
        **kwargs: Any,
    ) -> Any:
        get_proxies = self._normalize_get(get)
        map_proxies = self._normalize_map(map)
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
            self,
            args=tuple(args),
            kwargs=dict(kwargs),
            get_proxies=get_proxies,
            map_proxies=map_proxies,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
        )
        if deferred is not None:
            return deferred
        return self._execute_now(
            args=tuple(args),
            kwargs=dict(kwargs),
            get_proxies=get_proxies,
            map_proxies=map_proxies,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
        )

    def generate(
        self,
        *args: Any,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None = None,
        map: dict[_ModuleProxy, MapFn] | None = None,
        stop_at_last_get: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run generation with hooks active across the whole call."""

        if stop_at_last_get:
            raise ValueError("stop_at_last_get=True is not supported for model.generate().")
        generate_fn = getattr(self.wrapped, "generate", None)
        if not callable(generate_fn):
            raise AttributeError(
                f"Wrapped model {type(self.wrapped).__name__} does not define generate()."
            )
        return self._execute_now(
            execute=lambda call_kwargs: generate_fn(*args, **call_kwargs),
            args=tuple(args),
            kwargs=dict(kwargs),
            get_proxies=self._normalize_get(get),
            map_proxies=self._normalize_map(map),
            grad=False,
            stop_at_last_get=False,
        )

    def _normalize_get(
        self,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None,
    ) -> list[_ModuleProxy]:
        if get is None:
            return []
        proxies = [get] if isinstance(get, _ModuleProxy) else list(get)
        for proxy in proxies:
            self._validate_proxy(proxy)
        return proxies

    def _normalize_map(self, map: dict[_ModuleProxy, MapFn] | None) -> dict[_ModuleProxy, MapFn]:
        if map is None:
            return {}
        normalized = dict(map)
        for proxy, fn in normalized.items():
            self._validate_proxy(proxy)
            if not callable(fn):
                raise TypeError(f"Map for {proxy.path!r} is not callable.")
        return normalized

    def _validate_proxy(self, proxy: object) -> _ModuleProxy:
        if not isinstance(proxy, _ModuleProxy):
            raise TypeError("get= and map= must use tinyinterp module proxies.")
        if proxy._hooks is not self._hooks:
            raise ValueError(f"Proxy {proxy.path!r} does not belong to this model.")
        return proxy

    def _execute_now(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        get_proxies: list[_ModuleProxy],
        map_proxies: dict[_ModuleProxy, MapFn],
        grad: bool,
        stop_at_last_get: bool = False,
        requested_calls: int = 1,
        execute: Callable[[dict[str, Any]], Any] | None = None,
    ) -> Any:
        if execute is None:

            def execute(call_kwargs: dict[str, Any]) -> Any:
                return self.wrapped(*args, **call_kwargs)

        debug = get_debug()
        if debug >= 1:
            log_call_start(
                get_proxies,
                map_proxies,
                grad=grad,
                stop_at_last_get=stop_at_last_get,
                args=args,
                kwargs=kwargs,
            )

        t0 = time.perf_counter_ns()
        self._hooks.activate(
            get_proxies,
            map_proxies,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
        )
        t1 = time.perf_counter_ns()

        activations: dict[int, torch.Tensor] = {}
        failed = False
        stopped_early = False
        model_output: Any = None
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                model_output = execute(dict(kwargs))
        except _EarlyStop:
            stopped_early = True
        except Exception:
            failed = True
            raise
        finally:
            t2 = time.perf_counter_ns()
            activations = self._hooks.collect_and_deactivate(strict=not failed)
            t3 = time.perf_counter_ns()

        forward_ns = t2 - t1
        hook_overhead_ns = (t1 - t0) + (t3 - t2)
        total_time_ns = t3 - t0
        activation_bytes = sum(
            tensor.element_size() * tensor.numel() for tensor in activations.values()
        )
        Counters.calls += requested_calls
        Counters.forward_passes += 1
        Counters.total_time_ns += total_time_ns
        Counters.forward_time_ns += forward_ns
        Counters.hook_overhead_ns += hook_overhead_ns
        Counters.activations_captured += len(activations)
        Counters.activations_bytes += activation_bytes
        if stopped_early:
            Counters.early_stops += requested_calls

        if debug >= 2:
            log_timing(
                activate_ns=t1 - t0,
                forward_ns=forward_ns,
                collect_ns=t3 - t2,
                n_activations=len(activations),
                activation_bytes=activation_bytes,
                stopped_early=stopped_early,
            )
        graph_path = get_graph_path()
        if not failed and graph_path is not None and (get_proxies or map_proxies):
            render_intervention_graph(get_proxies, map_proxies, output_path=graph_path)

        if not get_proxies and not map_proxies:
            return model_output
        return Output(
            model_output,
            activations,
            self._hooks.id_map,
            completed_forward=not stopped_early,
        )


def _wrap_proxy(
    module: nn.Module,
    path: str,
    hooks: HookState,
    renames: Mapping[str, str],
) -> _ModuleProxy:
    if isinstance(module, nn.ModuleList):
        return _ModuleListProxy(module, path, hooks, renames)
    return _ModuleProxy(module, path, hooks, renames)


def _join_path(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _layers_sort_key(path: str, module: nn.ModuleList) -> tuple[int, int, int]:
    lowered = path.lower()
    score = 0
    if "language_model.layers" in lowered:
        score += 400
    elif lowered.endswith("model.layers"):
        score += 300
    elif lowered.endswith("layers"):
        score += 200
    elif lowered.endswith(".h"):
        score += 150
    if "vision" in lowered:
        score -= 500
    return (score, len(module), -lowered.count("."))


def _load_model(name_or_path: str, **load_kwargs: Any) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            f'ti.Model("{name_or_path}") requires `transformers` to load models by name.\n'
            "Install with: pip install tinyinterp[transformers]\n"
            "Or pass an already-loaded model: ti.Model(your_model)"
        ) from exc

    return AutoModelForCausalLM.from_pretrained(name_or_path, **load_kwargs)


def _maybe_load_tokenizer(name_or_path: str) -> Any | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception:
        return None
