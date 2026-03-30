"""Model wrapper and module proxying for tinyinterp."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from .context import get_debug, get_graph_path
from .counters import Counters
from .debug import log_call_start, log_model_ready, log_timing, render_intervention_graph
from .executors import _LocalExecutor, _RemoteExecutor
from .hooks import HookState, MapFn, _EarlyStop, install_hooks
from .output import GenerateOutput, Output, generate_output_from_value, merge_generate_outputs
from .requests import (
    normalize_request_row,
    request_items,
)


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
        wrapped: nn.Module | str | object,
        *,
        rename: Mapping[str, str] | None = None,
        tokenizer: Any | None = None,
        **load_kwargs: Any,
    ) -> None:
        if _is_server_instance(wrapped):
            raise TypeError(
                "ti.Model(server) was removed. Start the server with "
                '`server.serve(...)` and connect with `ti.Model("unix:///path.sock")`.'
            )
        if isinstance(wrapped, str) and _is_remote_endpoint(wrapped):
            if rename is not None or tokenizer is not None or load_kwargs:
                raise TypeError(
                    "Remote ti.Model(...) does not accept rename=, tokenizer=, or loading kwargs."
                )
            self.tokenizer = None
            self._normalize_requests_locally = False
            self._executor = _RemoteExecutor(_endpoint_path(wrapped))
            return
        if isinstance(wrapped, str):
            self.wrapped = _load_model(wrapped, **load_kwargs)
            if not isinstance(self.wrapped, nn.Module):
                raise TypeError(
                    "_load_model() must return a torch.nn.Module, "
                    f"got {type(self.wrapped).__name__}."
                )
            self.tokenizer = tokenizer if tokenizer is not None else _maybe_load_tokenizer(wrapped)
        else:
            if load_kwargs:
                raise TypeError(
                    "Loading kwargs are only valid when wrapped is a string model name."
                )
            if not isinstance(wrapped, nn.Module):
                raise TypeError(
                    "ti.Model(...) expects a torch.nn.Module, a model name/path, "
                    "or a unix:// endpoint."
                )
            self.wrapped = wrapped
            self.tokenizer = (
                tokenizer if tokenizer is not None else getattr(wrapped, "tokenizer", None)
            )

        self._renames = dict(rename or {})
        self._hooks = install_hooks(self.wrapped)
        self._root = _wrap_proxy(self.wrapped, "", self._hooks, self._renames)
        self._layers_proxy: _ModuleListProxy | None = None
        self._normalize_requests_locally = True
        self._executor = _LocalExecutor(self)

        if get_debug() >= 1:
            log_model_ready(type(self.wrapped).__name__, self._hooks.n_modules)

    def __getattr__(self, name: str) -> Any:
        return self._executor.get_attr(name)

    def __dir__(self) -> list[str]:
        return self._executor.dir()

    @property
    def layers(self) -> _ModuleListProxy:
        """Shortcut to the biggest ``ModuleList`` in the model."""
        return self._executor.layers

    @property
    def device(self) -> torch.device | tuple[torch.device, ...]:
        return self._executor.device

    @property
    def capabilities(self) -> dict[str, Any]:
        return dict(self._executor.capabilities)

    def __call__(
        self,
        *args: Any,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None = None,
        map: dict[_ModuleProxy, MapFn] | None = None,
        grad: bool = False,
        stop_at_last_get: bool = False,
        **kwargs: Any,
    ) -> Any:
        if (
            request_rows := self._normalize_request_args(args, add_generation_prompt=False)
        ) is not None:
            return self._executor.call_requests(
                requests=request_rows,
                kwargs=_filter_model_kwargs(self.wrapped, kwargs),
                get=get,
                mapping=map,
                grad=grad,
                stop_at_last_get=stop_at_last_get,
            )
        return self._executor.call(
            args=tuple(args),
            kwargs=dict(kwargs),
            get=get,
            mapping=map,
            grad=grad,
            stop_at_last_get=stop_at_last_get,
        )

    def generate(
        self,
        *args: Any,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None = None,
        map: dict[_ModuleProxy, MapFn] | None = None,
        stop_at_last_get: bool = False,
        capture: str = "all",
        **kwargs: Any,
    ) -> GenerateOutput:
        """Run generation with hooks active across the whole call."""

        if stop_at_last_get:
            raise ValueError("stop_at_last_get=True is not supported for model.generate().")
        if (
            request_rows := self._normalize_request_args(args, add_generation_prompt=True)
        ) is not None:
            return self._normalize_generate_result(
                self._executor.generate_requests(
                    requests=request_rows,
                    kwargs=kwargs,
                    get=get,
                    mapping=map,
                    capture=capture,
                ),
                prompt_lengths=[int(row["input_ids"].shape[-1]) for row in request_rows],
            )
        return self._normalize_generate_result(
            self._executor.generate(
                args=tuple(args),
                kwargs=dict(kwargs),
                get=get,
                mapping=map,
                capture=capture,
            ),
            prompt_length=self._infer_generate_prompt_length(args, kwargs),
        )

    def collect(
        self,
        requests: Sequence[Any] | Any,
        *,
        get: Sequence[_ModuleProxy] | _ModuleProxy | None = None,
        map: dict[_ModuleProxy, MapFn] | None = None,
        stop_at_last_get: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Collect activations over a request list using the local model API."""

        return self._executor.collect(
            requests=requests,
            get=get,
            mapping=map,
            stop_at_last_get=stop_at_last_get,
            kwargs=dict(kwargs),
        )

    def _normalize_request_args(
        self,
        args: tuple[Any, ...],
        *,
        add_generation_prompt: bool,
    ) -> list[dict[str, torch.Tensor]] | None:
        if not self._normalize_requests_locally:
            return None
        if len(args) != 1:
            return None
        items = request_items(args[0])
        if items is None:
            return None
        return [
            self._normalize_request_row(
                request,
                add_generation_prompt=add_generation_prompt,
            )
            for request in items
        ]

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

    def _normalize_generate_result(
        self,
        value: Any,
        *,
        prompt_length: int | None = None,
        prompt_lengths: list[int] | None = None,
    ) -> GenerateOutput:
        if isinstance(value, GenerateOutput):
            return value
        if isinstance(value, list):
            if all(isinstance(item, GenerateOutput) for item in value):
                return merge_generate_outputs(cast(list[GenerateOutput], value))
            if prompt_lengths is None:
                raise TypeError(
                    "model.generate(...) could not infer prompt lengths for batched outputs."
                )
            if len(value) != len(prompt_lengths):
                raise ValueError("model.generate(...) returned an unexpected number of outputs.")
            return merge_generate_outputs(
                [
                    generate_output_from_value(item, prompt_length=prompt_lengths[idx])
                    for idx, item in enumerate(value)
                ]
            )
        if prompt_length is None:
            if prompt_lengths is not None and len(prompt_lengths) == 1:
                prompt_length = prompt_lengths[0]
            else:
                raise TypeError(
                    "model.generate(...) could not infer the prompt length for its output."
                )
        return generate_output_from_value(value, prompt_length=prompt_length)

    @staticmethod
    def _infer_generate_prompt_length(
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> int | None:
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return int(input_ids.shape[-1])
        if args and isinstance(args[0], torch.Tensor):
            return int(args[0].shape[-1])
        return None

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
        always_output: bool = False,
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
        hook_token = self._hooks.activate(
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
            activations = self._hooks.collect_and_deactivate(hook_token, strict=not failed)
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

        if not always_output and not get_proxies and not map_proxies:
            return model_output
        return Output(
            model_output,
            activations,
            self._hooks.id_map,
            path_to_sid=self._hooks.path_to_sid,
            completed_forward=not stopped_early,
        )

    def _normalize_request_row(
        self,
        request: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        return normalize_request_row(
            request,
            tokenizer=self.tokenizer,
            add_generation_prompt=add_generation_prompt,
            owner="Model",
        )

    def close(self) -> None:
        self._executor.close()


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


def _filter_model_kwargs(model: nn.Module, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    forward = getattr(model, "forward", None)
    if forward is None:
        return dict(kwargs)
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def _endpoint_path(path: str) -> str:
    return path[len("unix://") :]


def _is_remote_endpoint(path: str) -> bool:
    return path.startswith("unix://")


def _is_server_instance(value: object) -> bool:
    try:
        from .server.inference import Server
    except Exception:
        return False
    return isinstance(value, Server)


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
