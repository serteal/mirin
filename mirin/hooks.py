"""Lazy hook management for mirin.

Hooks are bookkept for every module up front but ``register_forward_hook`` is
deferred until a site is first requested via ``get=`` or ``map=``. The hook
then stays installed, so subsequent calls pay no registration cost.

Why: PyTorch dispatches every module's hook list on every forward, and each
fired hook does at minimum a ``ContextVar.get`` + a set membership check
before short-circuiting. For an 8B Llama that's ~600 firings per forward.
During decode you pay this per generated token. Lazy registration shrinks
the fan-out to the modules the user has actually asked about — usually one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .context import get_debug
from .counters import Counters
from .debug import log_hook_event

MapFn = Callable[[torch.Tensor], torch.Tensor]


class _EarlyStop(Exception):
    """Internal control-flow exception used for capture-only early stop."""


@dataclass(slots=True)
class _CallState:
    get_sids: set[int]
    map_fns: dict[int, MapFn]
    grad: bool
    stop_enabled: bool
    remaining_gets: int
    buffers: dict[int, torch.Tensor] = field(default_factory=dict)
    stop_seen: set[int] = field(default_factory=set)


class HookState:
    """Lazy hooks with per-call capture state stored in a context variable.

    Bookkeeping (sid assignment, path list, module handle) happens up front
    for every module, but ``register_forward_hook`` is deferred until the
    site is first requested via ``activate``. Once installed, the hook stays
    installed for the lifetime of the model so repeat calls are cheap.

    Hook capture follows the current execution context. This supports overlapping
    calls in one process, but forwards that dispatch hooked module execution onto
    other threads are intentionally unsupported.
    """

    __slots__ = ("_id_map", "_paths", "_modules", "_handles", "_current")

    def __init__(self) -> None:
        self._id_map: dict[int, int] = {}
        self._paths: list[str] = []
        self._modules: list[nn.Module] = []
        self._handles: dict[int, RemovableHandle] = {}
        self._current: ContextVar[_CallState | None] = ContextVar(
            "mirin_hook_state", default=None
        )

    @property
    def n_modules(self) -> int:
        return len(self._paths)

    @property
    def n_active_hooks(self) -> int:
        """Number of modules currently carrying a forward hook."""
        return len(self._handles)

    @property
    def id_map(self) -> dict[int, int]:
        return self._id_map

    @property
    def path_to_sid(self) -> dict[str, int]:
        return {path: sid for sid, path in enumerate(self._paths)}

    def register(self, module: nn.Module, path: str) -> None:
        """Bookkeep this module so ``sid_for`` works later. The actual forward
        hook is installed lazily in :meth:`activate` the first time the site
        is requested.
        """
        sid = len(self._paths)
        self._id_map[id(module)] = sid
        self._paths.append(path)
        self._modules.append(module)

    def sid_for(self, module: nn.Module) -> int:
        return self._id_map[id(module)]

    def _ensure_hook(self, sid: int) -> None:
        if sid in self._handles:
            return
        module = self._modules[sid]
        path = self._paths[sid]
        self._handles[sid] = module.register_forward_hook(self._make_hook(path, sid))

    def activate(
        self,
        get_proxies: Sequence[Any],
        map_dict: dict[Any, MapFn],
        *,
        grad: bool,
        stop_at_last_get: bool = False,
    ) -> Token[_CallState | None]:
        get_sids = {self.sid_for(proxy._module) for proxy in get_proxies}
        map_sids = {self.sid_for(proxy._module): fn for proxy, fn in map_dict.items()}
        for sid in get_sids:
            self._ensure_hook(sid)
        for sid in map_sids:
            self._ensure_hook(sid)
        state = _CallState(
            get_sids=get_sids,
            map_fns=map_sids,
            grad=grad,
            stop_enabled=stop_at_last_get and bool(get_sids),
            remaining_gets=len(get_sids),
        )
        return self._current.set(state)

    def collect_and_deactivate(
        self,
        token: Token[_CallState | None],
        *,
        strict: bool,
    ) -> dict[int, torch.Tensor]:
        state = self._current.get()
        self._current.reset(token)
        if state is None:
            return {}
        missing = [self._paths[sid] for sid in sorted(state.get_sids.difference(state.buffers))]
        if missing and strict:
            joined = ", ".join(missing)
            raise RuntimeError(f"Requested modules did not capture activations: {joined}")
        return dict(state.buffers)

    def _make_hook(
        self,
        path: str,
        sid: int,
    ) -> Callable[[nn.Module, tuple[object, ...], object], object]:
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            state = self._current.get()
            debug = get_debug()
            if state is None:
                if debug >= 4:
                    log_hook_event(path, sid=sid, get=False, map_fn=None)
                return output

            get_flag = sid in state.get_sids
            map_fn = state.map_fns.get(sid)
            if not get_flag and map_fn is None:
                if debug >= 4:
                    log_hook_event(path, sid=sid, get=False, map_fn=None)
                return output

            activation = _extract(output)
            if get_flag:
                if state.grad and activation.requires_grad:
                    activation.retain_grad()
                    state.buffers[sid] = activation
                elif map_fn is not None:
                    state.buffers[sid] = activation.detach().clone()
                else:
                    state.buffers[sid] = activation.detach()

            if debug >= 4:
                log_hook_event(path, sid=sid, get=get_flag, map_fn=map_fn, activation=activation)

            if state.stop_enabled and get_flag and sid not in state.stop_seen:
                # stop_at_last_get counts unique requested sites, not repeated hits of the
                # same site within one forward pass.
                state.stop_seen.add(sid)
                state.remaining_gets -= 1
                if state.remaining_gets == 0:
                    raise _EarlyStop

            if map_fn is None:
                return output

            map_input = (
                activation.clone() if state.grad and activation.requires_grad else activation
            )
            mapped = map_fn(map_input)
            if not isinstance(mapped, torch.Tensor):
                raise TypeError(f"Map function must return a tensor, got {type(mapped).__name__}.")
            Counters.maps_applied += 1
            return _replace(output, mapped)

        return hook


def install_hooks(model: nn.Module) -> HookState:
    """Bookkeep every module in the tree. Forward hooks are not installed
    here; ``HookState.activate`` installs them lazily on first use of each
    site, then keeps them installed for the model's lifetime.
    """

    state = HookState()
    for path, module in model.named_modules():
        state.register(module, _format_path(path))
    return state


def _extract(output: object) -> torch.Tensor:
    """Get the main tensor from a module output."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}.")


def _replace(output: object, new: torch.Tensor) -> object:
    """Put a modified tensor back into the original output format."""

    if isinstance(output, torch.Tensor):
        return new
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (new,) + output[1:]
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        cast(Any, output).last_hidden_state = new
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        cast(Any, output).logits = new
        return output
    raise TypeError(f"Cannot replace tensor inside {type(output).__name__}.")


def _format_path(path: str) -> str:
    return path or "<root>"
