"""Permanent hook management for tinyinterp."""

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
    """Permanent hooks with per-call capture state stored in a context variable.

    Hook capture follows the current execution context. This supports overlapping
    calls in one process, but forwards that dispatch hooked module execution onto
    other threads are intentionally unsupported.
    """

    __slots__ = ("_id_map", "_paths", "_handles", "_current")

    def __init__(self) -> None:
        self._id_map: dict[int, int] = {}
        self._paths: list[str] = []
        self._handles: list[RemovableHandle] = []
        self._current: ContextVar[_CallState | None] = ContextVar(
            "tinyinterp_hook_state", default=None
        )

    @property
    def n_modules(self) -> int:
        return len(self._paths)

    @property
    def id_map(self) -> dict[int, int]:
        return self._id_map

    @property
    def path_to_sid(self) -> dict[str, int]:
        return {path: sid for sid, path in enumerate(self._paths)}

    def register(self, module: nn.Module, path: str) -> None:
        sid = len(self._paths)
        self._id_map[id(module)] = sid
        self._paths.append(path)
        self._handles.append(module.register_forward_hook(self._make_hook(path, sid)))

    def sid_for(self, module: nn.Module) -> int:
        return self._id_map[id(module)]

    def activate(
        self,
        get_proxies: Sequence[Any],
        map_dict: dict[Any, MapFn],
        *,
        grad: bool,
        stop_at_last_get: bool = False,
    ) -> Token[_CallState | None]:
        get_sids = {self.sid_for(proxy._module) for proxy in get_proxies}
        state = _CallState(
            get_sids=get_sids,
            map_fns={self.sid_for(proxy._module): fn for proxy, fn in map_dict.items()},
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
    """Install permanent forward hooks on every module in the tree."""

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
