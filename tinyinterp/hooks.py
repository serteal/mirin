"""Permanent hook management for tinyinterp."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .context import get_debug
from .counters import Counters
from .debug import log_hook_event

MapFn = Callable[[torch.Tensor], torch.Tensor]


class HookState:
    """Mutable per-model state shared by all permanent hooks."""

    __slots__ = (
        "_id_map",
        "_paths",
        "_get_flags",
        "_map_fns",
        "_buffers",
        "_handles",
        "grad",
    )

    def __init__(self) -> None:
        self._id_map: dict[int, int] = {}
        self._paths: list[str] = []
        self._get_flags: list[bool] = []
        self._map_fns: list[MapFn | None] = []
        self._buffers: list[torch.Tensor | None] = []
        self._handles: list[RemovableHandle] = []
        self.grad = False

    @property
    def n_modules(self) -> int:
        return len(self._paths)

    @property
    def id_map(self) -> dict[int, int]:
        return self._id_map

    def register(self, module: nn.Module, path: str) -> None:
        sid = len(self._paths)
        self._id_map[id(module)] = sid
        self._paths.append(path)
        self._get_flags.append(False)
        self._map_fns.append(None)
        self._buffers.append(None)
        self._handles.append(module.register_forward_hook(self._make_hook(path, sid)))

    def sid_for(self, module: nn.Module) -> int:
        return self._id_map[id(module)]

    def activate(
        self,
        get_proxies: Sequence[Any],
        map_dict: dict[Any, MapFn],
        *,
        grad: bool,
    ) -> None:
        self.grad = grad
        for proxy in get_proxies:
            self._get_flags[self.sid_for(proxy._module)] = True
        for proxy, fn in map_dict.items():
            self._map_fns[self.sid_for(proxy._module)] = fn

    def collect_and_deactivate(self, *, strict: bool) -> dict[int, torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        missing: list[str] = []
        for sid, get_flag in enumerate(self._get_flags):
            if not get_flag:
                continue
            buffer = self._buffers[sid]
            if buffer is None:
                if strict:
                    missing.append(self._paths[sid])
                continue
            captured[sid] = buffer
        self.reset()
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Requested modules did not capture activations: {joined}")
        return captured

    def reset(self) -> None:
        for sid in range(len(self._paths)):
            self._get_flags[sid] = False
            self._map_fns[sid] = None
            self._buffers[sid] = None
        self.grad = False

    def _make_hook(
        self,
        path: str,
        sid: int,
    ) -> Callable[[nn.Module, tuple[object, ...], object], object]:
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            get_flag = self._get_flags[sid]
            map_fn = self._map_fns[sid]
            debug = get_debug()

            if not get_flag and map_fn is None:
                if debug >= 4:
                    log_hook_event(path, sid=sid, get=False, map_fn=None)
                return output

            activation = _extract(output)
            if get_flag:
                if self.grad and activation.requires_grad:
                    activation.retain_grad()
                    self._buffers[sid] = activation
                elif map_fn is not None:
                    self._buffers[sid] = activation.detach().clone()
                else:
                    self._buffers[sid] = activation.detach()

            if debug >= 4:
                log_hook_event(path, sid=sid, get=get_flag, map_fn=map_fn, activation=activation)

            if map_fn is None:
                return output

            mapped = map_fn(activation)
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
    return new


def _format_path(path: str) -> str:
    return path or "<root>"
