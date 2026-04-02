"""Small exploration helpers for module proxies."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch.nn as nn


class _ProxyLike(Protocol):
    _module: nn.Module


class _RemoteProxyLike(Protocol):
    def _remote_children(self) -> list[tuple[str, str]]: ...


def find(proxy: _ProxyLike, pattern: str) -> Any | None:
    """Return the first direct child whose name or class matches ``pattern``."""

    if _is_remote_proxy(proxy):
        return _find_remote(proxy, pattern)
    _ensure_proxy(proxy)
    needle = pattern.lower()
    for name, child in proxy._module.named_children():
        if needle == name.lower() or needle == type(child).__name__.lower():
            return getattr(proxy, name)
    for name, child in proxy._module.named_children():
        if needle in name.lower() or needle in type(child).__name__.lower():
            return getattr(proxy, name)
    return None


def find_all(proxy: _ProxyLike, pattern: str) -> list[Any]:
    """Return all direct children whose class name matches ``pattern``."""

    if _is_remote_proxy(proxy):
        return _find_all_remote(proxy, pattern)
    _ensure_proxy(proxy)
    needle = pattern.lower()
    return [
        getattr(proxy, name)
        for name, child in proxy._module.named_children()
        if needle in name.lower() or needle in type(child).__name__.lower()
    ]


def children(proxy: _ProxyLike) -> list[tuple[str, str]]:
    """List direct child names and class names for exploration."""

    if _is_remote_proxy(proxy):
        return cast(_RemoteProxyLike, proxy)._remote_children()
    _ensure_proxy(proxy)
    return [(name, type(child).__name__) for name, child in proxy._module.named_children()]


def _ensure_proxy(proxy: object) -> _ProxyLike:
    module = getattr(proxy, "_module", None)
    if not isinstance(module, nn.Module):
        raise TypeError("Expected a mirin module proxy.")
    return cast(_ProxyLike, proxy)


def _is_remote_proxy(proxy: object) -> bool:
    return callable(getattr(proxy, "_remote_children", None))


def _find_remote(proxy: object, pattern: str) -> Any | None:
    matches = _find_all_remote(proxy, pattern)
    if not matches:
        return None
    return matches[0]


def _find_all_remote(proxy: object, pattern: str) -> list[Any]:
    remote = cast(_RemoteProxyLike, proxy)
    needle = pattern.lower()
    exact: list[Any] = []
    partial: list[Any] = []
    for name, class_name in remote._remote_children():
        child = getattr(proxy, name)
        lowered_name = name.lower()
        lowered_class = class_name.lower()
        if needle == lowered_name or needle == lowered_class:
            exact.append(child)
            continue
        if needle in lowered_name or needle in lowered_class:
            partial.append(child)
    return exact + partial
