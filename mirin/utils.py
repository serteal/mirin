"""Small exploration helpers for local module proxies."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch.nn as nn


class _ProxyLike(Protocol):
    _module: nn.Module


def find(proxy: _ProxyLike, pattern: str) -> Any | None:
    """Return the first direct child whose name or class matches ``pattern``."""
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
    _ensure_proxy(proxy)
    needle = pattern.lower()
    return [
        getattr(proxy, name)
        for name, child in proxy._module.named_children()
        if needle in name.lower() or needle in type(child).__name__.lower()
    ]


def children(proxy: _ProxyLike) -> list[tuple[str, str]]:
    """List direct child names and class names for exploration."""
    _ensure_proxy(proxy)
    return [(name, type(child).__name__) for name, child in proxy._module.named_children()]


def _ensure_proxy(proxy: object) -> _ProxyLike:
    module = getattr(proxy, "_module", None)
    if not isinstance(module, nn.Module):
        raise TypeError("Expected a mirin module proxy.")
    return cast(_ProxyLike, proxy)
