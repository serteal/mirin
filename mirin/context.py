"""Scoped configuration for mirin."""

from __future__ import annotations

import os
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Literal


def _parse_debug(value: str) -> int:
    """Parse a debug level from an environment variable."""

    try:
        return int(value)
    except ValueError:
        return 0


def _parse_graph(value: str) -> str | None:
    """Parse a graph output path from an environment variable."""

    stripped = value.strip()
    if not stripped or stripped == "0":
        return None
    if stripped == "1":
        return "/tmp/graph.svg"
    return stripped


_DEFAULT_DEBUG = _parse_debug(os.environ.get("DEBUG", "0"))
_DEFAULT_GRAPH = _parse_graph(os.environ.get("GRAPH", "0"))
_DEBUG_OVERRIDE: ContextVar[int | None] = ContextVar("debug_override", default=None)
_GRAPH_OVERRIDE: ContextVar[str | None | bool] = ContextVar("graph_override", default=None)


def get_debug() -> int:
    """Return the active debug level for the current context."""

    override = _DEBUG_OVERRIDE.get()
    return _DEFAULT_DEBUG if override is None else override


def get_graph_path() -> str | None:
    """Return the active graph output path, if graph rendering is enabled."""

    override = _GRAPH_OVERRIDE.get()
    if override is None:
        return _DEFAULT_GRAPH
    if override is False:
        return None
    if override is True:
        return "/tmp/graph.svg"
    return override


class context(ContextDecorator):
    """Temporarily override mirin configuration inside a block."""

    def __init__(
        self,
        *,
        debug: int | None = None,
        graph: str | os.PathLike[str] | bool | None = None,
    ) -> None:
        self._debug = debug
        if isinstance(graph, Path):
            self._graph: str | bool | None = str(graph)
        elif isinstance(graph, os.PathLike):
            self._graph = os.fspath(graph)
        else:
            self._graph = graph
        self._debug_token: Token[int | None] | None = None
        self._graph_token: Token[str | None | bool] | None = None

    def __enter__(self) -> context:
        if self._debug is not None:
            self._debug_token = _DEBUG_OVERRIDE.set(self._debug)
        if self._graph is not None:
            self._graph_token = _GRAPH_OVERRIDE.set(self._graph)
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        if self._debug_token is not None:
            _DEBUG_OVERRIDE.reset(self._debug_token)
            self._debug_token = None
        if self._graph_token is not None:
            _GRAPH_OVERRIDE.reset(self._graph_token)
            self._graph_token = None
        return False
