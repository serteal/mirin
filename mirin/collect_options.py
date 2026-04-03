"""Collector option parsing for local model collection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COLLECTOR_OPTION_NAMES = frozenset(
    {
        "use_cache",
        "stop_at_last_get",
        "token_budget",
        "activation_budget_bytes",
        "activation_output",
        "pin_memory",
        "mmap_path",
    }
)


def split_collector_kwargs(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split collect kwargs into collector configuration and per-call kwargs."""

    collector_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in COLLECTOR_OPTION_NAMES:
            collector_kwargs[key] = value
        else:
            call_kwargs[key] = value
    return collector_kwargs, call_kwargs
