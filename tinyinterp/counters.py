"""Aggregate counters for tinyinterp calls."""

from __future__ import annotations

from threading import Lock

_COUNTER_FIELDS = (
    "calls",
    "forward_passes",
    "forward_time_ns",
    "hook_overhead_ns",
    "activations_captured",
    "activations_bytes",
    "buffer_pool_hits",
    "buffer_pool_misses",
    "maps_applied",
    "batch_groups",
    "batch_fusions",
    "prefix_layers_saved",
)


class _Counters:
    """Track cumulative tinyinterp work across calls."""

    calls: int
    forward_passes: int
    forward_time_ns: int
    hook_overhead_ns: int
    activations_captured: int
    activations_bytes: int
    buffer_pool_hits: int
    buffer_pool_misses: int
    maps_applied: int
    batch_groups: int
    batch_fusions: int
    prefix_layers_saved: int

    __slots__ = (*_COUNTER_FIELDS, "_lock")

    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""

        with self._lock:
            for field in _COUNTER_FIELDS:
                setattr(self, field, 0)

    def summary(self) -> str:
        """Return a readable multi-line summary of current counters."""

        with self._lock:
            lines = [
                "tinyinterp counters:",
                f"  calls:                 {self.calls}",
                f"  forward_passes:        {self.forward_passes}",
                f"  forward_time:          {self.forward_time_ns / 1e6:.3f}ms",
                f"  hook_overhead:         {self.hook_overhead_ns / 1e6:.3f}ms",
                f"  activations_captured:  {self.activations_captured}",
                f"  activations_bytes:     {self.activations_bytes}",
                f"  batch_groups:          {self.batch_groups}",
                f"  batch_fusions:         {self.batch_fusions}",
            ]
        return "\n".join(lines)


Counters = _Counters()
