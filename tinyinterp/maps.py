"""Named map constructors for common activation interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class _Replace:
    value: Any

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x) + _coerce_value(self.value, x)


@dataclass(frozen=True, slots=True)
class _Add:
    delta: Any

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + _coerce_value(self.delta, x)


@dataclass(frozen=True, slots=True)
class _Scale:
    factor: Any

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * _coerce_value(self.factor, x)


@dataclass(frozen=True, slots=True)
class _Noise:
    std: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


@dataclass(frozen=True, slots=True)
class _Zero:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def replace(value: Any) -> _Replace:
    """Return a callable that replaces an activation with a fixed value."""

    return _Replace(value)


def add(delta: Any) -> _Add:
    """Return a callable that adds a value to an activation."""

    return _Add(delta)


def scale(factor: Any) -> _Scale:
    """Return a callable that scales an activation."""

    return _Scale(factor)


def zero() -> _Zero:
    """Return a callable that zeros an activation."""

    return _Zero()


def noise(std: float) -> _Noise:
    """Return a callable that adds Gaussian noise to an activation."""

    return _Noise(std)


def slice_head(tensor: torch.Tensor, head: int, n_heads: int) -> torch.Tensor:
    """Return one attention head slice from the final dimension."""

    d_head = tensor.shape[-1] // n_heads
    return tensor[..., head * d_head : (head + 1) * d_head]


def map_head(head: int, fn: Any, n_heads: int) -> Any:
    """Apply a transform to one attention head slice."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        d_head = x.shape[-1] // n_heads
        out = x.clone()
        out[..., head * d_head : (head + 1) * d_head] = fn(
            out[..., head * d_head : (head + 1) * d_head]
        )
        return out

    return transform


def _coerce_value(value: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)
