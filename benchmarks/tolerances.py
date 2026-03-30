"""Shared numeric tolerances for benchmark correctness checks."""

from __future__ import annotations

from typing import Literal

import torch

ComparisonMode = Literal["same_impl", "cross_impl"]


def comparison_tolerances(
    tensor: torch.Tensor,
    *,
    mode: ComparisonMode,
) -> tuple[float, float]:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        if mode == "cross_impl":
            return (3e-1, 1e-1)
        return (1e-2, 1e-2)
    return (1e-5, 1e-5)


def compare_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    mode: ComparisonMode,
) -> dict[str, float | bool]:
    atol, rtol = comparison_tolerances(left, mode=mode)
    return {
        "ok": torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol),
        "max_abs_diff": float((left.float() - right.float()).abs().max().item()),
    }
