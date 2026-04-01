"""Unit tests for low-level hook helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mirin.hooks import _replace


def test_replace_rejects_unknown_output_structures() -> None:
    with pytest.raises(TypeError, match="Cannot replace tensor"):
        _replace(SimpleNamespace(hidden=torch.ones(1, 2, 3)), torch.zeros(1, 2, 3))
