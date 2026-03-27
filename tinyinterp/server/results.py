"""Result objects returned by the inference server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class PlanResult:
    """Narrow server result for one executed plan."""

    activations: dict[str, Any] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    session_id: str | None = None
    prompt_length: int | None = None
    completed_forward: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
