"""Benchmark model registry for real-model benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One real-model benchmark target."""

    model_name: str
    family: str
    size_label: str
    source: str
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL_SPECS = (
    ModelSpec(
        model_name="Qwen/Qwen3-1.7B",
        family="qwen3",
        size_label="1.7B",
        source="huggingface",
    ),
    ModelSpec(
        model_name="Qwen/Qwen3.5-4B",
        family="qwen3.5",
        size_label="4B",
        source="huggingface",
        notes="May require newer transformers support than the local environment provides.",
    ),
    ModelSpec(
        model_name="google/gemma-2-2b-it",
        family="gemma2",
        size_label="2B",
        source="huggingface",
    ),
    ModelSpec(
        model_name="google/gemma-3-4b-it",
        family="gemma3",
        size_label="4B",
        source="huggingface",
        notes="Run in text-only mode for this matrix.",
    ),
    ModelSpec(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        family="llama3.1",
        size_label="8B",
        source="huggingface",
    ),
)


def resolve_models(model_names: list[str] | None) -> list[ModelSpec]:
    """Return the default benchmark registry or a filtered subset."""

    if not model_names:
        return list(DEFAULT_MODEL_SPECS)
    by_name = {spec.model_name: spec for spec in DEFAULT_MODEL_SPECS}
    resolved: list[ModelSpec] = []
    for model_name in model_names:
        resolved.append(by_name.get(model_name, ModelSpec(model_name, "custom", "custom", "user")))
    return resolved
