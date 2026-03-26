"""Output wrapper for model results plus captured activations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class Output:
    """Expose model outputs normally while allowing proxy-based activation lookup."""

    def __init__(
        self,
        model_output: Any,
        activations: Mapping[int, Any],
        id_to_sid: Mapping[int, int],
    ) -> None:
        self._model_output = model_output
        self.activations = dict(activations)
        self._id_to_sid = dict(id_to_sid)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_output, name)

    def __getitem__(self, key: Any) -> Any:
        module = getattr(key, "_module", None)
        if module is not None:
            sid = self._id_to_sid.get(id(module))
            if sid is None or sid not in self.activations:
                raise KeyError(f"No activation was captured for {key!r}.")
            return self.activations[sid]
        return self._model_output[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._model_output)

    def __len__(self) -> int:
        return len(self._model_output)
