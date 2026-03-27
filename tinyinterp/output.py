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
        *,
        path_to_sid: Mapping[str, int] | None = None,
        completed_forward: bool = True,
    ) -> None:
        self._model_output = model_output
        self.activations = dict(activations)
        self._id_to_sid = dict(id_to_sid)
        self._path_to_sid = dict(path_to_sid or {})
        self.completed_forward = completed_forward

    @property
    def partial(self) -> bool:
        return not self.completed_forward

    def __getattr__(self, name: str) -> Any:
        model_output = self._require_model_output()
        if isinstance(model_output, Mapping) and name in model_output:
            return model_output[name]
        return getattr(model_output, name)

    def __getitem__(self, key: Any) -> Any:
        module = getattr(key, "_module", None)
        if module is not None:
            sid = self._id_to_sid.get(id(module))
            if sid is None or sid not in self.activations:
                raise KeyError(f"No activation was captured for {key!r}.")
            return self.activations[sid]
        path = getattr(key, "path", None)
        if isinstance(path, str):
            sid = self._path_to_sid.get(path)
            if sid is None or sid not in self.activations:
                raise KeyError(f"No activation was captured for {key!r}.")
            return self.activations[sid]
        return self._require_model_output()[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._require_model_output())

    def __len__(self) -> int:
        return len(self._require_model_output())

    def _require_model_output(self) -> Any:
        if self.completed_forward:
            return self._model_output
        raise RuntimeError(
            "This output was captured with stop_at_last_get=True and has no final model output."
        )
