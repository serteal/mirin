"""Output wrapper for model results plus captured activations."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
from operator import index
from typing import Any

import torch


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
        self._activations = dict(activations)
        self._id_to_sid = dict(id_to_sid)
        self._path_to_sid = dict(path_to_sid or {})
        self.completed_forward = completed_forward

    @property
    def partial(self) -> bool:
        return not self.completed_forward

    @property
    def activations(self) -> _ActivationView:
        return _ActivationView(self)

    def __getattr__(self, name: str) -> Any:
        model_output = self._require_model_output()
        if isinstance(model_output, Mapping) and name in model_output:
            return _resolve_value(model_output[name])
        return _resolve_value(getattr(model_output, name))

    def __getitem__(self, key: Any) -> Any:
        module = getattr(key, "_module", None)
        if module is not None:
            sid = self._id_to_sid.get(id(module))
            if sid is not None and sid in self._activations:
                return _resolve_value(self._activations[sid])
        if isinstance(key, str):
            sid = self._path_to_sid.get(key)
            if sid is not None and sid in self._activations:
                return _resolve_value(self._activations[sid])
        path = getattr(key, "path", None) or getattr(key, "_path", None)
        if isinstance(path, str):
            sid = self._path_to_sid.get(path)
            if sid is not None and sid in self._activations:
                return _resolve_value(self._activations[sid])
        if key in self._activations:
            return _resolve_value(self._activations[key])
        return _resolve_value(self._require_model_output()[key])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._require_model_output())

    def __len__(self) -> int:
        return len(self._require_model_output())

    def release(self) -> None:
        _release_value(self._model_output)
        _release_value(self._activations)

    def _require_model_output(self) -> Any:
        if self.completed_forward:
            return self._model_output
        raise RuntimeError(
            "This output was captured with stop_at_last_get=True and has no final model output."
        )


class GenerateOutput(Output):
    """Generation output plus optional captured activations."""

    @property
    def sequences(self) -> Any:
        return _resolve_value(self._require_model_output()["sequences"])

    @property
    def generated_ids(self) -> Any:
        return _resolve_value(self._require_model_output()["generated_ids"])

    @property
    def prompt_length(self) -> Any:
        return self._require_model_output()["prompt_length"]

    @property
    def generated_length(self) -> Any:
        return self._require_model_output()["generated_length"]


class _ActivationView(Mapping[Any, Any]):
    """Read-only activation mapping that resolves lazy values on access."""

    def __init__(self, output: Output) -> None:
        self._output = output

    def __getitem__(self, key: Any) -> Any:
        return _resolve_value(self._output._activations[key])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._output._activations)

    def __len__(self) -> int:
        return len(self._output._activations)

    def keys(self) -> KeysView[Any]:
        return self._output._activations.keys()

    def items(self) -> ItemsView[Any, Any]:
        return ItemsView(_ResolvedActivationMapping(self._output))

    def values(self) -> ValuesView[Any]:
        return ValuesView(_ResolvedActivationMapping(self._output))


class _ResolvedActivationMapping(Mapping[Any, Any]):
    def __init__(self, output: Output) -> None:
        self._output = output

    def __getitem__(self, key: Any) -> Any:
        return self._output.activations[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._output._activations)

    def __len__(self) -> int:
        return len(self._output._activations)


def _resolve_value(value: Any) -> Any:
    resolver = getattr(value, "resolve", None)
    if callable(resolver):
        return resolver()
    return value


def _release_value(value: Any) -> None:
    releaser = getattr(value, "release", None)
    if callable(releaser):
        releaser()
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _release_value(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _release_value(item)


def output_from_path_activations(
    model_output: Any,
    activations: Mapping[str, Any],
    *,
    completed_forward: bool = True,
) -> Output:
    """Build an ``Output`` whose activations are keyed by module path."""

    return Output(
        model_output,
        activations,
        {},
        path_to_sid={path: path for path in activations},
        completed_forward=completed_forward,
    )


def generate_output_from_path_activations(
    sequences: Any,
    generated_ids: Any,
    activations: Mapping[str, Any],
    *,
    prompt_length: int | list[int] | None,
    generated_length: int | list[int] | None,
    completed_forward: bool = True,
) -> GenerateOutput:
    """Build a ``GenerateOutput`` whose activations are keyed by module path."""

    return GenerateOutput(
        {
            "sequences": sequences,
            "generated_ids": generated_ids,
            "prompt_length": prompt_length,
            "generated_length": generated_length,
        },
        activations,
        {},
        path_to_sid={path: path for path in activations},
        completed_forward=completed_forward,
    )


def generate_output_from_value(
    value: Any,
    *,
    prompt_length: int,
) -> GenerateOutput:
    """Coerce a generate result into the public ``GenerateOutput`` contract."""

    if isinstance(value, GenerateOutput):
        return value
    if isinstance(value, Output):
        sequences = _resolve_value(value._model_output)
        if not isinstance(sequences, torch.Tensor):
            raise TypeError(
                "model.generate(...) must return token tensors or GenerateOutput-compatible values."
            )
        generated_length = int(sequences.shape[-1]) - prompt_length
        return GenerateOutput(
            {
                "sequences": sequences,
                "generated_ids": sequences[:, prompt_length:],
                "prompt_length": prompt_length,
                "generated_length": generated_length,
            },
            value._activations,
            value._id_to_sid,
            path_to_sid=value._path_to_sid,
            completed_forward=value.completed_forward,
        )
    if isinstance(value, torch.Tensor):
        generated_length = int(value.shape[-1]) - prompt_length
        return generate_output_from_path_activations(
            value,
            value[:, prompt_length:],
            {},
            prompt_length=prompt_length,
            generated_length=generated_length,
        )
    raise TypeError(
        "model.generate(...) must return token tensors or GenerateOutput-compatible values."
    )


def merge_generate_outputs(outputs: Sequence[GenerateOutput]) -> GenerateOutput:
    """Merge one or more generate outputs into a single batched ``GenerateOutput``."""

    if not outputs:
        raise ValueError("Expected at least one GenerateOutput.")
    if len(outputs) == 1:
        return outputs[0]
    sequences = [_require_batched_tensor(output.sequences, name="sequences") for output in outputs]
    generated_ids = [
        _require_batched_tensor(output.generated_ids, name="generated_ids") for output in outputs
    ]
    counts = [int(sequence.shape[0]) for sequence in sequences]
    prompt_lengths = [
        length
        for output, count in zip(outputs, counts, strict=True)
        for length in _lengths_from_field(output.prompt_length, count, name="prompt_length")
    ]
    generated_lengths = [
        length
        for output, count in zip(outputs, counts, strict=True)
        for length in _lengths_from_field(output.generated_length, count, name="generated_length")
    ]
    activation_keys = set(outputs[0]._activations)
    if any(set(output._activations) != activation_keys for output in outputs[1:]):
        raise ValueError("All merged GenerateOutput values must expose the same activation keys.")
    merged_activations = {
        key: _pad_and_concat(
            [
                _require_batched_tensor(output._activations[key], name=f"activation {key!r}")
                for output in outputs
            ]
        )
        for key in activation_keys
    }
    id_to_sid: dict[int, int] = {}
    path_to_sid: dict[str, int] = {}
    for output in outputs:
        id_to_sid.update(output._id_to_sid)
        path_to_sid.update(output._path_to_sid)
    return GenerateOutput(
        {
            "sequences": _pad_and_concat(sequences),
            "generated_ids": _pad_and_concat(generated_ids),
            "prompt_length": prompt_lengths,
            "generated_length": generated_lengths,
        },
        merged_activations,
        id_to_sid,
        path_to_sid=path_to_sid,
        completed_forward=all(output.completed_forward for output in outputs),
    )


def _require_batched_tensor(value: Any, *, name: str) -> torch.Tensor:
    resolved = _resolve_value(value)
    if not isinstance(resolved, torch.Tensor) or resolved.ndim < 2:
        raise TypeError(f"Expected batched tensor {name}, got {type(resolved).__name__}.")
    return resolved


def _lengths_from_field(value: Any, batch_size: int, *, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * batch_size
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        lengths = [_coerce_length(item, name=name) for item in value]
        if len(lengths) != batch_size:
            raise ValueError(
                f"{name} length {len(lengths)} did not match batch size {batch_size}."
            )
        return lengths
    raise TypeError(f"Expected {name} to be an int or sequence of ints.")


def _coerce_length(value: Any, *, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"Expected {name} to contain only integers.") from exc


def _pad_and_concat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("Expected at least one tensor to merge.")
    if any(value.ndim != values[0].ndim for value in values[1:]):
        raise ValueError("Expected tensors with matching rank.")
    if values[0].ndim < 2:
        return torch.cat(list(values), dim=0)
    trailing = values[0].shape[2:]
    if any(value.shape[2:] != trailing for value in values[1:]):
        raise ValueError("Expected tensors with matching non-time dimensions.")
    max_width = max(int(value.shape[1]) for value in values)
    padded: list[torch.Tensor] = []
    for value in values:
        if int(value.shape[1]) == max_width:
            padded.append(value)
            continue
        item = value.new_zeros((value.shape[0], max_width, *value.shape[2:]))
        item[:, : value.shape[1]] = value
        padded.append(item)
    return torch.cat(padded, dim=0)
