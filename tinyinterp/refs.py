"""Lazy refs for values owned by remote runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch

from .output import GenerateOutput, Output

VALUE_REF_KIND = "value_ref"
GRAD_REF_KIND = "grad_ref"
_MISSING = object()


class RemoteValueRef:
    __slots__ = ("_value_id", "_fetch", "_release", "_cached", "_released")

    def __init__(
        self,
        value_id: str,
        *,
        fetch: Callable[[str], Any],
        release: Callable[[str], None],
    ) -> None:
        self._value_id = value_id
        self._fetch = fetch
        self._release = release
        self._cached: Any | None = None
        self._released = False

    def resolve(self) -> Any:
        if self._released and self._cached is None:
            raise RuntimeError("Remote value handle is already released.")
        if self._cached is None:
            self._cached = self._fetch(self._value_id)
        return self._cached

    def release(self) -> None:
        if self._released:
            return
        self._release(self._value_id)
        self._released = True

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "cached" if self._cached is not None else "pending"
        return f"RemoteValueRef({self._value_id!r}, {state})"


class _RemoteGradHandle:
    __slots__ = (
        "_grad_id",
        "_fetch_value",
        "_fetch_grad",
        "_fetch_input_grads",
        "_backward",
        "_release",
        "_refs",
        "_released",
        "_cached_input_grads",
    )

    def __init__(
        self,
        grad_id: str,
        *,
        fetch_value: Callable[[str, Any], Any],
        fetch_grad: Callable[[str, Any], Any],
        fetch_input_grads: Callable[[str], Any],
        backward: Callable[[str, Any, torch.Tensor | None], None],
        release: Callable[[str], None],
    ) -> None:
        self._grad_id = grad_id
        self._fetch_value = fetch_value
        self._fetch_grad = fetch_grad
        self._fetch_input_grads = fetch_input_grads
        self._backward = backward
        self._release = release
        self._refs = 0
        self._released = False
        self._cached_input_grads: Any = _MISSING

    def retain(self) -> None:
        if self._released:
            raise RuntimeError("Remote grad handle is already released.")
        self._refs += 1

    def release_ref(self) -> None:
        if self._released:
            return
        self._refs = max(self._refs - 1, 0)
        if self._refs == 0:
            self.release()

    def release(self) -> None:
        if self._released:
            return
        self._release(self._grad_id)
        self._released = True

    def fetch_value(self, target: Any) -> Any:
        if self._released:
            raise RuntimeError("Remote grad handle is already released.")
        return self._fetch_value(self._grad_id, target)

    def fetch_grad(self, target: Any) -> Any:
        if self._released:
            raise RuntimeError("Remote grad handle is already released.")
        return self._fetch_grad(self._grad_id, target)

    def input_grads(self) -> Any:
        if self._cached_input_grads is _MISSING:
            if self._released:
                raise RuntimeError("Remote grad handle is already released.")
            self._cached_input_grads = self._fetch_input_grads(self._grad_id)
        return self._cached_input_grads

    def backward(self, target: Any, gradient: torch.Tensor | None) -> None:
        if self._released:
            raise RuntimeError("Remote grad handle is already released.")
        self._backward(self._grad_id, target, gradient)


class RemoteGradValueRef:
    __slots__ = ("_handle", "_target", "_shape", "_dtype", "_cached", "_cached_grad")
    __array_priority__ = 1000

    def __init__(
        self,
        handle: _RemoteGradHandle,
        target: Any,
        *,
        shape: list[int] | None = None,
        dtype: str | None = None,
    ) -> None:
        self._handle = handle
        self._target = target
        self._shape = tuple(shape) if shape is not None else None
        self._dtype = dtype
        self._cached: Any = _MISSING
        self._cached_grad: Any = _MISSING
        self._handle.retain()

    def resolve(self) -> RemoteGradValueRef:
        return self

    def tensor(self) -> Any:
        if self._cached is _MISSING:
            self._cached = self._handle.fetch_value(self._target)
        return self._cached

    @property
    def grad(self) -> Any:
        if self._cached_grad is _MISSING:
            self._cached_grad = self._handle.fetch_grad(self._target)
        return self._cached_grad

    @property
    def input_grads(self) -> Any:
        return self._handle.input_grads()

    def backward(self, gradient: torch.Tensor | None = None) -> None:
        self._handle.backward(self._target, gradient)

    def release(self) -> None:
        self._handle.release_ref()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tensor(), name)

    def __getitem__(self, key: Any) -> Any:
        return self.tensor()[key]

    def __copy__(self) -> RemoteGradValueRef:
        return type(self)(
            self._handle,
            self._target,
            shape=list(self._shape) if self._shape is not None else None,
            dtype=self._dtype,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> RemoteGradValueRef:
        if id(self) in memo:
            return cast(RemoteGradValueRef, memo[id(self)])
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del cls, types
        return func(
            *_resolve_grad_refs(args),
            **_resolve_grad_refs(kwargs or {}),
        )

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._cached is not _MISSING:
            return repr(self._cached)
        shape = list(self._shape) if self._shape is not None else "?"
        return f"RemoteGradValueRef(target={self._target!r}, shape={shape}, dtype={self._dtype!r})"


class _RemoteInputGradRef:
    __slots__ = ("_handle",)

    def __init__(self, handle: _RemoteGradHandle) -> None:
        self._handle = handle
        self._handle.retain()

    def resolve(self) -> Any:
        return self._handle.input_grads()

    def release(self) -> None:
        self._handle.release_ref()

    def __copy__(self) -> _RemoteInputGradRef:
        return type(self)(self._handle)

    def __deepcopy__(self, memo: dict[int, Any]) -> _RemoteInputGradRef:
        if id(self) in memo:
            return cast(_RemoteInputGradRef, memo[id(self)])
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def value_ref_descriptor(value_id: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    return {
        "kind": VALUE_REF_KIND,
        "id": value_id,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def grad_ref_descriptor(grad_id: str, target: Any, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    return {
        "kind": GRAD_REF_KIND,
        "grad_id": grad_id,
        "target": target,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def decode_value_ref(
    value: Any,
    *,
    fetch: Callable[[str], Any],
    release: Callable[[str], None],
) -> Any:
    if isinstance(value, Mapping) and value.get("kind") == VALUE_REF_KIND:
        return RemoteValueRef(
            cast(str, value["id"]),
            fetch=fetch,
            release=release,
        )
    return value


def _decode_remote_value(
    value: Any,
    *,
    fetch: Callable[[str], Any],
    release: Callable[[str], None],
    grad_handle: _RemoteGradHandle | None,
) -> Any:
    if isinstance(value, Mapping) and value.get("kind") == GRAD_REF_KIND:
        if grad_handle is None:
            raise RuntimeError("Remote grad ref arrived without a grad handle.")
        return RemoteGradValueRef(
            grad_handle,
            value.get("target"),
            shape=cast(list[int] | None, value.get("shape")),
            dtype=cast(str | None, value.get("dtype")),
        )
    return decode_value_ref(value, fetch=fetch, release=release)


def output_from_remote_dict(
    data: Mapping[str, Any],
    *,
    fetch: Callable[[str], Any],
    release: Callable[[str], None],
    path_to_sid: Mapping[str, int],
    fetch_grad_value: Callable[[str, Any], Any] | None = None,
    fetch_target_grad: Callable[[str, Any], Any] | None = None,
    fetch_input_grads: Callable[[str], Any] | None = None,
    backward_grad: Callable[[str, Any, torch.Tensor | None], None] | None = None,
    release_grad: Callable[[str], None] | None = None,
) -> Output:
    grad_id = data.get("grad_id")
    grad_handle: _RemoteGradHandle | None = None
    if isinstance(grad_id, str):
        if (
            fetch_grad_value is None
            or fetch_target_grad is None
            or fetch_input_grads is None
            or backward_grad is None
            or release_grad is None
        ):
            raise RuntimeError("Remote grad output requires grad callbacks.")
        grad_handle = _RemoteGradHandle(
            grad_id,
            fetch_value=fetch_grad_value,
            fetch_grad=fetch_target_grad,
            fetch_input_grads=fetch_input_grads,
            backward=backward_grad,
            release=release_grad,
        )
    activations = {
        int(sid): _decode_remote_value(
            value,
            fetch=fetch,
            release=release,
            grad_handle=grad_handle,
        )
        for sid, value in cast(Mapping[Any, Any], data.get("activations", {})).items()
    }
    logits = _decode_remote_value(
        data.get("logits"),
        fetch=fetch,
        release=release,
        grad_handle=grad_handle,
    )
    if any(
        data.get(key) is not None
        for key in ("sequences", "generated_ids", "prompt_length", "generated_length")
    ):
        model_output = {
            "sequences": _decode_remote_value(
                data.get("sequences"),
                fetch=fetch,
                release=release,
                grad_handle=grad_handle,
            ),
            "generated_ids": _decode_remote_value(
                data.get("generated_ids"),
                fetch=fetch,
                release=release,
                grad_handle=grad_handle,
            ),
            "prompt_length": data.get("prompt_length"),
            "generated_length": data.get("generated_length"),
        }
        return GenerateOutput(
            model_output,
            activations,
            {},
            path_to_sid=path_to_sid,
            completed_forward=bool(data.get("completed_forward", True)),
        )
    model_output = {
        "logits": logits,
        "completed_forward": bool(data.get("completed_forward", True)),
    }
    if grad_handle is not None:
        model_output["input_grads"] = _RemoteInputGradRef(grad_handle)
    return Output(
        model_output,
        activations,
        {},
        path_to_sid=path_to_sid,
        completed_forward=bool(data.get("completed_forward", True)),
    )


def _resolve_grad_refs(value: Any) -> Any:
    if isinstance(value, RemoteGradValueRef):
        return value.tensor()
    if isinstance(value, tuple):
        return tuple(_resolve_grad_refs(item) for item in value)
    if isinstance(value, list):
        return [_resolve_grad_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_grad_refs(item) for key, item in value.items()}
    return value
