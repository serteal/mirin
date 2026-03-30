"""Unit tests for lazy remote refs."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from tinyinterp.refs import (
    RemoteGradValueRef,
    RemoteValueRef,
    _RemoteGradHandle,
    grad_ref_descriptor,
    output_from_remote_dict,
    value_ref_descriptor,
)


def test_remote_value_ref_fetches_once_and_releases_after_resolve() -> None:
    fetched: list[str] = []
    released: list[str] = []

    def fetch(value_id: str) -> Any:
        fetched.append(value_id)
        return torch.tensor([1.0])

    def release(value_id: str) -> None:
        released.append(value_id)

    ref = RemoteValueRef("abc", fetch=fetch, release=release)

    first = ref.resolve()
    second = ref.resolve()
    ref.release()

    assert torch.equal(first, torch.tensor([1.0]))
    assert torch.equal(second, torch.tensor([1.0]))
    assert fetched == ["abc"]
    assert released == ["abc"]


def test_remote_value_ref_release_before_resolve_disables_fetch() -> None:
    released: list[str] = []

    def fetch(_value_id: str) -> Any:
        raise AssertionError("fetch should not run after release()")

    def release(value_id: str) -> None:
        released.append(value_id)

    ref = RemoteValueRef("def", fetch=fetch, release=release)
    ref.release()

    assert released == ["def"]
    with pytest.raises(RuntimeError, match="already released"):
        ref.resolve()


def test_output_from_remote_dict_resolves_site_and_logits() -> None:
    fetched: list[str] = []
    released: list[str] = []

    def fetch(value_id: str) -> Any:
        fetched.append(value_id)
        if value_id == "act":
            return torch.ones(1, 2, 3)
        if value_id == "logits":
            return torch.zeros(1, 2, 5)
        raise KeyError(value_id)

    def release(value_id: str) -> None:
        released.append(value_id)

    output = output_from_remote_dict(
        {
            "activations": {7: value_ref_descriptor("act", torch.ones(1, 2, 3))},
            "logits": value_ref_descriptor("logits", torch.zeros(1, 2, 5)),
            "completed_forward": True,
        },
        fetch=fetch,
        release=release,
        path_to_sid={"transformer.h.0": 7},
    )

    assert torch.equal(output["transformer.h.0"], torch.ones(1, 2, 3))
    assert torch.equal(output.logits, torch.zeros(1, 2, 5))
    assert fetched == ["act", "logits"]
    output.release()
    assert sorted(released) == ["act", "logits"]


def test_output_from_remote_dict_grad_refs_support_backward_and_input_grads() -> None:
    backward_calls: list[tuple[str, Any, torch.Tensor | None]] = []

    output = output_from_remote_dict(
        {
            "grad_id": "grad-1",
            "activations": {7: grad_ref_descriptor("grad-1", "7", torch.ones(1, 2, 3))},
            "logits": grad_ref_descriptor("grad-1", "logits", torch.zeros(1, 2, 5)),
            "completed_forward": True,
        },
        fetch=lambda _value_id: None,
        release=lambda _value_id: None,
        path_to_sid={"transformer.h.0": 7},
        fetch_grad_value=lambda _grad_id, target: (
            torch.ones(1, 2, 3) if target == "7" else torch.zeros(1, 2, 5)
        ),
        fetch_target_grad=lambda _grad_id, target: (
            torch.full((1, 2, 3), 0.5) if target == "7" else None
        ),
        fetch_input_grads=lambda _grad_id: {"x": torch.full((1, 4), 0.25)},
        backward_grad=lambda grad_id, target, gradient: backward_calls.append(
            (grad_id, target, gradient)
        ),
        release_grad=lambda _grad_id: None,
    )

    act = output["transformer.h.0"]
    upstream = torch.full_like(act, 0.25)
    act.backward(upstream)

    assert torch.equal(act, torch.ones(1, 2, 3))
    assert torch.equal(act.grad, torch.full((1, 2, 3), 0.5))
    assert torch.equal(output.input_grads["x"], torch.full((1, 4), 0.25))
    assert backward_calls == [("grad-1", "7", upstream)]


def test_remote_grad_value_ref_copy_and_deepcopy_retain_handle() -> None:
    released: list[str] = []
    handle = _RemoteGradHandle(
        "grad-copy",
        fetch_value=lambda _grad_id, _target: torch.ones(1, 2, 3),
        fetch_grad=lambda _grad_id, _target: None,
        fetch_input_grads=lambda _grad_id: {},
        backward=lambda _grad_id, _target, _gradient: None,
        release=lambda grad_id: released.append(grad_id),
    )
    ref = RemoteGradValueRef(handle, "site")
    clone = copy.copy(ref)
    deep = copy.deepcopy(ref)

    ref.release()
    assert released == []
    clone.release()
    assert released == []
    deep.release()
    assert released == ["grad-copy"]
