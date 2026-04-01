"""Unit tests for remote protocol framing and validation."""

from __future__ import annotations

import io
import struct

import pytest
import torch

from mirin.server.protocol import (
    MAX_DTYPE_NAME_BYTES,
    MAX_TENSOR_BYTES,
    MAX_TENSOR_NDIM,
    deserialize_request,
    deserialize_response,
    deserialize_tensor,
    serialize_request,
    serialize_response,
    serialize_tensor,
)


def test_tensor_roundtrip_preserves_shape_dtype_and_requires_grad() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_(True)

    restored = deserialize_tensor(io.BytesIO(serialize_tensor(value)))

    assert torch.equal(restored, value.detach())
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.requires_grad


def test_request_and_response_roundtrip_support_tensors() -> None:
    request = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long), "flag": True}
    response = {"logits": torch.zeros((1, 3, 5)), "ok": True}

    assert torch.equal(
        deserialize_request(serialize_request(request))["input_ids"],
        request["input_ids"],
    )
    decoded = deserialize_response(serialize_response(response))
    assert torch.equal(decoded["logits"], response["logits"])
    assert decoded["ok"] is True


def test_deserialize_tensor_rejects_excessive_ndim() -> None:
    payload = struct.pack("<IIB", MAX_TENSOR_NDIM + 1, len(b"torch.float32"), 0)

    with pytest.raises(ValueError, match="ndim"):
        deserialize_tensor(io.BytesIO(payload))


def test_deserialize_tensor_rejects_excessive_dtype_name_length() -> None:
    payload = struct.pack("<IIB", 1, MAX_DTYPE_NAME_BYTES + 1, 0)

    with pytest.raises(ValueError, match="dtype name length"):
        deserialize_tensor(io.BytesIO(payload))


def test_deserialize_tensor_rejects_excessive_raw_len() -> None:
    dtype = b"torch.float32"
    payload = b"".join(
        [
            struct.pack("<IIB", 1, len(dtype), 0),
            dtype,
            struct.pack("<1q", 1),
            struct.pack("<Q", MAX_TENSOR_BYTES + 1),
        ]
    )

    with pytest.raises(ValueError, match="payload"):
        deserialize_tensor(io.BytesIO(payload))


def test_deserialize_tensor_rejects_truncated_payload() -> None:
    dtype = b"torch.float32"
    payload = b"".join(
        [
            struct.pack("<IIB", 1, len(dtype), 0),
            dtype,
            struct.pack("<1q", 2),
            struct.pack("<Q", 8),
            b"\x00\x00\x00\x00",
        ]
    )

    with pytest.raises(ValueError, match="Truncated"):
        deserialize_tensor(io.BytesIO(payload))
