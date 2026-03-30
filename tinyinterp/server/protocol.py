"""Binary protocol helpers for tinyinterp remote execution."""

from __future__ import annotations

import enum
import io
import json
import socket
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import torch

PROTO_VERSION = 3
MAX_MESSAGE_TENSORS = 4096
MAX_META_BYTES = 64 << 20
MAX_TENSOR_NDIM = 16
MAX_DTYPE_NAME_BYTES = 64
MAX_TENSOR_BYTES = 8 << 30


class Cmd(enum.IntEnum):
    (
        TREE,
        COMPILE,
        CALL,
        CALL_MANY,
        GENERATE,
        GENERATE_MANY,
        FETCH_VALUE,
        RELEASE_VALUE,
        HELLO,
        CALL_GRAD,
        FETCH_GRAD_VALUE,
        FETCH_TARGET_GRAD,
        FETCH_INPUT_GRADS,
        BACKWARD,
        RELEASE_GRAD,
    ) = range(15)


HDR = "<BI"
HDR_SZ = struct.calcsize(HDR)


def send(sock: socket.socket, cmd: int, payload: bytes) -> None:
    sock.sendall(struct.pack(HDR, cmd, len(payload)) + payload)


def recv(sock: socket.socket) -> tuple[int, bytes]:
    hdr = _recvall(sock, HDR_SZ)
    cmd, length = struct.unpack(HDR, hdr)
    return cmd, _recvall(sock, length) if length > 0 else b""


def _recvall(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data += chunk
    return data


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    orig_dtype = tensor.dtype
    requires_grad = tensor.requires_grad
    tensor = tensor.detach().contiguous().cpu()
    if orig_dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    shape = tensor.shape
    dtype_str = str(orig_dtype).encode()
    raw = tensor.numpy().tobytes()
    hdr = struct.pack("<IIB", len(shape), len(dtype_str), int(requires_grad))
    shape_bytes = struct.pack(f"<{len(shape)}q", *shape)
    return hdr + dtype_str + shape_bytes + struct.pack("<Q", len(raw)) + raw


def deserialize_tensor(buf: io.BytesIO) -> torch.Tensor:
    ndim, dtype_len, requires_grad = struct.unpack("<IIB", _read_exact(buf, 9))
    if ndim > MAX_TENSOR_NDIM:
        raise ValueError(f"Remote tensor ndim {ndim} exceeds limit {MAX_TENSOR_NDIM}.")
    if dtype_len == 0 or dtype_len > MAX_DTYPE_NAME_BYTES:
        raise ValueError(f"Remote tensor dtype name length {dtype_len} is invalid.")
    dtype_str = _read_exact(buf, dtype_len).decode()
    shape = struct.unpack(f"<{ndim}q", _read_exact(buf, 8 * ndim))
    if any(dim < 0 for dim in shape):
        raise ValueError("Remote tensor shape cannot contain negative dimensions.")
    raw_len = struct.unpack("<Q", _read_exact(buf, 8))[0]
    if raw_len > MAX_TENSOR_BYTES:
        raise ValueError(f"Remote tensor payload {raw_len} exceeds limit {MAX_TENSOR_BYTES}.")
    element_size = _element_size_for_dtype(dtype_str)
    expected_raw_len = 0 if 0 in shape else element_size
    for dim in shape:
        if dim == 0:
            expected_raw_len = 0
            break
        if expected_raw_len > MAX_TENSOR_BYTES // dim:
            raise ValueError("Remote tensor shape exceeds payload limit.")
        expected_raw_len *= dim
    if raw_len != expected_raw_len:
        raise ValueError(
            f"Remote tensor payload size mismatch: got {raw_len}, expected {expected_raw_len}."
        )
    raw = _read_exact(buf, raw_len)
    if "bfloat16" in dtype_str:
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.int16).copy().reshape(shape)
        tensor = torch.from_numpy(arr).view(torch.bfloat16)
    else:
        dtype = getattr(torch, dtype_str.split(".")[-1])
        import numpy as np

        native = torch.zeros((), dtype=dtype).numpy().dtype
        tensor = torch.from_numpy(np.frombuffer(raw, dtype=native).copy().reshape(shape))
    if requires_grad and (tensor.is_floating_point() or tensor.is_complex()):
        tensor.requires_grad_(True)
    return tensor


def serialize_request(kwargs: Mapping[str, Any]) -> bytes:
    return _serialize_message(dict(kwargs))


def deserialize_request(data: bytes) -> dict[str, Any]:
    value = _deserialize_message(data)
    if not isinstance(value, dict):
        raise TypeError(f"Expected request mapping, got {type(value).__name__}.")
    return value


def serialize_response(result: Any) -> bytes:
    return _serialize_message(result)


def deserialize_response(data: bytes) -> Any:
    return _deserialize_message(data)


def build_tree(model: Any) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    for sid, (path, module) in enumerate(model.wrapped.named_modules()):
        nodes.append({"sid": sid, "path": path, "type": type(module).__name__})
    return nodes


def index_tree(
    tree: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[tuple[str, str, str]]], dict[str, int]]:
    types = {
        str(node["path"]): str(node["type"]) for node in tree if "path" in node and "type" in node
    }
    path_to_sid = {
        str(node["path"]): int(node["sid"]) for node in tree if "path" in node and "sid" in node
    }
    children: dict[str, list[tuple[str, str, str]]] = {}
    for path, type_name in types.items():
        if not path:
            continue
        parent, _, name = path.rpartition(".")
        children.setdefault(parent, []).append((name, path, type_name))
    for items in children.values():
        items.sort(
            key=lambda item: (
                not item[0].isdigit(),
                int(item[0]) if item[0].isdigit() else item[0],
            )
        )
    return types, children, path_to_sid


def server_capabilities(*, has_tokenizer: bool, grad: bool = False) -> dict[str, Any]:
    return {
        "backend": "remote",
        "remote": True,
        "grad": grad,
        "lazy_values": True,
        "request_tokenization": has_tokenizer,
        "protocol": PROTO_VERSION,
    }


def _serialize_message(value: Any) -> bytes:
    buf = io.BytesIO()
    tensors: list[torch.Tensor] = []
    encoded = _encode_value(value, tensors)
    meta = json.dumps(encoded, separators=(",", ":")).encode()
    buf.write(struct.pack("<I", len(tensors)))
    for tensor in tensors:
        buf.write(serialize_tensor(tensor))
    buf.write(struct.pack("<I", len(meta)))
    buf.write(meta)
    return buf.getvalue()


def _deserialize_message(data: bytes) -> Any:
    buf = io.BytesIO(data)
    tensor_count = struct.unpack("<I", _read_exact(buf, 4))[0]
    if tensor_count > MAX_MESSAGE_TENSORS:
        raise ValueError(f"Remote message tensor count {tensor_count} exceeds limit.")
    tensors = [deserialize_tensor(buf) for _ in range(tensor_count)]
    meta_len = struct.unpack("<I", _read_exact(buf, 4))[0]
    if meta_len > MAX_META_BYTES:
        raise ValueError(f"Remote message metadata {meta_len} exceeds limit {MAX_META_BYTES}.")
    return _decode_value(json.loads(_read_exact(buf, meta_len).decode()), tensors)


def _encode_value(value: Any, tensors: list[torch.Tensor]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        tensors.append(value)
        return {"__type__": "tensor", "index": len(tensors) - 1}
    if isinstance(value, list):
        return [_encode_value(item, tensors) for item in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(item, tensors) for item in value]}
    if isinstance(value, Mapping):
        return {
            "__type__": "dict",
            "items": [
                [_encode_value(key, tensors), _encode_value(item, tensors)]
                for key, item in value.items()
            ],
        }
    raise TypeError(f"Unsupported protocol value {type(value).__name__}.")


def _decode_value(value: Any, tensors: Sequence[torch.Tensor]) -> Any:
    if isinstance(value, list):
        return [_decode_value(item, tensors) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__type__")
    if tag == "tensor":
        return tensors[int(value["index"])]
    if tag == "tuple":
        return tuple(_decode_value(item, tensors) for item in value["items"])
    if tag == "dict":
        return {
            _decode_value(key, tensors): _decode_value(item, tensors)
            for key, item in value["items"]
        }
    return {str(key): _decode_value(item, tensors) for key, item in value.items()}


def tree_payload(model: Any) -> bytes:
    return json.dumps(build_tree(model)).encode()


def _read_exact(buf: io.BytesIO, n: int) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise ValueError("Truncated remote tensor payload.")
    return data


def _element_size_for_dtype(dtype_str: str) -> int:
    if "bfloat16" in dtype_str:
        return 2
    dtype_name = dtype_str.split(".")[-1]
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported remote tensor dtype {dtype_str!r}.")
    return torch.empty((), dtype=dtype).element_size()
