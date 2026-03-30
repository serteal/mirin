"""Persistent server and remote client for tinyinterp."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from .. import maps as maps_mod
from ..output import GenerateOutput
from ..refs import grad_ref_descriptor, output_from_remote_dict, value_ref_descriptor
from ..requests import request_items
from .protocol import (
    PROTO_VERSION,
    Cmd,
    build_tree,
    deserialize_request,
    deserialize_response,
    index_tree,
    recv,
    send,
    serialize_request,
    serialize_response,
    server_capabilities,
    tree_payload,
)
from .results import PlanResult

logger = logging.getLogger(__name__)

# --- server ---


def serve(server: Any, sock_path: str) -> None:
    """Listen on a Unix socket and dispatch requests to *server*. Blocks forever."""
    server._serve_shutdown.clear()
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    ssock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ssock.bind(sock_path)
    ssock.listen(8)
    ssock.settimeout(0.2)
    tree = build_tree(server._model)
    tree_json = tree_payload(server._model)
    sid_to_path = {int(node["sid"]): str(node["path"]) for node in tree}
    compile_cache: dict[bytes, str] = {}
    server._listen_socket = ssock
    server._listen_path = sock_path
    print(f"tinyinterp server listening on {sock_path}")
    try:
        while not server._serve_shutdown.is_set():
            try:
                conn, _ = ssock.accept()
            except TimeoutError:
                continue
            except OSError:
                if server._serve_shutdown.is_set():
                    break
                raise
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 << 20)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
            with server._client_lock:
                server._client_sockets.add(conn)
            t = threading.Thread(
                target=_handle_client,
                args=(server, conn, tree_json, sid_to_path, compile_cache),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        if server._listen_socket is ssock:
            server._listen_socket = None
        if server._listen_path == sock_path:
            server._listen_path = None
        try:
            ssock.close()
        except OSError:
            pass
        if os.path.exists(sock_path):
            os.unlink(sock_path)


def _handle_client(
    server: Any,
    conn: socket.socket,
    tree_json: bytes,
    sid_to_path: Mapping[int, str],
    compile_cache: dict[bytes, str],
) -> None:
    path_to_sid = {path: sid for sid, path in sid_to_path.items()}
    values: dict[str, Any] = {}
    grad_ids: set[str] = set()
    try:
        while True:
            cmd, payload = recv(conn)
            if cmd == Cmd.HELLO:
                send(
                    conn,
                    Cmd.HELLO,
                    serialize_response(
                        server_capabilities(
                            has_tokenizer=getattr(server._model, "tokenizer", None) is not None,
                            grad=True,
                        )
                    ),
                )
            elif cmd == Cmd.TREE:
                send(conn, Cmd.TREE, tree_json)
            elif cmd == Cmd.COMPILE:
                plan_id = compile_cache.get(payload)
                if plan_id is None:
                    request = deserialize_request(payload)
                    get_sids = cast(list[int] | None, request.get("get_sids"))
                    get = [sid_to_path[sid] for sid in get_sids] if get_sids else None
                    raw_specs = cast(list[dict[str, Any]] | None, request.get("map_specs"))
                    mapping = None
                    if raw_specs:
                        mapping = {
                            sid_to_path[int(spec["sid"])]: _decode_remote_map_spec(spec)
                            for spec in raw_specs
                        }
                    plan = server.compile(get=get, mapping=mapping, output=request.get("output"))
                    plan_id = plan.id
                    compile_cache[payload] = plan_id
                send(conn, Cmd.COMPILE, serialize_response(plan_id))
            elif cmd == Cmd.CALL:
                kwargs = deserialize_request(payload)
                plan_id = cast(str, kwargs.pop("_plan_id"))
                stop_at_last_get = bool(kwargs.pop("_stop_at_last_get", False))
                plan = server._resolve_plan(plan_id)
                if stop_at_last_get:
                    collector = server.open_collector(plan=plan, stop_at_last_get=True)
                    try:
                        result = server.collect_batch(collector, kwargs)
                    finally:
                        server.close_collector(collector)
                else:
                    result = server.call(plan, **kwargs)
                send(
                    conn,
                    Cmd.CALL,
                    serialize_response(_plan_result_to_wire(server, values, result, path_to_sid)),
                )
            elif cmd == Cmd.CALL_GRAD:
                kwargs = deserialize_request(payload)
                plan_id = cast(str, kwargs.pop("_plan_id"))
                grad_id, result = server.call_grad(plan_id, **kwargs)
                grad_ids.add(grad_id)
                send(
                    conn,
                    Cmd.CALL_GRAD,
                    serialize_response(
                        _plan_result_to_wire(server, values, result, path_to_sid, grad_id=grad_id)
                    ),
                )
            elif cmd == Cmd.CALL_MANY:
                kwargs = deserialize_request(payload)
                prompts = kwargs.pop("_prompts")
                plan_id = cast(str, kwargs.pop("_plan_id"))
                stop_at_last_get = bool(kwargs.pop("_stop_at_last_get", False))
                plan = server._resolve_plan(plan_id)
                if stop_at_last_get:
                    collector = server.open_collector(plan=plan, stop_at_last_get=True)
                    try:
                        results = server.collect_many(collector, prompts, **kwargs)
                    finally:
                        server.close_collector(collector)
                else:
                    results = server.call_many(prompts, plan=plan, **kwargs)
                send(
                    conn,
                    Cmd.CALL_MANY,
                    serialize_response(
                        [_plan_result_to_wire(server, values, r, path_to_sid) for r in results]
                    ),
                )
            elif cmd == Cmd.GENERATE:
                kwargs = deserialize_request(payload)
                plan = server._resolve_plan(cast(str, kwargs.pop("_plan_id")))
                result = server.generate(plan=plan, **kwargs)
                payload_value = (
                    _plan_result_to_wire(server, values, result, path_to_sid)
                    if isinstance(result, PlanResult)
                    else _generate_output_to_wire(server, values, result, path_to_sid)
                    if isinstance(result, GenerateOutput)
                    else _generate_tensor_to_wire(
                        server,
                        values,
                        result,
                        prompt_length=_tensor_prompt_length(kwargs),
                    )
                    if isinstance(result, torch.Tensor)
                    else result
                )
                send(conn, Cmd.GENERATE, serialize_response(payload_value))
            elif cmd == Cmd.GENERATE_MANY:
                kwargs = deserialize_request(payload)
                prompts = kwargs.pop("_prompts")
                plan = server._resolve_plan(cast(str, kwargs.pop("_plan_id")))
                results = server.generate_many(prompts, plan=plan, **kwargs)
                payload_value = (
                    _plan_result_to_wire(server, values, results, path_to_sid)
                    if isinstance(results, PlanResult)
                    else _generate_output_to_wire(server, values, results, path_to_sid)
                    if isinstance(results, GenerateOutput)
                    else results
                )
                send(conn, Cmd.GENERATE_MANY, serialize_response(payload_value))
            elif cmd == Cmd.FETCH_VALUE:
                request = deserialize_request(payload)
                value_id = cast(str, request["value_id"])
                value = values.pop(value_id, None)
                if value is None:
                    send(
                        conn,
                        Cmd.FETCH_VALUE,
                        serialize_response(
                            {"__remote_error__": f"Unknown value handle {value_id!r}."}
                        ),
                    )
                    continue
                _adjust_remote_value_count(server, -1)
                send(conn, Cmd.FETCH_VALUE, serialize_response(value))
            elif cmd == Cmd.RELEASE_VALUE:
                request = deserialize_request(payload)
                value_id = cast(str, request["value_id"])
                if values.pop(value_id, None) is not None:
                    _adjust_remote_value_count(server, -1)
                send(conn, Cmd.RELEASE_VALUE, serialize_response(True))
            elif cmd == Cmd.FETCH_GRAD_VALUE:
                request = deserialize_request(payload)
                send(
                    conn,
                    Cmd.FETCH_GRAD_VALUE,
                    serialize_response(
                        server.fetch_grad_value(
                            cast(str, request["grad_id"]), cast(str | int, request["target"])
                        )
                    ),
                )
            elif cmd == Cmd.FETCH_TARGET_GRAD:
                request = deserialize_request(payload)
                send(
                    conn,
                    Cmd.FETCH_TARGET_GRAD,
                    serialize_response(
                        server.fetch_target_grad(
                            cast(str, request["grad_id"]), cast(str | int, request["target"])
                        )
                    ),
                )
            elif cmd == Cmd.FETCH_INPUT_GRADS:
                request = deserialize_request(payload)
                send(
                    conn,
                    Cmd.FETCH_INPUT_GRADS,
                    serialize_response(server.fetch_input_grads(cast(str, request["grad_id"]))),
                )
            elif cmd == Cmd.BACKWARD:
                request = deserialize_request(payload)
                server.backward_grad(
                    cast(str, request["grad_id"]),
                    cast(str | int, request["target"]),
                    cast(torch.Tensor | None, request.get("gradient")),
                )
                send(conn, Cmd.BACKWARD, serialize_response(True))
            elif cmd == Cmd.RELEASE_GRAD:
                request = deserialize_request(payload)
                grad_id = cast(str, request["grad_id"])
                released = server.release_grad(grad_id)
                grad_ids.discard(grad_id)
                send(conn, Cmd.RELEASE_GRAD, serialize_response(released))
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            _adjust_remote_value_count(server, -len(values))
            values.clear()
            for grad_id in list(grad_ids):
                try:
                    server.release_grad(grad_id)
                finally:
                    grad_ids.discard(grad_id)
            with server._client_lock:
                server._client_sockets.discard(conn)
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _plan_result_to_wire(
    server: Any,
    values: dict[str, Any],
    result: Any,
    path_to_sid: Mapping[str, int],
    *,
    grad_id: str | None = None,
) -> dict[str, Any]:
    descriptor = (
        (lambda target, value: grad_ref_descriptor(cast(str, grad_id), target, value))
        if grad_id is not None
        else (lambda target, value: _remote_value_descriptor(server, values, value))
    )
    payload = {
        "activations": {
            path_to_sid[path]: descriptor(
                path if grad_id is not None else str(path_to_sid[path]), value
            )
            for path, value in result.activations.items()
        },
        "logits": descriptor("logits", result.logits),
        "completed_forward": result.completed_forward,
        "grad_id": grad_id,
    }
    if result.sequences is not None or result.token_ids is not None:
        payload["sequences"] = descriptor("sequences", result.sequences)
        payload["generated_ids"] = descriptor("generated_ids", result.token_ids)
        prompt_length = (
            result.prompt_length
            if result.prompt_length is not None
            else result.metadata.get("prompt_lengths")
        )
        generated_length = (
            result.metadata.get("generated_length")
            if result.metadata.get("generated_length") is not None
            else result.metadata.get("generated_lengths")
        )
        payload["prompt_length"] = prompt_length
        payload["generated_length"] = generated_length
    return payload


def _remote_value_descriptor(server: Any, values: dict[str, Any], value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    value_id = uuid.uuid4().hex
    values[value_id] = value
    _adjust_remote_value_count(server, 1)
    return value_ref_descriptor(value_id, value)


def _adjust_remote_value_count(server: Any, delta: int) -> None:
    with server._remote_value_lock:
        server._remote_value_count = max(server._remote_value_count + delta, 0)


def _decode_remote_map_spec(spec: Mapping[str, Any]) -> Any:
    op = str(spec["op"])
    if op == "zero":
        return maps_mod.zero()
    if op == "add":
        return maps_mod.add(spec["value"])
    if op == "scale":
        return maps_mod.scale(spec["value"])
    if op == "replace":
        return maps_mod.replace(spec["value"])
    raise TypeError(f"Unknown remote map op {op!r}.")


def _generate_output_to_wire(
    server: Any,
    values: dict[str, Any],
    result: GenerateOutput,
    path_to_sid: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "activations": {
            path_to_sid[path]: _remote_value_descriptor(server, values, value)
            for path, value in result._activations.items()
        },
        "logits": None,
        "sequences": _remote_value_descriptor(server, values, result._model_output["sequences"]),
        "generated_ids": _remote_value_descriptor(
            server,
            values,
            result._model_output["generated_ids"],
        ),
        "prompt_length": result._model_output["prompt_length"],
        "generated_length": result._model_output["generated_length"],
        "completed_forward": result.completed_forward,
        "grad_id": None,
    }


def _generate_tensor_to_wire(
    server: Any,
    values: dict[str, Any],
    sequence: torch.Tensor,
    *,
    prompt_length: int,
) -> dict[str, Any]:
    return {
        "activations": {},
        "logits": None,
        "sequences": _remote_value_descriptor(server, values, sequence),
        "generated_ids": _remote_value_descriptor(server, values, sequence[:, prompt_length:]),
        "prompt_length": prompt_length,
        "generated_length": int(sequence.shape[-1]) - prompt_length,
        "completed_forward": True,
        "grad_id": None,
    }


def _tensor_prompt_length(kwargs: Mapping[str, Any]) -> int:
    input_ids = kwargs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("Remote generate expected tensor input_ids.")
    return int(input_ids.shape[-1])


# --- client ---


class _RemoteProxy:
    """Lightweight proxy for a remote module, usable in get= and map=."""

    __slots__ = ("_model", "path", "_path", "sid")

    def __init__(self, model: _RemoteModel, path: str, sid: int) -> None:
        self._model = model
        self.path = path
        self._path = path
        self.sid = sid

    def __getattr__(self, name: str) -> Any:
        return self._model._child(self.path, name)

    def __getitem__(self, idx: int) -> Any:
        return self._model._child(self.path, str(idx))

    def __dir__(self) -> list[str]:
        return sorted(name for name, _ in self._remote_children())

    def __hash__(self) -> int:
        return hash((id(self._model), self.path))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _RemoteProxy)
            and other._model is self._model
            and other.path == self.path
        )

    def _remote_children(self) -> list[tuple[str, str]]:
        return self._model._children_for(self.path)

    def __repr__(self) -> str:
        return f"RemoteProxy({self.path!r})"


class _RemoteModuleList:
    """Lightweight proxy for a remote ModuleList."""

    __slots__ = ("_items",)

    def __init__(self, items: list[_RemoteProxy]) -> None:
        self._items = items

    def __getitem__(self, idx: int) -> _RemoteProxy:
        return self._items[idx]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


class _RemoteModel:
    """Client model that connects to a remote tinyinterp server over Unix socket."""

    def __init__(self, sock_path: str) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(sock_path)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 << 20)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
        self._lock = threading.Lock()
        self._closed = False
        self.capabilities = cast(dict[str, Any], self._roundtrip(Cmd.HELLO))
        remote_proto = int(self.capabilities.get("protocol", -1))
        if remote_proto != PROTO_VERSION:
            raise RuntimeError(
                f"tinyinterp protocol mismatch: client={PROTO_VERSION} server={remote_proto}"
            )
        tree = json.loads(cast(bytes, self._roundtrip(Cmd.TREE, raw=True)))
        self._types, self._children, self._path_to_sid = index_tree(tree)
        self._proxies: dict[str, _RemoteProxy] = {
            path: _RemoteProxy(self, path, self._path_to_sid[path]) for path in self._types if path
        }
        self._layers: _RemoteModuleList | None = None
        self._plan_cache: dict[str, str] = {}

    @property
    def layers(self) -> _RemoteModuleList:
        if self._layers is None:
            numbered = {
                parent: sorted(
                    (
                        (int(name), self._proxies[path])
                        for name, path, _ in children
                        if name.isdigit()
                    ),
                    key=lambda item: item[0],
                )
                for parent, children in self._children.items()
                if self._types.get(parent) == "ModuleList"
            }
            numbered = {parent: items for parent, items in numbered.items() if items}
            if not numbered:
                raise AttributeError("No layer list found.")
            best_prefix = max(numbered, key=lambda key: len(numbered[key]))
            items = [proxy for _, proxy in numbered[best_prefix]]
            self._layers = _RemoteModuleList(items)
        return self._layers

    def __getattr__(self, name: str) -> Any:
        return self._child("", name)

    def __dir__(self) -> list[str]:
        return sorted(
            set(list(object.__dir__(self)) + [name for name, _ in self._children_for("")])
        )

    def __call__(
        self,
        prompts: Sequence[Any] | None = None,
        *,
        get: Sequence[Any] | Any | None = None,
        map: dict[Any, Any] | None = None,
        grad: bool = False,
        stop_at_last_get: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Forward pass (prefill). Accepts prompts or raw tensors."""
        get_sids = self._normalize_get_sids(get)
        map_specs = self._normalize_map_specs(map)
        if stop_at_last_get:
            if not get_sids:
                raise ValueError("stop_at_last_get=True requires at least one get= site.")
            if map_specs:
                raise ValueError("stop_at_last_get=True does not support map=.")
            if grad:
                raise ValueError("stop_at_last_get=True does not support grad=True.")
        plan_id = self._compile_plan(
            get_sids,
            map_specs,
            output={"logits": False, "activations": True} if stop_at_last_get else None,
        )
        items = None if isinstance(prompts, torch.Tensor) else request_items(prompts)
        if grad and items is not None:
            raise ValueError("Remote grad=True only supports raw tensor model inputs.")
        if items is not None:
            if not items:
                raise ValueError("Expected at least one request.")
            req = {
                "_prompts": items,
                "_plan_id": plan_id,
                "_stop_at_last_get": stop_at_last_get,
                **kwargs,
            }
            outputs = cast(list[Mapping[str, Any]], self._request(Cmd.CALL_MANY, req))
            outputs = [self._remote_output(item) for item in outputs]
            return outputs[0] if len(outputs) == 1 else outputs
        if "input_ids" in kwargs:
            req = {
                "_plan_id": plan_id,
                "_stop_at_last_get": stop_at_last_get,
                **kwargs,
            }
            return self._remote_output(
                cast(Mapping[str, Any], self._request(Cmd.CALL_GRAD if grad else Cmd.CALL, req))
            )
        if prompts is not None:
            kwargs["input_ids"] = prompts
        req = {"_plan_id": plan_id, **kwargs}
        cmd = Cmd.CALL_GRAD if grad else Cmd.CALL
        return self._remote_output(cast(Mapping[str, Any], self._request(cmd, req)))

    def generate(
        self,
        prompts: Sequence[Any] | Any = (),
        *,
        get: Sequence[Any] | Any | None = None,
        map: dict[Any, Any] | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        capture: str = "all",
        **kwargs: Any,
    ) -> Any:
        """Generation (decode). Accepts prompts or raw tensors."""
        plan_id = self._compile_plan(
            self._normalize_get_sids(get),
            self._normalize_map_specs(map),
            output=None,
        )
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            capture=capture,
            **kwargs,
        )
        if "input_ids" in gen_kwargs:
            req = {"_plan_id": plan_id, **gen_kwargs}
            value = self._request(Cmd.GENERATE, req)
            return (
                self._remote_output(cast(Mapping[str, Any], value))
                if isinstance(value, Mapping) and "generated_ids" in value
                else value
            )
        items = None if isinstance(prompts, torch.Tensor) else request_items(prompts)
        if items is not None:
            if not items:
                raise ValueError("Expected at least one request.")
            req = {"_prompts": items, "_plan_id": plan_id, **gen_kwargs}
            value = self._request(Cmd.GENERATE_MANY, req)
            return (
                self._remote_output(cast(Mapping[str, Any], value))
                if isinstance(value, Mapping) and "generated_ids" in value
                else value
            )
        if isinstance(prompts, torch.Tensor):
            req = {"input_ids": prompts, "_plan_id": plan_id, **gen_kwargs}
            value = self._request(Cmd.GENERATE, req)
            return (
                self._remote_output(cast(Mapping[str, Any], value))
                if isinstance(value, Mapping) and "generated_ids" in value
                else value
            )
        raise TypeError("model.generate(...) expects token tensors or prompt requests.")

    def collect(
        self,
        requests: Sequence[Any] | Any,
        *,
        get: Sequence[Any] | Any | None = None,
        map: dict[Any, Any] | None = None,
        stop_at_last_get: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Collect activations over one or more requests on the remote server."""
        items = request_items(requests)
        if items is None:
            raise TypeError("model.collect(...) expects one request or a sequence of requests.")
        if not items:
            raise ValueError("Expected at least one request.")
        get_sids = self._normalize_get_sids(get)
        if not get_sids:
            raise ValueError("model.collect(...) requires at least one get= site.")
        if stop_at_last_get and map is not None:
            raise ValueError("stop_at_last_get=True does not support map=.")
        plan_id = self._compile_plan(
            get_sids,
            self._normalize_map_specs(map),
            output={"logits": False, "activations": True},
        )
        req = {
            "_prompts": items,
            "_plan_id": plan_id,
            "_stop_at_last_get": stop_at_last_get,
            **kwargs,
        }
        return [
            self._remote_output(item)
            for item in cast(list[Mapping[str, Any]], self._request(Cmd.CALL_MANY, req))
        ]

    def close(self) -> None:
        self._closed = True
        self._sock.close()

    def _compile_plan(
        self,
        get_sids: list[int] | None,
        map_specs: list[dict[str, Any]] | None,
        *,
        output: dict[str, Any] | None,
    ) -> str:
        request = {"get_sids": get_sids, "map_specs": map_specs, "output": output}
        key = hashlib.sha1(serialize_request(request)).hexdigest()
        plan_id = self._plan_cache.get(key)
        if plan_id is not None:
            return plan_id
        plan_id = cast(str, self._request(Cmd.COMPILE, request))
        self._plan_cache[key] = plan_id
        return plan_id

    def _normalize_get_sids(self, get: Any) -> list[int] | None:
        if get is None:
            return None
        sites = [get] if isinstance(get, (str, _RemoteProxy)) else list(get)
        return [self._site_id(site) for site in sites]

    def _normalize_map_specs(self, map: Any) -> list[dict[str, Any]] | None:
        if map is None:
            return None
        return [_encode_remote_map_spec(self._site_id(site), fn) for site, fn in map.items()]

    def _site_id(self, site: Any) -> int:
        if isinstance(site, _RemoteProxy):
            return site.sid
        if isinstance(site, str):
            try:
                return self._path_to_sid[site]
            except KeyError as exc:
                raise KeyError(f"Unknown remote site {site!r}.") from exc
        raise TypeError("Remote get= and map= must use remote proxies or path strings.")

    def _child(self, parent: str, name: str) -> _RemoteProxy:
        for child_name, path, _ in self._children.get(parent, []):
            if child_name == name:
                return self._proxies[path]
        if parent:
            raise AttributeError(f"No child {name!r} under {parent!r}.")
        raise AttributeError(f"No module named {name!r}")

    def _children_for(self, path: str) -> list[tuple[str, str]]:
        return [(name, type_name) for name, _, type_name in self._children.get(path, [])]

    def _fetch_value(self, value_id: str) -> Any:
        if self._closed:
            raise RuntimeError("Remote model is closed.")
        value = self._request(Cmd.FETCH_VALUE, {"value_id": value_id})
        if isinstance(value, Mapping) and "__remote_error__" in value:
            raise KeyError(str(value["__remote_error__"]))
        return value

    def _release_value(self, value_id: str) -> None:
        if self._closed:
            return
        try:
            self._request(Cmd.RELEASE_VALUE, {"value_id": value_id})
        except OSError as exc:
            logger.debug("remote value release failed for %s: %s", value_id, exc)

    def _fetch_grad_value(self, grad_id: str, target: str | int) -> Any:
        if self._closed:
            raise RuntimeError("Remote model is closed.")
        return self._request(Cmd.FETCH_GRAD_VALUE, {"grad_id": grad_id, "target": target})

    def _fetch_target_grad(self, grad_id: str, target: str | int) -> Any:
        if self._closed:
            raise RuntimeError("Remote model is closed.")
        return self._request(Cmd.FETCH_TARGET_GRAD, {"grad_id": grad_id, "target": target})

    def _fetch_input_grads(self, grad_id: str) -> Any:
        if self._closed:
            raise RuntimeError("Remote model is closed.")
        return self._request(Cmd.FETCH_INPUT_GRADS, {"grad_id": grad_id})

    def _backward_grad(
        self, grad_id: str, target: str | int, gradient: torch.Tensor | None = None
    ) -> None:
        if self._closed:
            raise RuntimeError("Remote model is closed.")
        self._request(Cmd.BACKWARD, {"grad_id": grad_id, "target": target, "gradient": gradient})

    def _release_grad(self, grad_id: str) -> None:
        if self._closed:
            return
        try:
            self._request(Cmd.RELEASE_GRAD, {"grad_id": grad_id})
        except OSError as exc:
            logger.debug("remote grad release failed for %s: %s", grad_id, exc)

    def _roundtrip(self, cmd: Cmd, payload: bytes = b"", *, raw: bool = False) -> Any:
        with self._lock:
            send(self._sock, cmd, payload)
            _, data = recv(self._sock)
        return data if raw else deserialize_response(data)

    def _request(self, cmd: Cmd, request: Mapping[str, Any]) -> Any:
        return self._roundtrip(cmd, serialize_request(request))

    def _remote_output(self, data: Mapping[str, Any]) -> Any:
        return output_from_remote_dict(
            data,
            fetch=self._fetch_value,
            release=self._release_value,
            path_to_sid=self._path_to_sid,
            fetch_grad_value=self._fetch_grad_value,
            fetch_target_grad=self._fetch_target_grad,
            fetch_input_grads=self._fetch_input_grads,
            backward_grad=self._backward_grad,
            release_grad=self._release_grad,
        )


# --- helpers ---


def _encode_remote_map_spec(sid: int, fn: Any) -> dict[str, Any]:
    if isinstance(fn, maps_mod._Zero):
        return {"sid": sid, "op": "zero"}
    if isinstance(fn, maps_mod._Add):
        return {"sid": sid, "op": "add", "value": fn.delta}
    if isinstance(fn, maps_mod._Scale):
        return {"sid": sid, "op": "scale", "value": fn.factor}
    if isinstance(fn, maps_mod._Replace):
        return {"sid": sid, "op": "replace", "value": fn.value}
    raise TypeError(
        "tinyinterp remote execution only supports built-in map ops: zero, add, scale, replace."
    )
