"""Compiled interpretability plans for the local runtime."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from .. import maps as maps_mod
from ..model import Model, _ModuleProxy


@dataclass(frozen=True, slots=True)
class MapSpec:
    """Serializable description of a built-in runtime map op."""

    path: str
    op: str
    value: Any | None = None


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """Narrow result selection for compiled runtime execution."""

    tokens: bool
    logits: bool
    activations: bool
    activations_to_cpu: bool
    logits_to_cpu: bool

    def fingerprint(self) -> str:
        payload = {
            "tokens": self.tokens,
            "logits": self.logits,
            "activations": self.activations,
            "activations_to_cpu": self.activations_to_cpu,
            "logits_to_cpu": self.logits_to_cpu,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """A fixed interpretability plan compiled against one local model."""

    id: str
    get_paths: tuple[str, ...]
    map_specs: tuple[MapSpec, ...]
    output: OutputPolicy
    fingerprint: str
    get_proxies: tuple[_ModuleProxy, ...]
    map_dict: dict[_ModuleProxy, Any]


SiteLike = str | _ModuleProxy
OutputPolicyLike = str | Mapping[str, bool] | None


def compile_plan(
    model: Model,
    *,
    get: Sequence[SiteLike] | SiteLike | None = None,
    mapping: Mapping[SiteLike, Any] | None = None,
    output: OutputPolicyLike = None,
) -> CompiledPlan:
    """Compile user-facing paths or proxies into a fixed runtime plan."""

    get_proxies = _normalize_get(model, get)
    map_dict, map_specs = _normalize_map(model, mapping)
    output_policy = _normalize_output_policy(output, has_get=bool(get_proxies))
    if get_proxies and not output_policy.activations:
        raise ValueError("Plans with get= must return activations.")

    fingerprint_payload = {
        "get": [proxy.path for proxy in get_proxies],
        "map": [
            {"path": spec.path, "op": spec.op, "value": _fingerprint_value(spec.value)}
            for spec in map_specs
        ],
        "output": output_policy.fingerprint(),
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return CompiledPlan(
        id=uuid.uuid4().hex,
        get_paths=tuple(proxy.path for proxy in get_proxies),
        map_specs=tuple(map_specs),
        output=output_policy,
        fingerprint=fingerprint,
        get_proxies=tuple(get_proxies),
        map_dict=map_dict,
    )


def resolve_site(model: Model, site: SiteLike) -> _ModuleProxy:
    """Resolve a dotted path or validate a local mirin proxy."""

    if isinstance(site, _ModuleProxy):
        return model._validate_proxy(site)
    current: Any = model
    if site in ("", "<root>"):
        return cast(_ModuleProxy, current._root)
    for part in site.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return model._validate_proxy(current)


def _normalize_get(
    model: Model,
    get: Sequence[SiteLike] | SiteLike | None,
) -> list[_ModuleProxy]:
    if get is None:
        return []
    sites = [get] if isinstance(get, (str, _ModuleProxy)) else list(get)
    return [resolve_site(model, site) for site in sites]


def _normalize_map(
    model: Model,
    mapping: Mapping[SiteLike, Any] | None,
) -> tuple[dict[_ModuleProxy, Any], list[MapSpec]]:
    if mapping is None:
        return {}, []
    normalized: dict[_ModuleProxy, Any] = {}
    specs: list[MapSpec] = []
    for site, fn in mapping.items():
        proxy = resolve_site(model, site)
        spec = _encode_map_fn(proxy.path, fn)
        normalized[proxy] = fn
        specs.append(spec)
    return normalized, specs


def _normalize_output_policy(output: OutputPolicyLike, *, has_get: bool) -> OutputPolicy:
    if output is None:
        return OutputPolicy(
            tokens=False,
            logits=True,
            activations=has_get,
            activations_to_cpu=False,
            logits_to_cpu=False,
        )
    if isinstance(output, str):
        named = output.lower()
        if named == "tokens_only":
            return OutputPolicy(
                tokens=True,
                logits=False,
                activations=False,
                activations_to_cpu=False,
                logits_to_cpu=False,
            )
        if named == "logits_slice":
            return OutputPolicy(
                tokens=True,
                logits=True,
                activations=False,
                activations_to_cpu=False,
                logits_to_cpu=False,
            )
        if named == "activations":
            return OutputPolicy(
                tokens=False,
                logits=False,
                activations=True,
                activations_to_cpu=True,
                logits_to_cpu=False,
            )
        raise ValueError(f"Unknown output policy {output!r}.")
    return OutputPolicy(
        tokens=bool(output.get("tokens", False)),
        logits=bool(output.get("logits", True)),
        activations=bool(output.get("activations", has_get)),
        activations_to_cpu=bool(output.get("activations_to_cpu", False)),
        logits_to_cpu=bool(output.get("logits_to_cpu", False)),
    )


def _encode_map_fn(path: str, fn: Any) -> MapSpec:
    if isinstance(fn, maps_mod._Zero):
        return MapSpec(path=path, op="zero")
    if isinstance(fn, maps_mod._Add):
        return MapSpec(path=path, op="add", value=fn.delta)
    if isinstance(fn, maps_mod._Scale):
        return MapSpec(path=path, op="scale", value=fn.factor)
    if isinstance(fn, maps_mod._Replace):
        return MapSpec(path=path, op="replace", value=fn.value)
    raise TypeError("mirin runtime only supports built-in map ops: zero, add, scale, replace.")


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(value.detach().cpu(), buffer)
        return {
            "tensor": True,
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "sha1": hashlib.sha1(buffer.getvalue()).hexdigest()[:16],
        }
    return value
