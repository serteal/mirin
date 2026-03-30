"""Support and availability metadata for benchmark libraries."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LibrarySpec:
    """One benchmarkable library or endpoint adapter."""

    name: str
    package: str | None = None
    module: str | None = None
    families: frozenset[str] | None = None
    note: str = ""


LOCAL_LIBRARY_SPECS = (
    LibrarySpec(name="raw_hf", package="transformers", module="transformers"),
    LibrarySpec(name="tinyinterp_local"),
    LibrarySpec(
        name="transformerlens",
        package="transformer-lens",
        module="transformer_lens",
        families=frozenset({"llama3.1", "qwen3", "gemma2"}),
        note="Conservative allowlist based on official supported-model docs.",
    ),
    LibrarySpec(name="nnterp", package="nnterp", module="nnterp"),
)

REMOTE_LIBRARY_SPECS = (
    LibrarySpec(name="hf_generate", package="transformers", module="transformers"),
    LibrarySpec(name="tinyinterp_local"),
    LibrarySpec(name="tinyinterp_remote"),
)


def local_support_map(model_family: str) -> dict[str, dict[str, Any]]:
    """Return local-library support metadata for one model family."""

    return {spec.name: support_entry(spec, model_family) for spec in LOCAL_LIBRARY_SPECS}


def remote_support_map(model_family: str) -> dict[str, dict[str, Any]]:
    """Return remote-library support metadata for one model family."""

    return {spec.name: support_entry(spec, model_family) for spec in REMOTE_LIBRARY_SPECS}


def support_entry(spec: LibrarySpec, model_family: str) -> dict[str, Any]:
    """Describe whether one benchmark adapter is runnable in this environment."""

    installed = _is_installed(spec)
    family_supported = spec.families is None or model_family in spec.families
    runnable = installed and family_supported

    reason = None
    if not installed:
        package = spec.package or spec.module or spec.name
        reason = f"{package} is not installed."
    elif not family_supported:
        reason = f"{spec.name} is not enabled for model family {model_family!r} in this matrix."

    return {
        "name": spec.name,
        "package": spec.package or "",
        "module": spec.module or "",
        "version": _package_version(spec.package),
        "installed": installed,
        "supported": family_supported,
        "runnable": runnable,
        "reason": reason or "",
        "note": spec.note,
    }


def runnable_support(
    support: dict[str, dict[str, Any]],
    name: str,
) -> tuple[bool, str]:
    """Return a runnable decision and human-readable reason."""

    entry = support[name]
    reason = entry.get("reason", "")
    return bool(entry["runnable"]), str(reason)


def _is_installed(spec: LibrarySpec) -> bool:
    if spec.package is None and spec.module is None:
        return True
    if spec.module is not None and importlib.util.find_spec(spec.module) is not None:
        return True
    if spec.package is None:
        return False
    try:
        importlib.metadata.version(spec.package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _package_version(package: str | None) -> str:
    if package is None:
        return "built-in"
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
