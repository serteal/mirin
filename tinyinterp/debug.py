"""Debug logging helpers for tinyinterp."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def log_model_ready(model_name: str, n_modules: int) -> None:
    """Print a short summary when a model is wrapped."""

    print(f"[ti] Model: {model_name}")
    print(f"[ti] Hooked modules: {n_modules}")


def log_call_start(
    get_proxies: Sequence[Any],
    map_proxies: Mapping[Any, Callable[[torch.Tensor], torch.Tensor]],
    *,
    grad: bool,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> None:
    """Print a summary of a single model call."""

    get_names = ", ".join(proxy.path for proxy in get_proxies)
    map_names = ", ".join(f"{proxy.path}:{_callable_name(fn)}" for proxy, fn in map_proxies.items())
    input_shape = _input_shape(args, kwargs)
    print(f"[ti] call: get=[{get_names}] map=[{map_names}] grad={grad} input_shape={input_shape}")


def log_timing(
    *,
    activate_ns: int,
    forward_ns: int,
    collect_ns: int,
    n_activations: int,
    activation_bytes: int,
) -> None:
    """Print a per-call timing breakdown."""

    total_ns = activate_ns + forward_ns + collect_ns
    print(f"[ti]   activate_hooks: {activate_ns / 1e6:.3f}ms")
    print(f"[ti]   forward_pass:   {forward_ns / 1e6:.3f}ms")
    print(
        f"[ti]   collect:        {collect_ns / 1e6:.3f}ms "
        f"({n_activations} activations, {activation_bytes} bytes)"
    )
    print(f"[ti]   TOTAL:          {total_ns / 1e6:.3f}ms")


def log_hook_event(
    path: str,
    *,
    sid: int,
    get: bool,
    map_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    activation: torch.Tensor | None = None,
) -> None:
    """Print a per-hook trace line at debug level 4."""

    map_name = "None" if map_fn is None else _callable_name(map_fn)
    if activation is None:
        print(f"[ti] hook[{path}] (id={sid}): SKIP (flags: get={get}, map={map_name})")
        return

    action_parts: list[str] = []
    if get:
        action_parts.append(
            "GET"
            f" -> buffer[{sid}] shape={list(activation.shape)}"
            f" dtype={_dtype_name(activation)} ({_format_bytes(_tensor_bytes(activation))})"
        )
    if map_fn is not None:
        action_parts.append(f"MAP -> {map_name} applied, output shape={list(activation.shape)}")
    print(f"[ti] hook[{path}] (id={sid}): {'; '.join(action_parts)}")


def render_intervention_graph(
    get_proxies: Sequence[Any],
    map_proxies: Mapping[Any, Callable[[torch.Tensor], torch.Tensor]],
    *,
    output_path: str,
) -> None:
    """Render a small SVG showing the modules captured or mapped in one call."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = sorted(
        {proxy.path for proxy in get_proxies} | {proxy.path for proxy in map_proxies}
    )
    height = max(120, 80 + 40 * len(row_labels))
    svg_lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" '
            f'viewBox="0 0 900 {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f8f8f5" />',
        (
            '<text x="24" y="28" font-family="monospace" '
            'font-size="14" fill="#222">tinyinterp intervention graph</text>'
        ),
        (
            '<text x="24" y="48" font-family="monospace" '
            'font-size="11" fill="#555">blue = get, red = map</text>'
        ),
    ]

    for row_idx, label in enumerate(row_labels):
        y = 68 + row_idx * 40
        mapped = any(proxy.path == label for proxy in map_proxies)
        fill = "#f4d7d7" if mapped else "#ffffff"
        svg_lines.append(
            f'<rect x="160" y="{y}" width="560" height="26" rx="6" '
            f'fill="{fill}" stroke="#333" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="176" y="{y + 17}" font-family="monospace" '
            f'font-size="12" fill="#222">{label}</text>'
        )
        if any(proxy.path == label for proxy in get_proxies):
            svg_lines.append(
                f'<circle cx="132" cy="{y + 13}" r="8" fill="#3b82f6" '
                'stroke="#1d4ed8" stroke-width="1" />'
            )
        if mapped:
            svg_lines.append(
                f'<circle cx="748" cy="{y + 13}" r="8" fill="#ef4444" '
                'stroke="#b91c1c" stroke-width="1" />'
            )

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="ascii")
    print(f"[ti] graph: {path}")


def _input_shape(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, torch.Tensor):
            return str(tuple(value.shape))
    return "unknown"


def _callable_name(fn: Callable[[torch.Tensor], torch.Tensor]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.numel()


def _format_bytes(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} B"
