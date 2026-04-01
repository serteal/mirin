"""Unit tests for debug helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mirin.debug import (
    _format_bytes,
    log_call_start,
    log_hook_event,
    log_model_ready,
    log_timing,
    render_intervention_graph,
)


class _Proxy:
    def __init__(self, path: str) -> None:
        self.path = path


def test_render_intervention_graph_requires_svg_file_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end with \\.svg"):
        render_intervention_graph([], {}, output_path=str(tmp_path / "graph.txt"))

    directory = tmp_path / "graph.svg"
    directory.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        render_intervention_graph([], {}, output_path=str(directory))


def test_render_intervention_graph_writes_svg(tmp_path: Path) -> None:
    path = tmp_path / "graph.svg"
    render_intervention_graph([], {}, output_path=str(path))
    assert path.exists()
    assert path.read_text(encoding="ascii").startswith("<svg")


def test_format_bytes_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _format_bytes(-1)


def test_debug_log_helpers_emit_readable_output(capsys: pytest.CaptureFixture[str]) -> None:
    proxy = _Proxy("model.layers.0")

    log_model_ready("ToyModel", 12)
    log_call_start(
        [proxy],
        {proxy: lambda x: x},
        grad=True,
        stop_at_last_get=False,
        args=(torch.zeros(1, 2),),
        kwargs={},
    )
    log_timing(
        activate_ns=1_000_000,
        forward_ns=2_000_000,
        collect_ns=1_000_000,
        n_activations=1,
        activation_bytes=16,
        stopped_early=False,
    )
    log_hook_event(
        "model.layers.0",
        sid=3,
        get=True,
        map_fn=None,
        activation=torch.zeros(1, 2),
    )

    output = capsys.readouterr().out
    assert "Model: ToyModel" in output
    assert "Hooked modules: 12" in output
    assert "call:" in output
    assert "TOTAL:" in output
    assert "hook[model.layers.0]" in output
