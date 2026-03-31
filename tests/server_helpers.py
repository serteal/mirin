"""Shared helpers for server and remote integration tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

import tinyinterp as ti

from .helpers import FakeTokenizer


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


class GradProbeModel(nn.Module):
    """Small differentiable model for remote grad handle tests."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        hidden = self.hidden(x)
        return {"logits": self.readout(torch.tanh(hidden))}


class SamplingGenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.readout = nn.Linear(4, 16, bias=False)
        self.last_generate_kwargs: dict[str, Any] | None = None

    def forward(self, input_ids: torch.Tensor) -> Any:
        hidden = self.embed(input_ids)
        return {"logits": self.readout(hidden)}

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        del attention_mask, use_cache
        self.last_generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
        }
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            next_token = torch.full_like(tokens[:, :1], 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


class _BarrierBlock(nn.Module):
    def __init__(self, gate: _ThreadGate) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gate.wait()
        return x


class _ThreadGate:
    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._arrived = 0
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.release = threading.Event()
        if parties <= 1:
            self.ready.set()
            self.release.set()

    def wait(self) -> None:
        with self._lock:
            self._arrived += 1
            if self._arrived >= self._parties:
                self.ready.set()
        if not self.ready.wait(timeout=5.0):
            raise RuntimeError("Timed out waiting for concurrent entry.")
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Timed out waiting for release.")


class BarrierProbeModel(nn.Module):
    def __init__(self, gate: _ThreadGate) -> None:
        super().__init__()
        self.entry = _BarrierBlock(gate)
        self.hidden = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        x = self.entry(x)
        hidden = self.hidden(x)
        return {"logits": self.readout(torch.tanh(hidden))}


def _seeded_model(factory: Callable[[], nn.Module], seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return factory()


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    interval_s: float = 0.02,
    message: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError(message)

class _HybridTextConfig:
    num_hidden_layers = 2
    layer_types = ["full_attention", "linear_attention"]


class Qwen3_5DynamicCache:
    def __init__(self, config: Any) -> None:
        self.layer_types = list(config.layer_types)
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]


def _start_server(
    factory: Callable[[], nn.Module], sock_path: str
) -> tuple[Any, Any]:
    import os
    import socket
    import threading
    import time

    server = ti.Server(factory(), tokenizer=FakeTokenizer())
    t = threading.Thread(target=lambda: server.serve(sock_path), daemon=True)
    t.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(sock_path)
            except OSError:
                time.sleep(0.02)
            else:
                probe.close()
                return server, t
            finally:
                try:
                    probe.close()
                except OSError:
                    pass
        time.sleep(0.02)
    raise RuntimeError(f"Server did not open socket {sock_path}.")


def _stop_server(server: Any, sock_path: str) -> None:
    import os

    server.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


def _ids(batch: int, seq: int = 4) -> torch.Tensor:
    return torch.arange(batch * seq, dtype=torch.long).view(batch, seq) % 16
