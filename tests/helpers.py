"""Small fake models used by tinyinterp tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn


@dataclass
class FakeOutput:
    """Minimal object with a logits attribute like HF model outputs."""

    logits: torch.Tensor

    def __getitem__(self, index: int | slice) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return (self.logits,)[index]


def _reshape_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    d_head = tensor.shape[-1] // n_heads
    return tensor.view(*tensor.shape[:-1], n_heads, d_head)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_head = q.shape[-1]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(d_head)
    weights = torch.softmax(scores, dim=-1)
    context = torch.einsum("bhqk,bkhd->bqhd", weights, v)
    return context, weights


class FakeGpt2Attention(nn.Module):
    """Tiny GPT-2 style attention with combined QKV projection."""

    supports_attention_pattern = True

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q_heads = _reshape_heads(q, self.n_heads)
        k_heads = _reshape_heads(k, self.n_heads)
        v_heads = _reshape_heads(v, self.n_heads)
        context, weights = _attention(q_heads, k_heads, v_heads)
        attn_out = self.c_proj(context.reshape(*context.shape[:-2], -1))
        return attn_out, (weights if output_attentions else None)


class FakeLlamaAttention(nn.Module):
    """Tiny LLaMA style attention with separate Q/K/V projections."""

    supports_attention_pattern = True

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_heads = _reshape_heads(self.q_proj(hidden_states), self.n_heads)
        k_heads = _reshape_heads(self.k_proj(hidden_states), self.n_heads)
        v_heads = _reshape_heads(self.v_proj(hidden_states), self.n_heads)
        context, weights = _attention(q_heads, k_heads, v_heads)
        attn_out = self.o_proj(context.reshape(*context.shape[:-2], -1))
        return attn_out, (weights if output_attentions else None)


class FakeGpt2Mlp(nn.Module):
    """Tiny GPT-2 style MLP with c_fc/c_proj."""

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = width * 4
        self.c_fc = nn.Linear(width, hidden)
        self.c_proj = nn.Linear(hidden, width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.c_fc(hidden_states))
        return cast(torch.Tensor, self.c_proj(hidden))


class FakeLlamaMlp(nn.Module):
    """Tiny LLaMA style gated MLP with up/gate/down projections."""

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = width * 4
        self.gate_proj = nn.Linear(width, hidden)
        self.up_proj = nn.Linear(width, hidden)
        self.down_proj = nn.Linear(hidden, width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = torch.sigmoid(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return cast(torch.Tensor, self.down_proj(gated * up))


class Gpt2Block(nn.Module):
    """Tiny GPT-2 shaped block with direct attention and MLP children."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.attn = FakeGpt2Attention(width, n_heads)
        self.mlp = FakeGpt2Mlp(width)
        self.fail = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        if self.fail:
            raise RuntimeError("boom")
        hidden_states = (
            hidden_states + self.attn(hidden_states, output_attentions=output_attentions)[0]
        )
        hidden_states = hidden_states + self.mlp(hidden_states)
        return (hidden_states,)


class LlamaBlock(nn.Module):
    """Tiny LLaMA shaped block with direct attention and MLP children."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = FakeLlamaAttention(width, n_heads)
        self.mlp = FakeLlamaMlp(width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states + self.self_attn(hidden_states, output_attentions=output_attentions)[0]
        )
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


class FakeGpt2Backbone(nn.Module):
    """Tiny GPT-2 style backbone with a typed block stack."""

    h: nn.ModuleList

    def __init__(self, width: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.h = nn.ModuleList(Gpt2Block(width, n_heads) for _ in range(n_layers))


class FakeGpt2Model(nn.Module):
    """Tiny GPT-2 shaped model with ``transformer.h`` blocks."""

    def __init__(
        self,
        *,
        vocab_size: int = 16,
        width: int = 8,
        n_layers: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.transformer = FakeGpt2Backbone(width, n_layers, n_heads)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.config = SimpleNamespace(
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=width,
            n_inner=width * 4,
            _attn_implementation="eager",
        )

    def forward(self, input_ids: torch.Tensor, *, output_attentions: bool = False) -> FakeOutput:
        hidden_states = self.embed(input_ids)
        for block in self.transformer.h:
            hidden_states = block(hidden_states, output_attentions=output_attentions)[0]
        return FakeOutput(logits=self.lm_head(hidden_states))


class FakeLlamaBackbone(nn.Module):
    """Tiny LLaMA shaped backbone with ``model.layers`` blocks."""

    layers: nn.ModuleList

    def __init__(self, width: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(LlamaBlock(width, n_heads) for _ in range(n_layers))


class FakeLlamaModel(nn.Module):
    """Tiny LLaMA shaped model with ``model.layers`` blocks."""

    def __init__(
        self,
        *,
        vocab_size: int = 16,
        width: int = 8,
        n_layers: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.model = FakeLlamaBackbone(width, n_layers, n_heads)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            hidden_size=width,
            intermediate_size=width * 4,
            _attn_implementation="eager",
        )

    def forward(self, input_ids: torch.Tensor, *, output_attentions: bool = False) -> FakeOutput:
        hidden_states = self.embed(input_ids)
        for block in self.model.layers:
            hidden_states = block(hidden_states, output_attentions=output_attentions)
        return FakeOutput(logits=self.lm_head(hidden_states))


def get_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path against a model."""

    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = cast(Any, current)[int(part)]
        else:
            current = getattr(current, part)
    return cast(nn.Module, current)


def get_proxy(model: Any, path: str) -> Any:
    """Resolve a dotted proxy path against a tinyinterp model."""

    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
