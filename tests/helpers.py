"""Small fake models used by mirin tests."""

from __future__ import annotations

import inspect
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


class FakeTokenizer:
    """Tiny tokenizer with text and chat-template support for model tests."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("FakeTokenizer only supports return_tensors='pt'.")
        texts = [text] if isinstance(text, str) else list(text)
        rows: list[list[int]] = []
        for item in texts:
            encoded = [1]
            encoded.extend(((ord(char) % 11) + 3) for char in item)
            rows.append(encoded[:16])
        max_len = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for idx, row in enumerate(rows):
            input_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[idx, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | torch.Tensor:
        rendered = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
        )
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:"
        if tokenize:
            return self(rendered)["input_ids"][0]
        return rendered


def request_contract_cases(
    *,
    add_generation_prompt: bool,
) -> list[tuple[str, Any, dict[str, torch.Tensor]]]:
    """Canonical request-form cases plus their expected tokenized inputs."""

    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hello"}]
    text_tokens = tokenizer("hello", return_tensors="pt")
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ),
    )
    message_tokens = tokenizer(rendered, return_tensors="pt")
    return [
        ("string", "hello", text_tokens),
        ("text-mapping", {"text": "hello"}, text_tokens),
        ("token-mapping", text_tokens, text_tokens),
        ("messages-mapping", {"messages": messages}, message_tokens),
        ("message-dict", messages[0], message_tokens),
        ("message-list", messages, message_tokens),
    ]


def filter_forward_inputs(
    model: nn.Module,
    kwargs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Filter kwargs down to what ``model.forward`` accepts."""

    signature = inspect.signature(model.forward)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def generate_lengths(output: Any) -> tuple[list[int], list[int]]:
    prompt_value = output.prompt_length
    generated_value = output.generated_length
    prompt_lengths = (
        [int(prompt_value)]
        if isinstance(prompt_value, int)
        else [int(length) for length in prompt_value]
    )
    generated_lengths = (
        [int(generated_value)]
        if isinstance(generated_value, int)
        else [int(length) for length in generated_value]
    )
    return prompt_lengths, generated_lengths


def generate_row(output: Any, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_lengths, generated_lengths = generate_lengths(output)
    sequences = cast(torch.Tensor, output.sequences)
    generated_ids = cast(torch.Tensor, output.generated_ids)
    total = prompt_lengths[idx] + generated_lengths[idx]
    return (
        sequences[idx : idx + 1, :total],
        generated_ids[idx : idx + 1, : generated_lengths[idx]],
    )


def generate_activation_row(
    output: Any,
    site: Any,
    idx: int,
    *,
    capture: str,
) -> torch.Tensor:
    prompt_lengths, generated_lengths = generate_lengths(output)
    width = (
        generated_lengths[idx]
        if capture == "generated"
        else prompt_lengths[idx] + generated_lengths[idx]
    )
    activations = cast(torch.Tensor, output[site])
    return activations[idx : idx + 1, :width]


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


class FakeDecoderAttention(nn.Module):
    """Tiny decoder-style attention with combined QKV projection."""

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


class FakeDecoderMlp(nn.Module):
    """Tiny decoder-style MLP with c_fc/c_proj."""

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


class DecoderBlock(nn.Module):
    """Tiny decoder-style block with direct attention and MLP children."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.attn = FakeDecoderAttention(width, n_heads)
        self.mlp = FakeDecoderMlp(width)
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


class FakeDecoderBackbone(nn.Module):
    """Tiny decoder-style backbone with a typed block stack."""

    h: nn.ModuleList

    def __init__(self, width: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.h = nn.ModuleList(DecoderBlock(width, n_heads) for _ in range(n_layers))


class FakeDecoderModel(nn.Module):
    """Tiny decoder-style model with ``transformer.h`` blocks."""

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
        self.transformer = FakeDecoderBackbone(width, n_layers, n_heads)
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

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        del attention_mask, do_sample, use_cache
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(tokens).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


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

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        del attention_mask, do_sample, use_cache
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(tokens).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


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
    """Resolve a dotted proxy path against a mirin model."""

    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
