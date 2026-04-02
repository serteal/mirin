"""Transformer-backed server tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import mirin as ti
from mirin.server.cache import QwenHybridAdapter

from .helpers import get_proxy
from .server_helpers import Qwen3_5DynamicCache, _HybridTextConfig


def test_qwen_hybrid_adapter_append_and_compact() -> None:
    adapter = QwenHybridAdapter()
    wrapped = nn.Module()
    wrapped.config = _HybridTextConfig()

    first = Qwen3_5DynamicCache(wrapped.config)
    first.key_cache[0] = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    first.value_cache[0] = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4) + 100
    first.conv_states[1] = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
    first.recurrent_states[1] = torch.arange(8, dtype=torch.float32).view(1, 2, 4)

    second = Qwen3_5DynamicCache(wrapped.config)
    second.key_cache[0] = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4) + 200
    second.value_cache[0] = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4) + 300
    second.conv_states[1] = torch.arange(6, dtype=torch.float32).view(1, 2, 3) + 400
    second.recurrent_states[1] = torch.arange(8, dtype=torch.float32).view(1, 2, 4) + 500

    merged = cast(Qwen3_5DynamicCache, adapter.append_cache(first, second, wrapped))
    assert adapter.supports_batched_decode()
    assert cast(torch.Tensor, merged.key_cache[0]).shape[0] == 2
    assert cast(torch.Tensor, merged.conv_states[1]).shape[0] == 2
    compacted = cast(Qwen3_5DynamicCache, adapter.compact_cache(merged, [1], wrapped))
    torch.testing.assert_close(
        cast(torch.Tensor, compacted.key_cache[0]),
        cast(torch.Tensor, second.key_cache[0]),
    )
    torch.testing.assert_close(
        cast(torch.Tensor, compacted.conv_states[1]),
        cast(torch.Tensor, second.conv_states[1]),
    )


def _build_llama31() -> tuple[torch.nn.Module, dict[str, Any]]:
    transformers = pytest.importorskip(
        "transformers",
        reason="Install optional transformers dependency with `uv sync --extra transformers`.",
    )
    torch.manual_seed(0)
    AutoConfig = transformers.AutoConfig
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    config.vocab_size = 64
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.max_position_embeddings = 32
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = 0
    config.use_cache = True
    config.attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    return model, inputs


def _build_qwen35() -> tuple[torch.nn.Module, dict[str, Any]]:
    transformers = pytest.importorskip(
        "transformers",
        reason="Install optional transformers dependency with `uv sync --extra transformers`.",
    )
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers build does not expose Qwen3_5TextConfig.")
    torch.manual_seed(0)
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    Qwen3_5TextConfig = transformers.Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
        attn_implementation="eager",
        layer_types=["full_attention", "linear_attention"],
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    return model, inputs


def test_server_transformers_prefill_decode_matches_generate() -> None:
    wrapped, inputs = _build_llama31()
    expected = cast(
        Any,
        wrapped,
    ).generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
    )

    server = ti.Server(_build_llama31()[0])
    plan = server.compile(get=["model.layers.1.self_attn"])
    session = server.open_session(plan=plan, cache="dynamic")
    prefill = server.prefill(session, **inputs)
    decoded = server.decode([session], max_new_tokens=3, do_sample=False)[0]
    actual = torch.cat([inputs["input_ids"], cast(torch.Tensor, decoded.token_ids)], dim=-1)

    assert torch.equal(actual, expected)
    assert prefill.logits is not None
    assert "model.layers.1.self_attn" in prefill.activations


def test_server_transformers_chunked_prefill_matches_full_capture() -> None:
    wrapped, inputs = _build_llama31()
    local_model = ti.Model(wrapped)
    proxy = get_proxy(local_model, "model.layers.1.self_attn")
    expected = local_model(**inputs, get=[proxy])

    server = ti.Server(_build_llama31()[0])
    plan = server.compile(get=["model.layers.1.self_attn"])
    session = server.open_session(plan=plan, cache="dynamic")
    actual = server.prefill(session, chunk_size=2, **inputs)

    assert actual.logits is not None
    torch.testing.assert_close(
        actual.activations["model.layers.1.self_attn"],
        expected[proxy],
        atol=1e-6,
        rtol=1e-6,
    )


def test_server_transformers_static_prefill_decode_matches_generate() -> None:
    wrapped, inputs = _build_llama31()
    expected = cast(
        Any,
        wrapped,
    ).generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
    )

    server = ti.Server(_build_llama31()[0])
    session = server.open_session(
        cache="static",
        limits={"max_total_tokens": int(inputs["input_ids"].shape[-1]) + 3},
    )
    _ = server.prefill(session, **inputs)
    decoded = server.decode([session], max_new_tokens=3, do_sample=False)[0]
    actual = torch.cat([inputs["input_ids"], cast(torch.Tensor, decoded.token_ids)], dim=-1)

    assert torch.equal(actual, expected)


def test_server_transformers_call_matches_local() -> None:
    wrapped, inputs = _build_llama31()
    local_model = ti.Model(wrapped)
    expected_proxy = get_proxy(local_model, "model.layers.1.self_attn")
    expected = local_model(**inputs, get=[expected_proxy])

    server = ti.Server(_build_llama31()[0])
    plan = server.compile(get=["model.layers.1.self_attn"])
    actual = server.call(plan, **inputs)

    assert torch.allclose(actual.activations["model.layers.1.self_attn"], expected[expected_proxy])


def test_server_transformers_qwen_prefill_many_decode_matches_generate() -> None:
    wrapped, inputs = _build_qwen35()
    prompt_a = inputs["input_ids"]
    prompt_b = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention = torch.ones_like(torch.cat([prompt_a, prompt_b], dim=0))
    expected_a = cast(
        torch.Tensor,
        wrapped.generate(
            input_ids=prompt_a,
            attention_mask=attention[:1],
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        ),
    )
    expected_b = cast(
        torch.Tensor,
        wrapped.generate(
            input_ids=prompt_b,
            attention_mask=attention[1:],
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        ),
    )

    server = ti.Server(_build_qwen35()[0])
    plan = server.compile(get=["model.layers.1.linear_attn"])
    sessions = [server.open_session(plan=plan, cache="dynamic") for _ in range(2)]
    _ = server.prefill_many(
        sessions,
        input_ids=torch.cat([prompt_a, prompt_b], dim=0),
        attention_mask=attention,
    )
    decoded = server.decode(sessions, max_new_tokens=2, do_sample=False)

    actual_a = torch.cat([prompt_a, cast(torch.Tensor, decoded[0].token_ids)], dim=-1)
    actual_b = torch.cat([prompt_b, cast(torch.Tensor, decoded[1].token_ids)], dim=-1)
    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)


def test_server_transformers_subset_decode_after_prefill_many_matches_generate() -> None:
    wrapped, inputs = _build_llama31()
    prompt_a = inputs["input_ids"]
    prompt_b = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention = torch.ones_like(torch.cat([prompt_a, prompt_b], dim=0))
    expected = cast(
        torch.Tensor,
        wrapped.generate(
            input_ids=prompt_a,
            attention_mask=attention[:1],
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        ),
    )

    server = ti.Server(_build_llama31()[0])
    sessions = [server.open_session(cache="dynamic"), server.open_session(cache="dynamic")]
    _ = server.prefill_many(
        sessions,
        input_ids=torch.cat([prompt_a, prompt_b], dim=0),
        attention_mask=attention,
    )
    decoded = server.decode([sessions[0]], max_new_tokens=2, do_sample=False)[0]
    actual = torch.cat([prompt_a, cast(torch.Tensor, decoded.token_ids)], dim=-1)

    assert torch.equal(actual, expected)
