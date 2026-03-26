"""Optional integration tests against tiny HuggingFace models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import tinyinterp as ti
from tinyinterp.hooks import _extract, _replace

from .helpers import get_module, get_proxy

transformers = pytest.importorskip(
    "transformers",
    reason="Install optional test dependency with `uv sync --extra transformers`.",
)

GPT2Config = transformers.GPT2Config
GPT2LMHeadModel = transformers.GPT2LMHeadModel
LlamaConfig = transformers.LlamaConfig
LlamaForCausalLM = transformers.LlamaForCausalLM


def _build_gpt2() -> tuple[torch.nn.Module, dict[str, Any]]:
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    return model, inputs


def _build_llama() -> tuple[torch.nn.Module, dict[str, Any]]:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    return model, inputs


@pytest.mark.parametrize("builder", [_build_gpt2, _build_llama])
def test_transformers_passthrough_matches_wrapped_model(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
) -> None:
    torch.manual_seed(0)
    wrapped, inputs = builder()
    model = ti.Model(wrapped)

    with torch.no_grad():
        expected = wrapped(**inputs)
        actual = model(**inputs)

    assert type(actual) is type(expected)
    assert torch.allclose(actual.logits, expected.logits)


def test_transformers_model_loads_from_local_path(tmp_path: Path) -> None:
    torch.manual_seed(0)
    wrapped, inputs = _build_gpt2()
    save_path = tmp_path / "gpt2"
    cast(Any, wrapped).save_pretrained(save_path)

    model = ti.Model(str(save_path))
    actual = model(**inputs)
    loaded = cast(Any, model.wrapped)

    assert model.wrapped.__class__.__name__ == wrapped.__class__.__name__
    assert torch.allclose(actual.logits, loaded(**inputs).logits)


@pytest.mark.parametrize(
    ("builder", "path", "layers_path"),
    [
        (_build_gpt2, "transformer.h.1.attn", "transformer.h.1"),
        (_build_llama, "model.layers.1.self_attn", "model.layers.1"),
    ],
)
def test_transformers_navigation_and_get_match_manual_hook(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
    path: str,
    layers_path: str,
) -> None:
    wrapped, inputs = builder()
    model = ti.Model(wrapped)

    captured: dict[str, torch.Tensor] = {}

    def capture(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        captured["act"] = _extract(output).detach()

    handle = get_module(wrapped, path).register_forward_hook(capture)
    try:
        with torch.no_grad():
            _ = wrapped(**inputs)
    finally:
        handle.remove()

    proxy = get_proxy(model, path)
    output = model(**inputs, get=[proxy])

    assert model.layers[1].path == layers_path
    assert torch.allclose(output[proxy], captured["act"])


@pytest.mark.parametrize(
    ("builder", "path"),
    [
        (_build_gpt2, "transformer.h.1.attn"),
        (_build_llama, "model.layers.1.self_attn"),
    ],
)
def test_transformers_map_matches_manual_hook(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
    path: str,
) -> None:
    wrapped, inputs = builder()

    def manual_zero(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
        return _replace(output, torch.zeros_like(_extract(output)))

    handle = get_module(wrapped, path).register_forward_hook(manual_zero)
    try:
        with torch.no_grad():
            expected = wrapped(**inputs).logits
    finally:
        handle.remove()

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    with torch.no_grad():
        actual = model(**inputs, map={proxy: ti.zero()}).logits

    assert torch.allclose(actual, expected)


def test_transformers_renames_enable_canonical_gpt2_access() -> None:
    wrapped, _inputs = _build_gpt2()
    model = ti.Model(wrapped, rename=ti.renames.llm)

    assert model.model.layers[0].self_attn == model.transformer.h[0].attn


def test_transformers_generate_matches_wrapped_model() -> None:
    torch.manual_seed(0)
    wrapped, inputs = _build_gpt2()
    model = ti.Model(wrapped)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 2,
        "do_sample": False,
        "use_cache": False,
    }

    with torch.no_grad():
        expected = cast(Any, wrapped).generate(**generate_kwargs)
        actual = model.generate(**generate_kwargs)

    assert torch.equal(actual, expected)


def test_transformers_generate_map_matches_manual_hook() -> None:
    torch.manual_seed(0)
    wrapped, inputs = _build_gpt2()
    model = ti.Model(wrapped)
    path = "transformer.h.0.attn"

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 2,
        "do_sample": False,
        "use_cache": False,
    }

    def manual_zero(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
        return _replace(output, torch.zeros_like(_extract(output)))

    module = get_module(wrapped, path)
    handle = module.register_forward_hook(manual_zero)
    try:
        with torch.no_grad():
            expected = cast(Any, wrapped).generate(**generate_kwargs)
    finally:
        handle.remove()

    actual = model.generate(**generate_kwargs, map={get_proxy(model, path): ti.zero()})

    assert torch.equal(actual._model_output, expected)
