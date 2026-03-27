"""Phase 2 architecture coverage tests for tiny HuggingFace causal LMs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch

import tinyinterp as ti
from tinyinterp.hooks import _extract, _replace

from .helpers import get_module

transformers = pytest.importorskip(
    "transformers",
    reason="Install optional test dependency with `uv sync --extra transformers`.",
)

AutoModelForCausalLM = transformers.AutoModelForCausalLM
BloomConfig = transformers.BloomConfig
FalconConfig = transformers.FalconConfig
Gemma3Config = transformers.Gemma3Config
Gemma3TextConfig = transformers.Gemma3TextConfig
GPTNeoConfig = transformers.GPTNeoConfig
GPTNeoXConfig = transformers.GPTNeoXConfig
LlamaConfig = transformers.LlamaConfig
MistralConfig = transformers.MistralConfig
OPTConfig = transformers.OPTConfig
PhiConfig = transformers.PhiConfig
Qwen3_5TextConfig = transformers.Qwen3_5TextConfig
StableLmConfig = transformers.StableLmConfig


@dataclass(frozen=True)
class FamilySpec:
    name: str
    make_config: Callable[[], Any]
    layers_path: str
    find_pattern: str
    find_path: str


def _tiny_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }


def _generate_kwargs(model: torch.nn.Module) -> dict[str, Any]:
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    return {
        **_tiny_inputs(),
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": pad_token_id,
    }


def _family_specs() -> list[FamilySpec]:
    return [
        FamilySpec(
            name="llama",
            make_config=lambda: LlamaConfig(
                vocab_size=64,
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
            ),
            layers_path="model.layers.0",
            find_pattern="self_attn",
            find_path="model.layers.0.self_attn",
        ),
        FamilySpec(
            name="gpt_neox",
            make_config=lambda: GPTNeoXConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=16,
                bos_token_id=1,
                eos_token_id=2,
                use_cache=False,
                attention_bias=True,
            ),
            layers_path="gpt_neox.layers.0",
            find_pattern="attention",
            find_path="gpt_neox.layers.0.attention",
        ),
        FamilySpec(
            name="gpt_neo",
            make_config=lambda: GPTNeoConfig(
                vocab_size=64,
                hidden_size=16,
                num_layers=2,
                num_heads=2,
                intermediate_size=32,
                max_position_embeddings=16,
                attention_types=[[["global"], 2]],
                bos_token_id=1,
                eos_token_id=2,
                use_cache=False,
            ),
            layers_path="transformer.h.0",
            find_pattern="attn",
            find_path="transformer.h.0.attn",
        ),
        FamilySpec(
            name="mistral",
            make_config=lambda: MistralConfig(
                vocab_size=64,
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
            ),
            layers_path="model.layers.0",
            find_pattern="self_attn",
            find_path="model.layers.0.self_attn",
        ),
        FamilySpec(
            name="gemma3",
            make_config=lambda: Gemma3Config(
                text_config=Gemma3TextConfig(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    max_position_embeddings=16,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                    use_cache=False,
                    attn_implementation="eager",
                    layer_types=["full_attention", "full_attention"],
                )
            ),
            layers_path="model.language_model.layers.0",
            find_pattern="self_attn",
            find_path="model.language_model.layers.0.self_attn",
        ),
        FamilySpec(
            name="phi",
            make_config=lambda: PhiConfig(
                vocab_size=64,
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
            ),
            layers_path="model.layers.0",
            find_pattern="self_attn",
            find_path="model.layers.0.self_attn",
        ),
        FamilySpec(
            name="bloom",
            make_config=lambda: BloomConfig(
                vocab_size=64,
                hidden_size=16,
                n_layer=2,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                attention_softmax_in_fp32=False,
                pretraining_tp=1,
            ),
            layers_path="transformer.h.0",
            find_pattern="self_attention",
            find_path="transformer.h.0.self_attention",
        ),
        FamilySpec(
            name="opt",
            make_config=lambda: OPTConfig(
                vocab_size=64,
                hidden_size=16,
                ffn_dim=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=16,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                use_cache=False,
                do_layer_norm_before=True,
            ),
            layers_path="model.decoder.layers.0",
            find_pattern="self_attn",
            find_path="model.decoder.layers.0.self_attn",
        ),
        FamilySpec(
            name="qwen35",
            make_config=lambda: Qwen3_5TextConfig(
                vocab_size=64,
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
                layer_types=["full_attention", "full_attention"],
            ),
            layers_path="model.layers.0",
            find_pattern="self_attn",
            find_path="model.layers.0.self_attn",
        ),
        FamilySpec(
            name="falcon",
            make_config=lambda: FalconConfig(
                vocab_size=64,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_kv_heads=2,
                max_position_embeddings=16,
                bos_token_id=1,
                eos_token_id=2,
                use_cache=False,
                new_decoder_architecture=True,
                alibi=False,
            ),
            layers_path="transformer.h.0",
            find_pattern="self_attention",
            find_path="transformer.h.0.self_attention",
        ),
        FamilySpec(
            name="stablelm",
            make_config=lambda: StableLmConfig(
                vocab_size=64,
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
            ),
            layers_path="model.layers.0",
            find_pattern="self_attn",
            find_path="model.layers.0.self_attn",
        ),
    ]


FAMILY_SPECS = _family_specs()


def _build_model(spec: FamilySpec) -> torch.nn.Module:
    torch.manual_seed(0)
    model = cast(torch.nn.Module, AutoModelForCausalLM.from_config(spec.make_config()))
    model.eval()
    return model


@pytest.mark.parametrize("spec", FAMILY_SPECS, ids=lambda spec: spec.name)
def test_phase2_architectures_support_layers_find_and_get(spec: FamilySpec) -> None:
    raw_model = _build_model(spec)
    model = ti.Model(raw_model)
    inputs = _tiny_inputs()

    with torch.no_grad():
        expected = raw_model(**inputs)
        actual = model(**inputs)

    assert torch.allclose(actual.logits, expected.logits)
    assert len(model.layers) == 2
    assert model.layers[0].path == spec.layers_path

    site = ti.find(model.layers[0], spec.find_pattern)
    assert site is not None
    assert site.path == spec.find_path

    captured = model(**inputs, get=[site])
    activation = captured[site]

    assert activation.shape[0] == inputs["input_ids"].shape[0]
    assert activation.shape[1] == inputs["input_ids"].shape[1]


@pytest.mark.parametrize("spec", FAMILY_SPECS, ids=lambda spec: spec.name)
def test_phase2_architectures_support_generate_map(spec: FamilySpec) -> None:
    raw_model = _build_model(spec)
    model = ti.Model(raw_model)
    generate_kwargs = _generate_kwargs(raw_model)
    site = cast(Any, ti.find(model.layers[0], spec.find_pattern))
    assert site is not None

    def manual_zero(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
        return _replace(output, torch.zeros_like(_extract(output)))

    handle = get_module(raw_model, spec.find_path).register_forward_hook(manual_zero)
    try:
        with torch.no_grad():
            expected = cast(Any, raw_model).generate(**generate_kwargs)
    finally:
        handle.remove()

    with torch.no_grad():
        actual = model.generate(**generate_kwargs, map={site: ti.zero()})

    assert torch.equal(actual._model_output, expected)
