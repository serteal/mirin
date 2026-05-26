"""Optional integration tests against tiny HuggingFace model families."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import mirin as ti
from mirin.hooks import _extract, _replace

from .helpers import get_module, get_proxy

transformers = pytest.importorskip(
    "transformers",
    reason="Install optional test dependency with `uv sync --extra transformers`.",
)

AutoConfig = transformers.AutoConfig
AutoModelForCausalLM = transformers.AutoModelForCausalLM
Gemma3Config = transformers.Gemma3Config
Gemma3TextConfig = transformers.Gemma3TextConfig
Qwen3_5TextConfig = getattr(transformers, "Qwen3_5TextConfig", None)

LLAMA31_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA3_MODEL_NAME = "google/gemma-3-4b-it"
QWEN35_MODEL_NAME = "Qwen/Qwen3.5-4B"


def _tiny_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }


def _build_llama31() -> tuple[torch.nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(LLAMA31_MODEL_NAME)
    _configure_text_config(config)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model, _tiny_inputs()


def _build_gemma3() -> tuple[torch.nn.Module, dict[str, Any]]:
    config = Gemma3Config(
        text_config=Gemma3TextConfig(
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
            use_cache=False,
            attn_implementation="eager",
            layer_types=["full_attention", "full_attention"],
        )
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model, _tiny_inputs()


def _build_qwen35() -> tuple[torch.nn.Module, dict[str, Any]]:
    if Qwen3_5TextConfig is None:
        pytest.skip("transformers build does not expose Qwen3_5TextConfig.")
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
        use_cache=False,
        attn_implementation="eager",
        layer_types=["full_attention", "full_attention"],
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model, _tiny_inputs()


_TRANSFORMER_BUILDERS = [
    _build_llama31,
    _build_gemma3,
    *([] if Qwen3_5TextConfig is None else [_build_qwen35]),
]
_NAV_CASES = [
    (_build_llama31, "model.layers.1.self_attn", "model.layers.1"),
    (_build_gemma3, "model.language_model.layers.1.self_attn", "model.language_model.layers.1"),
]
_PATH_CASES_L1 = [
    (_build_llama31, "model.layers.1.self_attn"),
    (_build_gemma3, "model.language_model.layers.1.self_attn"),
]
_PATH_CASES_L0 = [
    (_build_llama31, "model.layers.0.self_attn"),
    (_build_gemma3, "model.language_model.layers.0.self_attn"),
]
_LAYER_CASES = [
    (_build_llama31, "model.layers.0.self_attn"),
    (_build_gemma3, "model.language_model.layers.0.self_attn"),
]
if Qwen3_5TextConfig is not None:
    _NAV_CASES.append((_build_qwen35, "model.layers.1.self_attn", "model.layers.1"))
    _PATH_CASES_L1.append((_build_qwen35, "model.layers.1.self_attn"))
    _PATH_CASES_L0.append((_build_qwen35, "model.layers.0.self_attn"))
    _LAYER_CASES.append((_build_qwen35, "model.layers.0.self_attn"))


@pytest.mark.parametrize("builder", _TRANSFORMER_BUILDERS)
def test_transformers_passthrough_matches_wrapped_model(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
) -> None:
    torch.manual_seed(0)
    wrapped, inputs = builder()
    model = ti.Model(wrapped)

    with torch.no_grad():
        expected = wrapped(**inputs)
        actual = model(**inputs)

    assert isinstance(actual, ti.Output)
    assert type(actual._model_output) is type(expected)
    assert torch.allclose(actual.logits, expected.logits)


def test_transformers_model_loads_from_local_path(tmp_path: Path) -> None:
    torch.manual_seed(0)
    wrapped, inputs = _build_llama31()
    save_path = tmp_path / "llama31"
    cast(Any, wrapped).save_pretrained(save_path)

    model = ti.Model(str(save_path))
    actual = model(**inputs)
    loaded = cast(Any, model.wrapped)

    assert model.wrapped.__class__.__name__ == wrapped.__class__.__name__
    assert torch.allclose(actual.logits, loaded(**inputs).logits)


@pytest.mark.parametrize(
    ("builder", "path", "layers_path"),
    _NAV_CASES,
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

    target_layer = model.layers[min(1, len(model.layers) - 1)]
    assert target_layer.path == layers_path
    assert torch.allclose(output[proxy], captured["act"])


@pytest.mark.parametrize(
    ("builder", "path"),
    _PATH_CASES_L1,
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


@pytest.mark.parametrize(
    ("builder", "layers_path"),
    _LAYER_CASES,
)
def test_transformers_layers_shortcut_finds_requested_families(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
    layers_path: str,
) -> None:
    wrapped, _inputs = builder()
    model = ti.Model(wrapped, rename=ti.renames.llm)
    site = ti.find(model.layers[0], "attn")
    assert site is not None
    assert site.path == layers_path


@pytest.mark.parametrize("builder", _TRANSFORMER_BUILDERS)
def test_transformers_generate_matches_wrapped_model(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
) -> None:
    torch.manual_seed(0)
    wrapped, inputs = builder()
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

    assert isinstance(actual, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected)
    assert torch.equal(actual.generated_ids, expected[:, actual.prompt_length :])


def test_transformers_generate_rejects_stop_at_last_get() -> None:
    wrapped, inputs = _build_llama31()
    model = ti.Model(wrapped)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 2,
        "do_sample": False,
        "use_cache": False,
    }

    with pytest.raises(ValueError, match="stop_at_last_get=True"):
        _ = model.generate(**generate_kwargs, stop_at_last_get=True)


def test_transformers_generate_get_capture_all_exposes_sequences_and_activations() -> None:
    wrapped, inputs = _build_llama31()
    model = ti.Model(wrapped)
    site = get_proxy(model, "model.layers.0.self_attn")

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 2,
        "do_sample": False,
        "use_cache": False,
    }

    with torch.no_grad():
        expected = cast(Any, wrapped).generate(**generate_kwargs)
        actual = model.generate(**generate_kwargs, get=[site], capture="all")

    assert isinstance(actual, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected)
    assert actual[site].shape[1] == expected.shape[-1]


@pytest.mark.parametrize(
    ("builder", "path"),
    _PATH_CASES_L0,
)
def test_transformers_generate_map_matches_manual_hook(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped, inputs = builder()
    model = ti.Model(wrapped)

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

    assert isinstance(actual, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected)


@pytest.mark.parametrize(
    ("builder", "path"),
    _PATH_CASES_L0,
)
def test_transformers_batch_fused_outputs_expose_logits(
    builder: Callable[[], tuple[torch.nn.Module, dict[str, Any]]],
    path: str,
) -> None:
    wrapped, inputs = builder()
    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)

    with torch.no_grad():
        expected_zero = model(**inputs, map={proxy: ti.zero()}).logits
        expected_shift = model(**inputs, map={proxy: ti.add(1.0)}).logits

    with ti.batch():
        out_zero = model(**inputs, map={proxy: ti.zero()})
        out_shift = model(**inputs, map={proxy: ti.add(1.0)})

    assert torch.allclose(out_zero.logits, expected_zero, atol=1e-7)
    assert torch.allclose(out_shift.logits, expected_shift, atol=1e-7)


def _configure_text_config(config: Any) -> None:
    if hasattr(config, "vocab_size"):
        config.vocab_size = 64
    if hasattr(config, "hidden_size"):
        config.hidden_size = 16
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = 32
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 2
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = 2
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = 2
    if hasattr(config, "head_dim"):
        config.head_dim = 8
    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = 32
    if hasattr(config, "bos_token_id"):
        config.bos_token_id = 1
    if hasattr(config, "eos_token_id"):
        config.eos_token_id = 2
    if hasattr(config, "pad_token_id"):
        config.pad_token_id = 0
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "layer_types"):
        config.layer_types = ["full_attention"] * int(config.num_hidden_layers)
