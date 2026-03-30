"""End-to-end tests for the in-process inference server."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import tinyinterp as ti
from tinyinterp.server.cache import QwenHybridAdapter

from .helpers import (
    FakeDecoderModel,
    FakeLlamaModel,
    FakeTokenizer,
    filter_forward_inputs,
    generate_activation_row,
    generate_row,
    get_proxy,
    request_contract_cases,
)


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


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_server_call_get_matches_local(factory: Callable[[], nn.Module]) -> None:
    wrapped = _seeded_model(factory)
    local_model = ti.Model(wrapped)
    proxy = local_model.layers[0]
    expected = local_model(_input_ids(), get=[proxy])

    server = ti.Server(_seeded_model(factory))
    plan = server.compile(
        get=["transformer.h.0"] if factory is FakeDecoderModel else ["model.layers.0"]
    )
    actual = server.call(plan, _input_ids())

    path = cast(str, plan.get_paths[0])
    assert torch.allclose(actual.activations[path], expected[proxy])
    assert torch.allclose(actual.logits, expected.logits)


def test_server_call_hot_path_allows_overlapping_stateless_requests() -> None:
    torch.manual_seed(0)
    expected_server = ti.Server(BarrierProbeModel(_ThreadGate(1)))
    expected_plan = expected_server.compile(get=["hidden"])
    x_a = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32)
    x_b = torch.tensor([[0.5, 0.6, -0.7, 0.8]], dtype=torch.float32)
    expected_a = expected_server.call(expected_plan, x=x_a).activations["hidden"]
    expected_b = expected_server.call(expected_plan, x=x_b).activations["hidden"]

    torch.manual_seed(0)
    gate = _ThreadGate(2)
    server = ti.Server(BarrierProbeModel(gate))
    plan = server.compile(get=["hidden"])
    outputs: dict[str, torch.Tensor] = {}
    errors: list[Exception] = []

    def worker(name: str, x: torch.Tensor) -> None:
        try:
            outputs[name] = cast(torch.Tensor, server.call(plan, x=x).activations["hidden"])
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("a", x_a)),
        threading.Thread(target=worker, args=("b", x_b)),
    ]
    for thread in threads:
        thread.start()
    assert gate.ready.wait(timeout=5.0)
    gate.release.set()
    for thread in threads:
        thread.join(timeout=3.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert torch.allclose(outputs["a"], expected_a)
    assert torch.allclose(outputs["b"], expected_b)


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_server_collector_map_matches_local(factory: Callable[[], nn.Module]) -> None:
    wrapped = _seeded_model(factory)
    local_model = ti.Model(wrapped)
    inputs = {"input_ids": _input_ids()}
    proxy = ti.find(local_model.layers[0], "attn")
    assert proxy is not None
    expected = local_model(**inputs, map={proxy: ti.zero()}).logits

    server = ti.Server(_seeded_model(factory))
    path = "transformer.h.0.attn" if factory is FakeDecoderModel else "model.layers.0.self_attn"
    plan = server.compile(mapping={path: ti.zero()})
    collector = server.open_collector(plan=plan, stop_at_last_get=False)
    actual = collector.collect_batch(inputs).logits

    assert torch.allclose(cast(torch.Tensor, actual), expected)


def test_server_collector_uses_stop_at_last_get_for_capture_only() -> None:
    server = ti.Server(_seeded_model(FakeDecoderModel))
    plan = server.compile(get=["transformer.h.0"], output={"logits": False, "activations": True})
    collector = server.open_collector(plan=plan)

    result = collector.collect_batch({"input_ids": _input_ids()})

    assert not result.completed_forward
    assert result.logits is None
    assert "transformer.h.0" in result.activations


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_server_generate_matches_wrapped_generate(factory: Callable[[], nn.Module]) -> None:
    wrapped = _seeded_model(factory)
    expected = cast(
        torch.Tensor,
        wrapped.generate(
            _input_ids(),
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
        ),
    )

    server = ti.Server(_seeded_model(factory))
    actual = server.generate(input_ids=_input_ids(), max_new_tokens=2, do_sample=False)

    assert torch.equal(actual, expected)


def test_server_decode_multi_session_matches_single_session_fake_model() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    session_a = server.open_session()
    session_b = server.open_session()
    prompt_a = _input_ids()
    prompt_b = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    _ = server.prefill(session_a, input_ids=prompt_a)
    _ = server.prefill(session_b, input_ids=prompt_b)

    batched = server.decode([session_a, session_b], max_new_tokens=3)

    server_single = ti.Server(_seeded_model(FakeLlamaModel))
    single_a = server_single.generate(input_ids=prompt_a, max_new_tokens=3, do_sample=False)
    single_b = server_single.generate(input_ids=prompt_b, max_new_tokens=3, do_sample=False)

    assert torch.equal(
        torch.cat([prompt_a, cast(torch.Tensor, batched[0].token_ids)], dim=-1),
        single_a,
    )
    assert torch.equal(
        torch.cat([prompt_b, cast(torch.Tensor, batched[1].token_ids)], dim=-1),
        single_b,
    )


def test_server_decode_rejects_empty_session_list() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))

    with pytest.raises(ValueError, match="at least one session"):
        server.decode([])


def test_server_prefill_rejects_empty_prompt() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    session = server.open_session()

    with pytest.raises(ValueError, match="at least one prompt token"):
        server.prefill(session, input_ids=torch.empty((1, 0), dtype=torch.long))


def test_server_stats_track_requests() -> None:
    server = ti.Server(_seeded_model(FakeDecoderModel))
    plan = server.compile(get=["transformer.h.0"])
    collector = server.open_collector(plan=plan)
    _ = collector.collect_batch({"input_ids": _input_ids()})
    session = server.open_session()
    _ = server.prefill(session, input_ids=_input_ids())
    _ = server.decode([session], max_new_tokens=1)
    stats = server.stats()

    assert stats["requests_served"] == 5
    assert stats["active_sessions"] >= 1
    assert stats["active_collectors"] >= 1
    assert "collect_batch" in stats["queues"]
    assert "decode" in stats["queues"]
    assert stats["last_admission"] is not None


def test_server_prefill_many_matches_single_prefill_fake_model() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    sessions = [server.open_session(), server.open_session()]
    prompts = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)

    prefills = server.prefill_many(sessions, input_ids=prompts)
    decoded = server.decode(sessions, max_new_tokens=2)

    single_server = ti.Server(_seeded_model(FakeLlamaModel))
    single_a = single_server.generate(input_ids=prompts[:1], max_new_tokens=2, do_sample=False)
    single_b = single_server.generate(input_ids=prompts[1:], max_new_tokens=2, do_sample=False)

    assert len(prefills) == 2
    assert torch.equal(
        torch.cat([prompts[:1], cast(torch.Tensor, decoded[0].token_ids)], dim=-1), single_a
    )
    assert torch.equal(
        torch.cat([prompts[1:], cast(torch.Tensor, decoded[1].token_ids)], dim=-1), single_b
    )


def test_server_prefill_many_subset_decode_matches_single_session_fake_model() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    sessions = [server.open_session(), server.open_session()]
    prompts = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)

    _ = server.prefill_many(sessions, input_ids=prompts)
    subset = server.decode([sessions[0]], max_new_tokens=2)

    single_server = ti.Server(_seeded_model(FakeLlamaModel))
    single = single_server.generate(input_ids=prompts[:1], max_new_tokens=2, do_sample=False)

    assert torch.equal(
        torch.cat([prompts[:1], cast(torch.Tensor, subset[0].token_ids)], dim=-1),
        single,
    )


def test_server_call_many_matches_single_calls() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    plan = server.compile(get=["model.layers.0.self_attn"])
    requests = [
        {"input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long)},
        {"input_ids": torch.tensor([2, 3, 4, 5], dtype=torch.long)},
    ]

    batched = server.call_many(requests, plan=plan)
    singles = [server.call(plan, request["input_ids"].unsqueeze(0)) for request in requests]

    assert len(batched) == len(singles)
    for actual, expected in zip(batched, singles, strict=True):
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual.activations["model.layers.0.self_attn"],
            expected.activations["model.layers.0.self_attn"],
            atol=1e-6,
            rtol=1e-6,
        )


def test_server_collect_many_matches_single_batches() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    plan = server.compile(get=["model.layers.0"], output={"logits": False, "activations": True})
    collector = server.open_collector(plan=plan, stop_at_last_get=True)
    requests = [
        {"input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long)},
        {"input_ids": torch.tensor([2, 3, 4, 5], dtype=torch.long)},
    ]

    batched = collector.collect_many(requests)
    singles = [
        collector.collect_batch({"input_ids": request["input_ids"].unsqueeze(0)})
        for request in requests
    ]

    assert len(batched) == len(singles)
    for actual, expected in zip(batched, singles, strict=True):
        assert torch.allclose(
            actual.activations["model.layers.0"],
            expected.activations["model.layers.0"],
        )


def test_server_generate_many_matches_individual_generates() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    requests = [
        {"input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long)},
        {"input_ids": torch.tensor([2, 3, 4, 5], dtype=torch.long)},
    ]

    batched = server.generate_many(requests, max_new_tokens=2, do_sample=False)
    singles = [
        cast(torch.Tensor, server.generate(
            input_ids=request["input_ids"].unsqueeze(0),
            max_new_tokens=2,
            do_sample=False,
        ))
        for request in requests
    ]

    assert isinstance(batched, ti.GenerateOutput)
    assert batched.sequences.shape[0] == len(singles)
    for idx, expected in enumerate(singles):
        actual_sequence, actual_generated = generate_row(batched, idx)
        prompt_length = requests[idx]["input_ids"].shape[-1]
        assert torch.equal(actual_sequence, expected)
        assert torch.equal(actual_generated, expected[:, prompt_length:])


def test_server_generate_many_variable_lengths_match_individual_generates() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    requests = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        },
    ]

    batched = server.generate_many(requests, max_new_tokens=2, do_sample=False)
    singles = [
        cast(torch.Tensor, server.generate(
            input_ids=request["input_ids"].unsqueeze(0),
            attention_mask=request["attention_mask"].unsqueeze(0),
            max_new_tokens=2,
            do_sample=False,
        ))
        for request in requests
    ]

    assert isinstance(batched, ti.GenerateOutput)
    assert batched.sequences.shape[0] == len(singles)
    for idx, expected in enumerate(singles):
        actual_sequence, actual_generated = generate_row(batched, idx)
        prompt_length = requests[idx]["input_ids"].shape[-1]
        assert torch.equal(actual_sequence, expected)
        assert torch.equal(actual_generated, expected[:, prompt_length:])


def test_server_generate_many_map_matches_local_batched_generate() -> None:
    requests = [
        {"input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long)},
        {"input_ids": torch.tensor([2, 3, 4, 5], dtype=torch.long)},
    ]
    batched_inputs = torch.stack([request["input_ids"] for request in requests], dim=0)
    attention_mask = torch.ones_like(batched_inputs)

    local = ti.Model(_seeded_model(FakeLlamaModel))
    proxy = get_proxy(local, "model.layers.0.self_attn")
    expected = local.generate(
        input_ids=batched_inputs,
        attention_mask=attention_mask,
        map={proxy: ti.zero()},
        max_new_tokens=2,
        do_sample=False,
    )
    expected_tokens = cast(torch.Tensor, expected.sequences)

    server = ti.Server(_seeded_model(FakeLlamaModel))
    plan = server.compile(
        mapping={"model.layers.0.self_attn": ti.zero()},
        output={"logits": True, "activations": False},
    )
    actual = server.generate_many(requests, plan=plan, max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected_tokens)


def test_server_generate_many_messages_uses_tokenizer() -> None:
    tokenizer = FakeTokenizer()
    server = ti.Server(_seeded_model(FakeLlamaModel), tokenizer=tokenizer)
    requests = [
        [{"role": "user", "content": "abcd"}],
        [{"role": "user", "content": "wxyz"}],
    ]

    batched = server.generate_many(requests, max_new_tokens=2, do_sample=False)
    singles = []
    for messages in requests:
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        encoded = tokenizer(rendered)
        singles.append(
            cast(torch.Tensor, server.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=2,
                do_sample=False,
            ))
        )

    assert isinstance(batched, ti.GenerateOutput)
    assert batched.sequences.shape[0] == len(singles)
    for idx, expected in enumerate(singles):
        actual_sequence, actual_generated = generate_row(batched, idx)
        prompt_length = cast(int, singles[idx].shape[-1] - actual_generated.shape[-1])
        assert torch.equal(actual_sequence, expected)
        assert torch.equal(actual_generated, expected[:, prompt_length:])


def test_server_generate_rejects_non_dynamic_cache_modes() -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))

    with pytest.raises(ValueError, match="cache='dynamic' only"):
        _ = server.generate(input_ids=_input_ids(), cache="static")

    with pytest.raises(ValueError, match="cache='dynamic' only"):
        _ = server.generate_many(
            [{"input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long)}],
            cache="none",
        )


def test_server_collector_reducer_and_mmap(tmp_path: Path) -> None:
    server = ti.Server(_seeded_model(FakeLlamaModel))
    plan = server.compile(get=["model.layers.0"], output={"logits": False, "activations": True})
    collector = server.open_collector(
        plan=plan,
        reducer="mean_tokens",
        mmap_path=str(tmp_path),
        pin_memory=False,
    )

    result = collector.collect_batch({"input_ids": _input_ids()})

    activation = cast(torch.Tensor, result.activations["model.layers.0"])
    assert activation.ndim == 2
    mmap_files = cast(dict[str, str], result.metadata["mmap_files"])
    assert "model.layers.0" in mmap_files
    assert Path(mmap_files["model.layers.0"]).exists()


def test_server_rejects_non_builtin_map_callable() -> None:
    server = ti.Server(_seeded_model(FakeDecoderModel))
    with pytest.raises(TypeError, match="built-in map ops"):
        _ = server.compile(mapping={"transformer.h.0": lambda x: x})


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


# --- Remote ti.Model(unix://...) API tests ---


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


def test_model_server_argument_removed() -> None:
    server = ti.Server(_seeded_model(FakeDecoderModel))
    try:
        with pytest.raises(TypeError, match="ti.Model\\(server\\) was removed"):
            _ = ti.Model(server)
    finally:
        server.close()


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_remote_model_call_matches_local(factory: Callable[[], nn.Module]) -> None:
    """ti.Model(unix://...) forward pass matches local ti.Model(...)."""

    wrapped = _seeded_model(factory)
    local_model = ti.Model(wrapped)
    local_site = local_model.layers[0]
    expected = local_model(_input_ids(), get=[local_site])

    sock = "/tmp/tinyinterp_test_call.sock"
    server, _ = _start_server(lambda: _seeded_model(factory), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        remote_site = remote.layers[0]
        actual = remote(input_ids=_input_ids(), get=[remote_site])

        assert isinstance(remote, ti.Model)
        assert isinstance(actual, ti.Output)
        assert torch.allclose(actual[remote_site], expected[local_site])
        assert torch.allclose(actual.logits, expected.logits)
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_capabilities_report_protocol_and_grad_support() -> None:
    sock = "/tmp/tinyinterp_test_caps.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        assert remote.capabilities["backend"] == "remote"
        assert remote.capabilities["remote"] is True
        assert remote.capabilities["grad"] is True
        assert remote.capabilities["lazy_values"] is True
        assert remote.capabilities["protocol"] == 3
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_remote_model_call_with_map(factory: Callable[[], nn.Module]) -> None:
    wrapped = _seeded_model(factory)
    local_model = ti.Model(wrapped)
    local_site = ti.find(local_model.layers[0], "attn")
    assert local_site is not None
    expected = local_model(_input_ids(), map={local_site: ti.zero()}).logits

    sock = "/tmp/tinyinterp_test_map.sock"
    server, _ = _start_server(lambda: _seeded_model(factory), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = ti.find(remote.layers[0], "attn")
        assert site is not None
        actual = remote(input_ids=_input_ids(), map={site: ti.zero()})

        assert torch.allclose(actual.logits, expected)
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_call_many_prompts() -> None:
    """ti.Model(unix://...) accepts prompt lists."""

    sock = "/tmp/tinyinterp_test_many.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        results = remote(["hello", "world"], get=[site])
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert torch.is_tensor(r[site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_collect_matches_local() -> None:
    """collect(...) matches between local and remote `ti.Model(...)`."""

    wrapped = _seeded_model(FakeDecoderModel)
    local_model = ti.Model(wrapped, tokenizer=FakeTokenizer())
    local_site = local_model.layers[0]
    expected = local_model.collect(["hello", "world"], get=[local_site])

    sock = "/tmp/tinyinterp_test_collect.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        remote_site = remote.layers[0]
        actual = remote.collect(["hello", "world"], get=[remote_site])

        assert len(actual) == len(expected) == 2
        for got, want in zip(actual, expected, strict=True):
            assert isinstance(got, ti.Output)
            assert got.partial
            assert torch.allclose(got[remote_site], want[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_returns_generate_output() -> None:
    """ti.Model(unix://...).generate(input_ids=...) returns GenerateOutput."""

    sock = "/tmp/tinyinterp_test_gen.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        result = remote.generate(input_ids=_input_ids(), max_new_tokens=2)

        assert isinstance(result, ti.GenerateOutput)
        assert result.sequences.shape[-1] == _input_ids().shape[-1] + 2
        assert result.generated_ids.shape[-1] == 2
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_many_prompts() -> None:
    """ti.Model(unix://...).generate(prompts, ...) accepts prompt lists."""

    sock = "/tmp/tinyinterp_test_gen_many.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        results = remote.generate(["hello", "world"], max_new_tokens=2)

        assert isinstance(results, ti.GenerateOutput)
        assert results.sequences.shape[0] == 2
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_empty_request_list_is_rejected_cleanly(tmp_path: Path) -> None:
    sock = str(tmp_path / "tinyinterp_test_empty_requests.sock")
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        with pytest.raises(ValueError, match="Expected at least one request"):
            _ = remote([])
        with pytest.raises(ValueError, match="Expected at least one request"):
            _ = remote.generate([], max_new_tokens=2, do_sample=False)
        with pytest.raises(ValueError, match="Expected at least one request"):
            _ = remote.collect([], get=[site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_generate_max_new_tokens_zero_returns_prompt_only(tmp_path: Path) -> None:
    sock = str(tmp_path / "tinyinterp_test_zero_generate.sock")
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        result = remote.generate("hello", max_new_tokens=0, do_sample=False)

        assert isinstance(result, ti.GenerateOutput)
        assert result.generated_length == 0
        assert result.generated_ids.shape == (1, 0)
        assert result.sequences.shape[-1] == result.prompt_length
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_generate_forwards_sampling_kwargs_to_wrapped_model(tmp_path: Path) -> None:
    sock = str(tmp_path / "tinyinterp_test_sampling_generate.sock")
    wrapped = SamplingGenerateModel()
    server, _ = _start_server(lambda: wrapped, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        result = remote.generate(
            input_ids=_input_ids(),
            max_new_tokens=3,
            do_sample=True,
            temperature=0.7,
            top_k=5,
        )

        assert isinstance(result, ti.GenerateOutput)
        assert wrapped.last_generate_kwargs == {
            "max_new_tokens": 3,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 5,
        }
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_get_matches_local_capture_all() -> None:
    wrapped = _seeded_model(FakeDecoderModel)
    local = ti.Model(wrapped, tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local.generate("hello", max_new_tokens=2, do_sample=False, get=[local_site])

    sock = "/tmp/tinyinterp_test_gen_get.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        remote_site = remote.layers[0]
        actual = remote.generate("hello", max_new_tokens=2, do_sample=False, get=[remote_site])

        assert isinstance(actual, ti.GenerateOutput)
        assert torch.equal(actual.sequences, expected.sequences)
        assert torch.equal(actual.generated_ids, expected.generated_ids)
        assert torch.allclose(actual[remote_site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_get_generated_only_trims_activations() -> None:
    sock = "/tmp/tinyinterp_test_gen_get_generated.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.generate(
            "hello",
            max_new_tokens=2,
            do_sample=False,
            get=[site],
            capture="generated",
        )

        assert isinstance(actual, ti.GenerateOutput)
        assert actual.generated_length == 2
        assert actual[site].shape[1] == 2
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_remote_model_call_request_forms_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local(**filter_forward_inputs(local.wrapped, expected_inputs), get=[local_site])

    sock = "/tmp/tinyinterp_test_remote_call_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote(request_value, get=[site])

        assert isinstance(actual, ti.Output)
        assert torch.allclose(actual[site], expected[local_site])
        assert torch.allclose(actual.logits, expected.logits)
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_remote_model_call_request_forms_with_map_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local(
        **filter_forward_inputs(local.wrapped, expected_inputs),
        get=[local_site],
        map={local_site: ti.zero()},
    )

    sock = "/tmp/tinyinterp_test_remote_call_map_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote(request_value, get=[site], map={site: ti.zero()})

        assert torch.allclose(actual[site], expected[local_site])
        assert torch.allclose(actual.logits, expected.logits)
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_remote_model_call_request_forms_stop_at_last_get_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local(
        **filter_forward_inputs(local.wrapped, expected_inputs),
        get=[local_site],
        stop_at_last_get=True,
    )

    sock = "/tmp/tinyinterp_test_remote_call_stop_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote(request_value, get=[site], stop_at_last_get=True)

        assert actual.partial
        assert torch.allclose(actual[site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
def test_remote_model_generate_request_forms_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    expected = local.generate(**expected_inputs, max_new_tokens=2, do_sample=False)

    sock = "/tmp/tinyinterp_test_remote_generate_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        actual = remote.generate(request_value, max_new_tokens=2, do_sample=False)

        assert isinstance(actual, ti.GenerateOutput)
        assert torch.equal(actual.sequences, expected.sequences)
        assert torch.equal(actual.generated_ids, expected.generated_ids)
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
def test_remote_model_generate_request_forms_with_map_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local.generate(
        **expected_inputs,
        max_new_tokens=2,
        do_sample=False,
        map={local_site: ti.zero()},
    )

    sock = "/tmp/tinyinterp_test_remote_generate_map_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.generate(
            request_value,
            max_new_tokens=2,
            do_sample=False,
            map={site: ti.zero()},
        )

        assert torch.equal(actual.sequences, expected.sequences)
        assert torch.equal(actual.generated_ids, expected.generated_ids)
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_remote_model_collect_request_forms_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local.collect(expected_inputs, get=[local_site])[0]

    sock = "/tmp/tinyinterp_test_remote_collect_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.collect(request_value, get=[site])[0]

        assert torch.allclose(actual[site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_remote_model_collect_request_forms_with_map_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local.collect(
        expected_inputs,
        get=[local_site],
        map={local_site: ti.zero()},
        stop_at_last_get=False,
    )[0]

    sock = "/tmp/tinyinterp_test_remote_collect_map_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.collect(
            request_value,
            get=[site],
            map={site: ti.zero()},
            stop_at_last_get=False,
        )[0]

        assert torch.allclose(actual[site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_collect_map_rejects_fast_path_default() -> None:
    sock = "/tmp/tinyinterp_test_remote_collect_map_error.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        with pytest.raises(ValueError, match="does not support map="):
            _ = remote.collect("hello", get=[site], map={site: ti.zero()})
        remote.close()
    finally:
        _stop_server(server, sock)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
@pytest.mark.parametrize("capture", ["all", "generated"])
def test_remote_model_generate_request_forms_with_capture_match_local(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
    capture: str,
) -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    expected = local.generate(
        **expected_inputs,
        max_new_tokens=2,
        do_sample=False,
        get=[local_site],
        capture=capture,
    )

    sock = "/tmp/tinyinterp_test_remote_generate_capture_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.generate(
            request_value,
            max_new_tokens=2,
            do_sample=False,
            get=[site],
            capture=capture,
        )

        assert torch.equal(actual.sequences, expected.sequences)
        assert torch.equal(actual.generated_ids, expected.generated_ids)
        assert torch.allclose(actual[site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_request_list_contracts_preserve_rows() -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    requests = ["hello", {"text": "world"}, [{"role": "user", "content": "tiny"}]]
    expected = [local.generate(request, max_new_tokens=2, do_sample=False) for request in requests]

    sock = "/tmp/tinyinterp_test_remote_generate_list_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        actual = remote.generate(requests, max_new_tokens=2, do_sample=False)

        assert isinstance(actual, ti.GenerateOutput)
        assert actual.sequences.shape[0] == len(expected) == 3
        for idx, want in enumerate(expected):
            got_sequence, got_generated = generate_row(actual, idx)
            assert torch.equal(got_sequence, want.sequences)
            assert torch.equal(got_generated, want.generated_ids)
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_generate_request_list_with_capture_preserves_rows() -> None:
    local = ti.Model(_seeded_model(FakeDecoderModel), tokenizer=FakeTokenizer())
    local_site = local.layers[0]
    requests = ["hello", {"text": "world"}, [{"role": "user", "content": "tiny"}]]
    expected = [
        local.generate(
            request,
            max_new_tokens=2,
            do_sample=False,
            get=[local_site],
            capture="all",
        )
        for request in requests
    ]

    sock = "/tmp/tinyinterp_test_remote_generate_list_capture_contract.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        actual = remote.generate(
            requests,
            max_new_tokens=2,
            do_sample=False,
            get=[site],
            capture="all",
        )

        assert isinstance(actual, ti.GenerateOutput)
        for idx, want in enumerate(expected):
            got_sequence, got_generated = generate_row(actual, idx)
            got_activation = generate_activation_row(actual, site, idx, capture="all")
            assert torch.equal(got_sequence, want.sequences)
            assert torch.equal(got_generated, want.generated_ids)
            assert torch.allclose(got_activation, want[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_proxy_navigation() -> None:
    """ti.Model(unix://...) supports nested proxy navigation and ti.find()."""

    sock = "/tmp/tinyinterp_test_nav.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        assert remote.transformer.h[0].path == "transformer.h.0"
        attn = ti.find(remote.layers[0], "attn")
        assert attn is not None
        assert attn.path == "transformer.h.0.attn"
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_model_stop_at_last_get_matches_local() -> None:
    """capture-only forwards work through the Unix-socket client."""

    local_model = ti.Model(_seeded_model(FakeDecoderModel))
    local_site = local_model.layers[0]
    expected = local_model(input_ids=_input_ids(), get=[local_site], stop_at_last_get=True)

    sock = "/tmp/tinyinterp_test_stop.sock"
    server, _ = _start_server(lambda: _seeded_model(FakeDecoderModel), sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        remote_site = remote.layers[0]
        actual = remote(input_ids=_input_ids(), get=[remote_site], stop_at_last_get=True)

        assert actual.partial
        assert torch.allclose(actual[remote_site], expected[local_site])
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_outputs_fetch_values_lazily() -> None:
    sock = "/tmp/tinyinterp_test_lazy.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        result = remote(input_ids=_input_ids(), get=[site])

        assert server.stats()["live_remote_values"] >= 2
        _ = result[site]
        assert server.stats()["live_remote_values"] >= 1
        _ = result.logits
        assert server.stats()["live_remote_values"] == 0
        remote.close()
    finally:
        _stop_server(server, sock)


def test_remote_close_releases_unresolved_values() -> None:
    sock = "/tmp/tinyinterp_test_release.sock"
    server, _ = _start_server(FakeDecoderModel, sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        site = remote.layers[0]
        _ = remote(input_ids=_input_ids(), get=[site])
        assert server.stats()["live_remote_values"] >= 2
        remote.close()
        _wait_until(
            lambda: server.stats()["live_remote_values"] == 0,
            timeout_s=5.0,
            message="remote value refs were not released after closing the client",
        )
    finally:
        _stop_server(server, sock)


# --- Memory budget and auto-chunking tests ---


def _ids(batch: int, seq: int = 4) -> torch.Tensor:
    return torch.arange(batch * seq, dtype=torch.long).view(batch, seq) % 16


class TestCpuMemoryDetection:
    def test_detects_positive_value(self) -> None:
        from tinyinterp.server.memory import _cpu_memory_bytes
        assert _cpu_memory_bytes() > 0

    def test_less_than_physical(self) -> None:
        """cgroup limit should be less than raw MemTotal on constrained VMs."""
        from tinyinterp.server.memory import _cpu_memory_bytes
        mem = _cpu_memory_bytes()
        assert mem < 1024 * (1024 ** 3)  # sanity: less than 1TB

    def test_at_least_1gb(self) -> None:
        from tinyinterp.server.memory import _cpu_memory_bytes
        assert _cpu_memory_bytes() > 1024 ** 3


class TestMemoryBudget:
    def test_gpu_budget_positive(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        b = server.budget
        # Fake model on CPU → gpu_budget is 0
        assert b.gpu_budget >= 0

    def test_cpu_budget_positive(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        assert server.budget.cpu_budget > 0

    def test_max_batch_size_decreases_with_seq_len(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        server.budget.gpu_budget = 10_000_000  # 10MB fake budget
        plan = server.compile(get=["transformer.h.0"])
        mb_short = server.budget.max_batch_size(plan, seq_len=16)
        mb_long = server.budget.max_batch_size(plan, seq_len=256)
        assert mb_short >= mb_long

    def test_max_batch_at_least_1(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        server.budget.gpu_budget = 1  # tiny
        plan = server.compile(get=["transformer.h.0"])
        assert server.budget.max_batch_size(plan, seq_len=128) >= 1

    def test_budget_respects_gpu_fraction(self) -> None:
        s1 = ti.Server(_seeded_model(FakeDecoderModel), gpu_fraction=0.5)
        s2 = ti.Server(_seeded_model(FakeDecoderModel), gpu_fraction=0.9)
        # Both on CPU so budgets are 0, but fractions stored correctly
        assert s1._gpu_fraction == 0.5
        assert s2._gpu_fraction == 0.9

    def test_budget_respects_cpu_fraction(self) -> None:
        s1 = ti.Server(_seeded_model(FakeDecoderModel), cpu_fraction=0.3)
        s2 = ti.Server(_seeded_model(FakeDecoderModel), cpu_fraction=0.8)
        b1 = s1.budget.cpu_budget
        b2 = s2.budget.cpu_budget
        assert b2 > b1

    def test_estimate_cpu_bytes_scales_with_sites(self) -> None:
        server = ti.Server(_seeded_model(FakeLlamaModel))
        b = server.budget
        plan_1 = server.compile(get=["model.layers.0"])
        plan_2 = server.compile(get=["model.layers.0", "model.layers.1"])
        est_1 = b.estimate_cpu_bytes(plan_1, batch_size=4, seq_len=128)
        est_2 = b.estimate_cpu_bytes(plan_2, batch_size=4, seq_len=128)
        # 2 sites should cost more than 1 site
        assert est_2 >= est_1

    def test_estimate_cpu_bytes_scales_with_batch(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        b = server.budget
        plan = server.compile(get=["transformer.h.0"])
        est_4 = b.estimate_cpu_bytes(plan, batch_size=4, seq_len=128)
        est_16 = b.estimate_cpu_bytes(plan, batch_size=16, seq_len=128)
        assert est_16 == est_4 * 4


class TestAutoChunk:
    def test_no_chunk_when_fits(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(4, 8)
        chunks = auto_chunk(ids, max_batch=10)
        assert len(chunks) == 1
        assert torch.equal(chunks[0]["input_ids"], ids)

    def test_chunks_when_exceeds(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(8, 4)
        chunks = auto_chunk(ids, max_batch=3)
        assert len(chunks) == 3  # 3+3+2
        total = sum(c["input_ids"].shape[0] for c in chunks)
        assert total == 8

    def test_chunks_preserve_data(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(6, 4)
        chunks = auto_chunk(ids, max_batch=2)
        reconstructed = torch.cat([c["input_ids"] for c in chunks], dim=0)
        assert torch.equal(reconstructed, ids)

    def test_extra_tensors_chunked_too(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(6, 4)
        mask = torch.ones(6, 4, dtype=torch.long)
        chunks = auto_chunk(ids, max_batch=2, extra_tensors={"attention_mask": mask})
        assert all("attention_mask" in c for c in chunks)
        assert chunks[0]["attention_mask"].shape[0] == 2

    def test_max_batch_1(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(3, 4)
        chunks = auto_chunk(ids, max_batch=1)
        assert len(chunks) == 3
        assert all(c["input_ids"].shape[0] == 1 for c in chunks)


class TestRemoteModelAutoChunking:
    def test_auto_chunks_large_batch(self) -> None:
        """Auto-chunking preserves the original batched output contract."""

        sock = "/tmp/tinyinterp_test_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(4, 4), get=[site])
            assert not isinstance(result, list)
            assert result[site].shape[0] == 4
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_auto_chunk_activations_correct(self) -> None:
        """Chunked results match unchunked results."""

        sock = "/tmp/tinyinterp_test_chunk_correct.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            ids = _ids(4, 4)

            server._budget = None
            server.budget.gpu_budget = 10 ** 9
            unchunked = remote(input_ids=ids, get=[site])

            server._budget = None
            server.budget.gpu_budget = 1
            chunked = remote(input_ids=ids, get=[site])

            assert not isinstance(unchunked, list)
            assert not isinstance(chunked, list)
            assert torch.allclose(chunked[site], unchunked[site])
            assert torch.allclose(
                cast(torch.Tensor, chunked.logits),
                cast(torch.Tensor, unchunked.logits),
            )
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_no_chunk_when_fits(self) -> None:
        """Small batch doesn't get chunked."""

        sock = "/tmp/tinyinterp_test_no_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 10 ** 9
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(2, 4), get=[site])
            assert not isinstance(result, list)
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_auto_chunks(self) -> None:
        """stop_at_last_get collection also auto-chunks."""

        sock = "/tmp/tinyinterp_test_collect_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(4, 4), get=[site], stop_at_last_get=True)
            assert not isinstance(result, list)
            assert result.partial
            assert result[site].shape[0] == 4
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_chunks_have_activations(self) -> None:
        """Chunked collection stitches activations back into one batch."""

        sock = "/tmp/tinyinterp_test_collect_shape.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(3, 4), get=[site], stop_at_last_get=True)
            assert not isinstance(result, list)
            act = result[site]
            assert act.shape[0] == 3
            assert act.ndim == 3
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_cpu_default_batch_not_chunked(self) -> None:
        """CPU-only servers keep the normal batched tensor contract by default."""

        sock = "/tmp/tinyinterp_test_cpu_batch.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(2, 4), get=[site])
            assert not isinstance(result, list)
            assert result[site].shape[0] == 2
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_fast_path_closes_collectors(self) -> None:
        """Internal collector fast path should not leak collector handles."""

        sock = "/tmp/tinyinterp_test_collectors.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            assert len(server._collectors) == 0
            _ = remote(input_ids=_ids(3, 4), get=[site], stop_at_last_get=True)
            assert len(server._collectors) == 0
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_grad_matches_local_for_tensor_inputs(self) -> None:
        sock = "/tmp/tinyinterp_test_grad.sock"
        local_model = ti.Model(_seeded_model(GradProbeModel))
        local_site = local_model.hidden
        local_input = torch.tensor(
            [[0.25, -0.5, 0.75, 1.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        expected = local_model(x=local_input, get=[local_site], grad=True)
        expected_value = expected[local_site].detach().clone()
        expected_logits = expected.logits.detach().clone()
        upstream = torch.full_like(expected_value, 0.25)
        expected[local_site].backward(upstream)
        expected_site_grad = cast(torch.Tensor, expected[local_site].grad).detach().clone()
        expected_input_grad = cast(torch.Tensor, local_input.grad).detach().clone()

        server, _ = _start_server(lambda: _seeded_model(GradProbeModel), sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.hidden
            remote_input = local_input.detach().clone().requires_grad_(True)
            actual = remote(x=remote_input, get=[site], grad=True)

            assert torch.allclose(actual[site], expected_value)
            assert torch.allclose(actual.logits, expected_logits)
            actual[site].backward(upstream)
            assert torch.allclose(actual[site].grad, expected_site_grad)
            assert torch.allclose(actual.input_grads["x"], expected_input_grad)
            actual.release()
            assert server.stats()["live_remote_grads"] == 0
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_grad_rejects_prompt_requests(self) -> None:
        sock = "/tmp/tinyinterp_test_grad_text.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            with pytest.raises(ValueError, match="raw tensor model inputs"):
                _ = remote("hello", get=[site], grad=True)
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_grad_handle_release_makes_future_fetches_fail(self) -> None:
        sock = "/tmp/tinyinterp_test_grad_release.sock"
        server, _ = _start_server(lambda: _seeded_model(GradProbeModel), sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.hidden
            actual = remote(
                x=torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32, requires_grad=True),
                get=[site],
                grad=True,
            )
            actual.release()
            assert server.stats()["live_remote_grads"] == 0
            with pytest.raises(RuntimeError, match="already released"):
                _ = actual[site].shape
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_remote_close_cleans_up_live_grad_handles(self) -> None:
        sock = "/tmp/tinyinterp_test_grad_disconnect.sock"
        server, _ = _start_server(lambda: _seeded_model(GradProbeModel), sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.hidden
            _ = remote(
                x=torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32, requires_grad=True),
                get=[site],
                grad=True,
            )
            assert server.stats()["live_remote_grads"] == 1
            remote.close()
            _wait_until(
                lambda: server.stats()["live_remote_grads"] == 0,
                timeout_s=5.0,
                interval_s=0.01,
                message="remote grad refs were not released after closing the client",
            )
        finally:
            _stop_server(server, sock)

    def test_stop_at_last_get_validation_matches_local(self) -> None:
        """Unsupported stop_at_last_get combinations should raise instead of being ignored."""

        sock = "/tmp/tinyinterp_test_validate.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            with pytest.raises(ValueError, match="requires at least one get= site"):
                _ = remote(input_ids=_input_ids(), stop_at_last_get=True)
            with pytest.raises(ValueError, match="does not support map="):
                _ = remote(
                    input_ids=_input_ids(),
                    get=[site],
                    map={site: ti.zero()},
                    stop_at_last_get=True,
                )
            with pytest.raises(ValueError, match="does not support grad=True"):
                _ = remote(input_ids=_input_ids(), get=[site], grad=True, stop_at_last_get=True)
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_single_example_not_chunked(self) -> None:
        """batch=1 is never chunked regardless of budget."""

        sock = "/tmp/tinyinterp_test_single.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(1, 4), get=[site])
            assert not isinstance(result, list)
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_prompt_list_not_affected_by_chunking(self) -> None:
        """List-of-strings path goes through call_many, not auto-chunking."""

        sock = "/tmp/tinyinterp_test_prompts.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            results = remote(["hello", "world"], get=[site])
            assert isinstance(results, list)
            assert len(results) == 2
            remote.close()
        finally:
            _stop_server(server, sock)


def test_model_tmp_path_is_loaded_like_a_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local checkpoint directory in /tmp should not be mistaken for a socket."""
    import tinyinterp.model as model_mod

    checkpoint = tmp_path / "checkpoint.sock"
    checkpoint.mkdir()
    wrapped = _seeded_model(FakeDecoderModel)
    monkeypatch.setattr(model_mod, "_load_model", lambda path, **_: wrapped)
    monkeypatch.setattr(model_mod, "_maybe_load_tokenizer", lambda _: None)

    model = ti.Model(str(checkpoint))

    assert isinstance(model, model_mod.Model)
    assert model.wrapped is wrapped
