"""Core server runtime tests."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn

import mirin as ti

from .helpers import (
    FakeDecoderModel,
    FakeLlamaModel,
    FakeTokenizer,
    generate_row,
    get_proxy,
)
from .server_helpers import BarrierProbeModel, _input_ids, _seeded_model, _ThreadGate


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
