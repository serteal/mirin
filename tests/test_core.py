"""Core tests for the proxy-based mirin API."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import mirin as ti

from .helpers import (
    FakeDecoderModel,
    FakeLlamaModel,
    FakeTokenizer,
    filter_forward_inputs,
    generate_activation_row,
    generate_row,
    get_module,
    get_proxy,
    request_contract_cases,
)


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


class _BarrierBlock(nn.Module):
    def __init__(self, gate: _ThreadGate) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gate.wait()
        return x


class _ConcurrentProbeModel(nn.Module):
    def __init__(self, gate: _ThreadGate) -> None:
        super().__init__()
        self.entry = _BarrierBlock(gate)
        self.hidden = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.entry(x)
        hidden = self.hidden(x)
        return {"logits": self.readout(torch.tanh(hidden))}


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


class _SharedLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        first = self.shared(x)
        second = self.shared(first)
        return {"logits": self.readout(torch.tanh(second))}


class _ThreadedDispatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        box: dict[str, torch.Tensor] = {}

        def worker() -> None:
            box["hidden"] = self.hidden(x)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        return {"logits": self.readout(torch.tanh(box["hidden"]))}


class _SamplingGenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.readout = nn.Linear(4, 16, bias=False)
        self.last_generate_kwargs: dict[str, Any] | None = None

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
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


class _GradMapModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4, bias=False)
        self.readout = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.hidden(x)
        return {"logits": self.readout(hidden)}


def _direct_site_activation(wrapped: nn.Module, path: str, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(wrapped, FakeDecoderModel) and path == "transformer.h.1.attn":
        hidden = wrapped.embed(inputs)
        hidden = wrapped.transformer.h[0](hidden)[0]
        return wrapped.transformer.h[1].attn(hidden)[0].detach()
    if isinstance(wrapped, FakeLlamaModel) and path == "model.layers.1.self_attn":
        hidden = wrapped.embed(inputs)
        hidden = wrapped.model.layers[0](hidden)
        return wrapped.model.layers[1].self_attn(hidden)[0].detach()
    raise AssertionError(f"Unhandled direct activation path {path!r} for {type(wrapped).__name__}.")


def _run_with_zeroed_module_output(
    wrapped: nn.Module,
    path: str,
    inputs: torch.Tensor,
) -> torch.Tensor:
    module = get_module(wrapped, path)
    original_forward = module.forward

    def zero_forward(*args: object, **kwargs: object) -> object:
        output = original_forward(*args, **kwargs)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (torch.zeros_like(output[0]), *output[1:])
        if isinstance(output, torch.Tensor):
            return torch.zeros_like(output)
        raise TypeError(f"Unsupported zero-forward output {type(output).__name__}.")

    module.forward = zero_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            return wrapped(inputs).logits.detach()
    finally:
        module.forward = original_forward  # type: ignore[method-assign]


@pytest.mark.parametrize("factory", [FakeDecoderModel, FakeLlamaModel])
def test_passthrough_matches_wrapped_model(factory: Callable[[], nn.Module]) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    model = ti.Model(wrapped)

    with torch.no_grad():
        expected = wrapped(_input_ids())
        actual = model(_input_ids())

    assert isinstance(actual, ti.Output)
    assert type(actual._model_output) is type(expected)
    assert torch.allclose(actual.logits, expected.logits)


def test_model_call_accepts_text_requests() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]
    encoded = model.tokenizer("hello", return_tensors="pt")

    actual = model("hello", get=[site])
    expected = model(input_ids=encoded["input_ids"], get=[site])

    assert isinstance(actual, ti.Output)
    assert torch.allclose(actual[site], expected[site])
    assert torch.allclose(actual.logits, expected.logits)


def test_model_rejects_non_module_objects() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        ti.Model(object())


def test_model_call_accepts_request_lists() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model(["hello", "world"], get=[site])
    expected = [model("hello", get=[site]), model("world", get=[site])]

    assert isinstance(actual, list)
    assert len(actual) == 2
    for got, want in zip(actual, expected, strict=True):
        assert torch.allclose(got[site], want[site])
        assert torch.allclose(got.logits, want.logits)


def test_model_generate_accepts_text_requests() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    encoded = model.tokenizer("hello", return_tensors="pt")

    actual = model.generate("hello", max_new_tokens=2, do_sample=False)
    expected = model.generate(**encoded, max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert isinstance(expected, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected.sequences)
    assert torch.equal(actual.generated_ids, expected.generated_ids)


def test_model_generate_accepts_request_lists() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())

    actual = model.generate(["hello", "world"], max_new_tokens=2, do_sample=False)
    expected = [
        model.generate("hello", max_new_tokens=2, do_sample=False),
        model.generate("world", max_new_tokens=2, do_sample=False),
    ]

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.sequences.shape[0] == 2
    for idx, want in enumerate(expected):
        got_sequence, got_generated = generate_row(actual, idx)
        assert isinstance(want, ti.GenerateOutput)
        assert torch.equal(got_sequence, want.sequences)
        assert torch.equal(got_generated, want.generated_ids)


def test_model_generate_empty_request_list_is_rejected_cleanly() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    with pytest.raises(ValueError, match="Expected at least one request"):
        _ = model([])
    with pytest.raises(ValueError, match="Expected at least one request"):
        _ = model.generate([], max_new_tokens=2, do_sample=False)
    with pytest.raises(ValueError, match="Expected at least one request"):
        _ = model.collect([], get=[site])


def test_model_generate_max_new_tokens_zero_returns_prompt_only() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())

    actual = model.generate("hello", max_new_tokens=0, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.generated_length == 0
    assert actual.generated_ids.shape == (1, 0)
    assert actual.sequences.shape[-1] == actual.prompt_length


def test_model_generate_varying_prompt_lengths_preserve_row_lengths() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())

    actual = model.generate(["h", "hello world"], max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.prompt_length == [2, 12]
    assert actual.generated_length == [2, 2]


def test_model_call_large_request_batch_preserves_rows() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]
    requests = [f"row-{idx}" for idx in range(32)]

    actual = model(requests, get=[site])

    assert isinstance(actual, list)
    assert len(actual) == 32
    assert all(isinstance(item, ti.Output) for item in actual)


def test_model_generate_forwards_sampling_kwargs_to_wrapped_model() -> None:
    wrapped = _SamplingGenerateModel()
    model = ti.Model(wrapped)

    actual = model.generate(
        _input_ids(),
        max_new_tokens=3,
        do_sample=True,
        temperature=0.7,
        top_k=5,
    )

    assert isinstance(actual, ti.GenerateOutput)
    assert wrapped.last_generate_kwargs == {
        "max_new_tokens": 3,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 5,
    }


def test_grad_flows_through_mapped_modules() -> None:
    torch.manual_seed(0)
    wrapped = _GradMapModel()
    model = ti.Model(wrapped)
    site = model.hidden
    actual_x = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32, requires_grad=True)
    actual = model(actual_x, get=[site], map={site: ti.scale(2.0)}, grad=True)
    actual.logits.sum().backward()

    expected_x = actual_x.detach().clone().requires_grad_(True)
    hidden = wrapped.hidden(expected_x) * 2.0
    expected_logits = wrapped.readout(hidden)
    expected_logits.sum().backward()

    assert torch.allclose(actual_x.grad, expected_x.grad)


def test_model_generate_supports_get_capture_all() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]
    encoded = model.tokenizer("hello", return_tensors="pt")

    actual = model.generate("hello", max_new_tokens=2, do_sample=False, get=[site], capture="all")
    expected = model.generate(**encoded, max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert isinstance(expected, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected.sequences)
    assert torch.equal(actual.generated_ids, expected.generated_ids)
    assert actual.prompt_length == encoded["input_ids"].shape[-1]
    assert actual.generated_length == 2
    assert actual[site].shape[1] == expected.sequences.shape[-1]


def test_model_generate_supports_get_capture_generated_only() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model.generate(
        "hello",
        max_new_tokens=2,
        do_sample=False,
        get=[site],
        capture="generated",
    )

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.generated_length == 2
    assert actual[site].shape[1] == 2


def test_model_generate_without_get_still_returns_generate_output() -> None:
    model = ti.Model(FakeDecoderModel())
    actual = model.generate(input_ids=_input_ids(), max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.prompt_length == _input_ids().shape[-1]
    assert actual.generated_length == 2
    assert actual.activations == {}


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_model_call_request_forms_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model(request_value, get=[site])
    expected = model(**filter_forward_inputs(model.wrapped, expected_inputs), get=[site])

    assert isinstance(actual, ti.Output)
    assert torch.allclose(actual[site], expected[site])
    assert torch.allclose(actual.logits, expected.logits)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_model_call_request_forms_with_map_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model(request_value, get=[site], map={site: ti.zero()})
    expected = model(
        **filter_forward_inputs(model.wrapped, expected_inputs),
        get=[site],
        map={site: ti.zero()},
    )

    assert torch.allclose(actual[site], expected[site])
    assert torch.allclose(actual.logits, expected.logits)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_model_call_request_forms_stop_at_last_get_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model(request_value, get=[site], stop_at_last_get=True)
    expected = model(
        **filter_forward_inputs(model.wrapped, expected_inputs),
        get=[site],
        stop_at_last_get=True,
    )

    assert actual.partial
    assert expected.partial
    assert torch.allclose(actual[site], expected[site])


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
def test_model_generate_request_forms_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())

    actual = model.generate(request_value, max_new_tokens=2, do_sample=False)
    expected = model.generate(**expected_inputs, max_new_tokens=2, do_sample=False)

    assert isinstance(actual, ti.GenerateOutput)
    assert isinstance(expected, ti.GenerateOutput)
    assert torch.equal(actual.sequences, expected.sequences)
    assert torch.equal(actual.generated_ids, expected.generated_ids)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
def test_model_generate_request_forms_with_map_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model.generate(
        request_value,
        max_new_tokens=2,
        do_sample=False,
        map={site: ti.zero()},
    )
    expected = model.generate(
        **expected_inputs,
        max_new_tokens=2,
        do_sample=False,
        map={site: ti.zero()},
    )

    assert torch.equal(actual.sequences, expected.sequences)
    assert torch.equal(actual.generated_ids, expected.generated_ids)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_model_collect_request_forms_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model.collect(request_value, get=[site])
    expected = model.collect(expected_inputs, get=[site])

    assert len(actual) == len(expected) == 1
    assert torch.allclose(actual[0][site], expected[0][site])


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=False),
)
def test_model_collect_request_forms_with_map_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model.collect(
        request_value,
        get=[site],
        map={site: ti.zero()},
        stop_at_last_get=False,
    )
    expected = model.collect(
        expected_inputs,
        get=[site],
        map={site: ti.zero()},
        stop_at_last_get=False,
    )

    assert len(actual) == len(expected) == 1
    assert torch.allclose(actual[0][site], expected[0][site])


def test_model_collect_map_rejects_fast_path_default() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    with pytest.raises(ValueError, match="does not support map="):
        _ = model.collect("hello", get=[site], map={site: ti.zero()})


def test_model_generate_request_list_contracts_preserve_rows() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    requests = [
        "hello",
        {"text": "world"},
        [{"role": "user", "content": "tiny"}],
    ]

    actual = model.generate(requests, max_new_tokens=2, do_sample=False)
    expected = [
        model.generate(request, max_new_tokens=2, do_sample=False) for request in requests
    ]

    assert isinstance(actual, ti.GenerateOutput)
    assert actual.sequences.shape[0] == len(expected) == 3
    for idx, want in enumerate(expected):
        got_sequence, got_generated = generate_row(actual, idx)
        assert torch.equal(got_sequence, want.sequences)
        assert torch.equal(got_generated, want.generated_ids)


@pytest.mark.parametrize(
    ("_case", "request_value", "expected_inputs"),
    request_contract_cases(add_generation_prompt=True),
)
@pytest.mark.parametrize("capture", ["all", "generated"])
def test_model_generate_request_forms_with_capture_match_token_inputs(
    _case: str,
    request_value: Any,
    expected_inputs: dict[str, torch.Tensor],
    capture: str,
) -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    site = model.layers[0]

    actual = model.generate(
        request_value,
        max_new_tokens=2,
        do_sample=False,
        get=[site],
        capture=capture,
    )
    expected = model.generate(
        **expected_inputs,
        max_new_tokens=2,
        do_sample=False,
        get=[site],
        capture=capture,
    )

    assert torch.equal(actual.sequences, expected.sequences)
    assert torch.equal(actual.generated_ids, expected.generated_ids)
    assert torch.allclose(actual[site], expected[site])


def test_model_generate_request_list_with_capture_preserves_rows() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())
    requests = ["hello", {"text": "world"}, [{"role": "user", "content": "tiny"}]]
    site = model.layers[0]

    actual = model.generate(
        requests,
        max_new_tokens=2,
        do_sample=False,
        get=[site],
        capture="all",
    )
    expected = [
        model.generate(
            request,
            max_new_tokens=2,
            do_sample=False,
            get=[site],
            capture="all",
        )
        for request in requests
    ]

    assert isinstance(actual, ti.GenerateOutput)
    for idx, want in enumerate(expected):
        got_sequence, got_generated = generate_row(actual, idx)
        got_activation = generate_activation_row(actual, site, idx, capture="all")
        assert torch.equal(got_sequence, want.sequences)
        assert torch.equal(got_generated, want.generated_ids)
        assert torch.allclose(got_activation, want[site])


def test_local_capabilities_report_grad_support() -> None:
    model = ti.Model(FakeDecoderModel(), tokenizer=FakeTokenizer())

    assert model.capabilities["backend"] == "local"
    assert model.capabilities["grad"] is True
    assert model.capabilities["request_tokenization"] is True


def test_concurrent_local_calls_keep_separate_hook_state() -> None:
    torch.manual_seed(0)
    expected_model = ti.Model(_ConcurrentProbeModel(_ThreadGate(1)))
    expected_site = expected_model.hidden
    x_a = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32)
    x_b = torch.tensor([[0.5, 0.6, -0.7, 0.8]], dtype=torch.float32)
    expected_a = expected_model(x=x_a, get=[expected_site])[expected_site]
    expected_b = expected_model(x=x_b, get=[expected_site])[expected_site]

    torch.manual_seed(0)
    gate = _ThreadGate(2)
    model = ti.Model(_ConcurrentProbeModel(gate))
    site = model.hidden
    outputs: dict[str, torch.Tensor] = {}
    errors: list[Exception] = []

    def worker(name: str, x: torch.Tensor) -> None:
        try:
            outputs[name] = cast(torch.Tensor, model(x=x, get=[site])[site]).detach().clone()
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
        thread.join()

    assert not errors
    assert torch.allclose(outputs["a"], expected_a)
    assert torch.allclose(outputs["b"], expected_b)


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (FakeDecoderModel, "transformer.h.1.attn"),
        (FakeLlamaModel, "model.layers.1.self_attn"),
    ],
)
def test_get_matches_manual_hook(
    factory: Callable[[], nn.Module],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    expected = _direct_site_activation(wrapped, path, _input_ids())

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    output = model(_input_ids(), get=[proxy])

    assert torch.allclose(output[proxy], expected)


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (FakeDecoderModel, "transformer.h.1.attn"),
        (FakeLlamaModel, "model.layers.1.self_attn"),
    ],
)
def test_stop_at_last_get_returns_partial_output_and_matches_manual_hook(
    factory: Callable[[], nn.Module],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    expected = _direct_site_activation(wrapped, path, _input_ids())

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    output = model(_input_ids(), get=[proxy], stop_at_last_get=True)

    assert output.partial
    assert not output.completed_forward
    assert torch.allclose(output[proxy], expected)
    with pytest.raises(RuntimeError, match="stop_at_last_get=True"):
        _ = output.logits


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (FakeDecoderModel, "transformer.h.1.attn"),
        (FakeLlamaModel, "model.layers.1.self_attn"),
    ],
)
def test_map_matches_manual_hook(
    factory: Callable[[], nn.Module],
    path: str,
) -> None:
    torch.manual_seed(0)
    wrapped = factory()
    expected = _run_with_zeroed_module_output(wrapped, path, _input_ids())

    model = ti.Model(wrapped)
    proxy = get_proxy(model, path)
    with torch.no_grad():
        actual = model(_input_ids(), map={proxy: ti.zero()}).logits

    assert torch.allclose(actual, expected)


def test_stop_at_last_get_stops_after_first_unique_site_hit() -> None:
    torch.manual_seed(0)
    wrapped = _SharedLinearModel()
    model = ti.Model(wrapped)
    proxy = model.shared
    x = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32)

    expected = wrapped.shared(x).detach()
    output = model(x, get=[proxy], stop_at_last_get=True)

    assert output.partial
    assert torch.allclose(output[proxy], expected)


def test_threaded_module_execution_is_rejected_without_leaking_hook_state() -> None:
    torch.manual_seed(0)
    model = ti.Model(_ThreadedDispatchModel())
    x = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Requested modules did not capture activations"):
        _ = model(x=x, get=[model.hidden])

    follow_up = model(x=x, get=[model.readout])
    assert torch.is_tensor(follow_up[model.readout])


def test_grad_with_in_place_map_keeps_captured_activation_unmodified() -> None:
    model = ti.Model(FakeLlamaModel())
    site = model.model.layers[0].self_attn
    baseline = model(_input_ids(), get=[site])
    mapped = model(
        _input_ids(),
        get=[site],
        map={site: lambda x: x.add_(1.0)},
        grad=True,
    )

    assert torch.allclose(mapped[site], baseline[site])
    assert not torch.allclose(mapped.logits, baseline.logits)


def test_call_cleans_up_after_exception() -> None:
    wrapped = FakeDecoderModel()
    block0 = cast(Any, wrapped.transformer.h[0])
    block0.fail = True
    model = ti.Model(wrapped)

    with pytest.raises(RuntimeError, match="boom"):
        _ = model(_input_ids(), get=[model.transformer.h[0]])

    block0.fail = False
    output = model(_input_ids(), get=[model.transformer.h[0]])

    assert output[model.transformer.h[0]].shape[1] == _input_ids().shape[1]


def test_stop_at_last_get_skips_later_blocks() -> None:
    wrapped = FakeDecoderModel()
    block1 = cast(Any, wrapped.transformer.h[1])
    block1.fail = True
    model = ti.Model(wrapped)

    with pytest.raises(RuntimeError, match="boom"):
        _ = model(_input_ids(), get=[model.transformer.h[0]])

    output = model(_input_ids(), get=[model.transformer.h[0]], stop_at_last_get=True)

    assert output.partial
    assert output[model.transformer.h[0]].shape[1] == _input_ids().shape[1]


def test_stop_at_last_get_captures_multiple_sites_before_stopping() -> None:
    wrapped = FakeDecoderModel()
    block1 = cast(Any, wrapped.transformer.h[1])
    block1.fail = True
    model = ti.Model(wrapped)
    attn = model.transformer.h[0].attn
    block = model.transformer.h[0]

    output = model(_input_ids(), get=[attn, block], stop_at_last_get=True)

    assert output.partial
    assert output[attn].shape[1] == _input_ids().shape[1]
    assert output[block].shape[1] == _input_ids().shape[1]


def test_stop_at_last_get_rejects_invalid_combinations() -> None:
    model = ti.Model(FakeDecoderModel())
    proxy = model.transformer.h[0]

    with pytest.raises(ValueError, match="requires at least one get="):
        _ = model(_input_ids(), stop_at_last_get=True)

    with pytest.raises(ValueError, match="does not support map="):
        _ = model(_input_ids(), get=[proxy], map={proxy: ti.zero()}, stop_at_last_get=True)

    with pytest.raises(ValueError, match="does not support grad=True"):
        _ = model(_input_ids(), get=[proxy], grad=True, stop_at_last_get=True)


@pytest.mark.parametrize(
    ("factory", "expected_prefix"),
    [
        (FakeDecoderModel, "transformer.h"),
        (FakeLlamaModel, "model.layers"),
    ],
)
def test_layers_finds_biggest_modulelist(
    factory: Callable[[], nn.Module],
    expected_prefix: str,
) -> None:
    model = ti.Model(factory())

    assert len(model.layers) == 2
    assert model.layers[1].path == f"{expected_prefix}.1"


def test_find_and_children_explore_real_tree() -> None:
    model = ti.Model(FakeDecoderModel())

    found = ti.find(model.layers[0], "attn")
    assert found == model.transformer.h[0].attn

    listed = dict(ti.children(model.layers[0]))
    assert listed["attn"] == "FakeDecoderAttention"
    assert listed["mlp"] == "FakeDecoderMlp"


def test_rename_pack_exposes_canonical_aliases() -> None:
    model = ti.Model(FakeDecoderModel(), rename=ti.renames.llm)

    canonical = model.model.layers[0].self_attn
    real = model.transformer.h[0].attn

    assert canonical == real
    assert canonical.path == "transformer.h.0.attn"


def test_counters_debug_and_graph(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ti.Counters.reset()
    model = ti.Model(FakeDecoderModel())
    graph_path = tmp_path / "graph.svg"

    with ti.context(debug=2, graph=graph_path):
        output = model(_input_ids(), get=[model.transformer.h[0].attn])

    assert ti.Counters.calls == 1
    assert ti.Counters.forward_passes == 1
    assert ti.Counters.activations_captured == 1
    assert (
        ti.Counters.activations_bytes
        == output[model.transformer.h[0].attn].numel()
        * output[model.transformer.h[0].attn].element_size()
    )
    assert graph_path.exists()

    stdout = capsys.readouterr().out
    assert "[ti] call:" in stdout
    assert "transformer.h.0.attn" in stdout
    assert "TOTAL:" in stdout


def test_map_head_targets_only_one_slice() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    fn = ti.map_head(1, ti.zero(), n_heads=3)
    out = fn(x)

    assert torch.equal(out[..., :4], x[..., :4])
    assert torch.equal(out[..., 4:8], torch.zeros_like(out[..., 4:8]))
    assert torch.equal(out[..., 8:], x[..., 8:])
