"""Remote ti.Model(unix://...) integration tests."""

from __future__ import annotations

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
    request_contract_cases,
)
from .server_helpers import (
    GradProbeModel,
    SamplingGenerateModel,
    _ids,
    _input_ids,
    _seeded_model,
    _start_server,
    _stop_server,
    _wait_until,
)


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

    sock = "/tmp/mirin_test_call.sock"
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
    sock = "/tmp/mirin_test_caps.sock"
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

    sock = "/tmp/mirin_test_map.sock"
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

    sock = "/tmp/mirin_test_many.sock"
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

    sock = "/tmp/mirin_test_collect.sock"
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

    sock = "/tmp/mirin_test_gen.sock"
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

    sock = "/tmp/mirin_test_gen_many.sock"
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
    sock = str(tmp_path / "mirin_test_empty_requests.sock")
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
    sock = str(tmp_path / "mirin_test_zero_generate.sock")
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
    sock = str(tmp_path / "mirin_test_sampling_generate.sock")
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

    sock = "/tmp/mirin_test_gen_get.sock"
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
    sock = "/tmp/mirin_test_gen_get_generated.sock"
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

    sock = "/tmp/mirin_test_remote_call_contract.sock"
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

    sock = "/tmp/mirin_test_remote_call_map_contract.sock"
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

    sock = "/tmp/mirin_test_remote_call_stop_contract.sock"
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

    sock = "/tmp/mirin_test_remote_generate_contract.sock"
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

    sock = "/tmp/mirin_test_remote_generate_map_contract.sock"
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

    sock = "/tmp/mirin_test_remote_collect_contract.sock"
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

    sock = "/tmp/mirin_test_remote_collect_map_contract.sock"
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
    sock = "/tmp/mirin_test_remote_collect_map_error.sock"
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

    sock = "/tmp/mirin_test_remote_generate_capture_contract.sock"
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

    sock = "/tmp/mirin_test_remote_generate_list_contract.sock"
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

    sock = "/tmp/mirin_test_remote_generate_list_capture_contract.sock"
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

    sock = "/tmp/mirin_test_nav.sock"
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

    sock = "/tmp/mirin_test_stop.sock"
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
    sock = "/tmp/mirin_test_lazy.sock"
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
    sock = "/tmp/mirin_test_release.sock"
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

    def test_grad_matches_local_for_tensor_inputs(self) -> None:
        sock = "/tmp/mirin_test_grad.sock"
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
        sock = "/tmp/mirin_test_grad_text.sock"
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
        sock = "/tmp/mirin_test_grad_release.sock"
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
        sock = "/tmp/mirin_test_grad_disconnect.sock"
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

        sock = "/tmp/mirin_test_validate.sock"
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

        sock = "/tmp/mirin_test_single.sock"
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

        sock = "/tmp/mirin_test_prompts.sock"
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
