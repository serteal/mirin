"""CUDA-specific end-to-end validation for local and server paths."""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

import pytest
import torch

import mirin as ti

from .helpers import FakeDecoderModel, FakeTokenizer, get_proxy
from .test_server_transformers import _build_llama31

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")


def _to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _wait_for_socket(sock_path: str, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(sock_path)
            except OSError:
                time.sleep(0.05)
            else:
                probe.close()
                return
            finally:
                try:
                    probe.close()
                except OSError:
                    pass
        time.sleep(0.05)
    raise RuntimeError(f"CUDA test server did not open socket {sock_path}.")


def test_cuda_local_server_remote_e2e() -> None:
    _require_cuda()

    wrapped, inputs = _build_llama31()
    wrapped = wrapped.to("cuda").eval()
    cuda_inputs = _to_cuda(inputs)
    local = ti.Model(wrapped)
    site = get_proxy(local, "model.layers.1.self_attn")
    local_output = local(**cuda_inputs, get=[site])
    expected = wrapped.generate(
        **cuda_inputs,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
    )

    assert local_output.logits.device.type == "cuda"
    assert local_output[site].device.type == "cuda"

    server = ti.Server(_build_llama31()[0], device="cuda")
    try:
        plan = server.compile(get=["model.layers.1.self_attn"])
        server_output = server.call(plan, **cuda_inputs)
        assert server_output.logits is not None
        session = server.open_session(plan=plan, cache="dynamic")
        _ = server.prefill(session, **cuda_inputs)
        decoded = server.decode([session], max_new_tokens=2, do_sample=False)[0]
        actual = torch.cat(
            [cuda_inputs["input_ids"], decoded.token_ids.to(cuda_inputs["input_ids"].device)],
            dim=-1,
        )
        assert torch.equal(actual, expected)
    finally:
        server.close()

    sock = "/tmp/mirin_test_cuda_remote.sock"
    remote_server = ti.Server(
        FakeDecoderModel().to("cuda"),
        tokenizer=FakeTokenizer(),
        device="cuda",
    )
    thread = threading.Thread(target=remote_server.serve, args=(sock,), daemon=True)
    thread.start()
    _wait_for_socket(sock)
    try:
        remote = ti.Model(f"unix://{sock}")
        remote_site = remote.layers[0]
        remote_outputs = remote(["hello", "world"], get=[remote_site])
        assert isinstance(remote_outputs, list)
        assert len(remote_outputs) == 2
        assert all(torch.is_tensor(output[remote_site]) for output in remote_outputs)
        remote.close()
    finally:
        remote_server.close()
        if os.path.exists(sock):
            os.unlink(sock)
