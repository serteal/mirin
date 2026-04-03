"""CUDA-specific end-to-end validation for local model paths."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import mirin as ti

from .helpers import FakeLlamaModel, get_proxy

pytestmark = pytest.mark.cuda


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")


def _to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def test_cuda_local_e2e() -> None:
    _require_cuda()

    wrapped = FakeLlamaModel().to("cuda").eval()
    cuda_inputs = _to_cuda({"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)})
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
    generated = local.generate(**cuda_inputs, max_new_tokens=2, do_sample=False)
    assert torch.equal(generated.sequences, expected)
    local.close()
