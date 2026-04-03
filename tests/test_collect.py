"""Tests for dataset-style mirin collection helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

import mirin as ti


class _TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(self.input_layernorm(x))


class _TinyModel(nn.Module):
    def __init__(self, width: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(width) for _ in range(n_layers)])
        self.model.embed = nn.Embedding(32, width)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        hidden = self.model.embed(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)
        return hidden


def _token_request(length: int, start: int) -> dict[str, torch.Tensor]:
    tokens = torch.arange(start, start + length, dtype=torch.long).unsqueeze(0) % 31
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
    }


def test_resolve_layer_sites_supports_block_and_layernorm() -> None:
    model = ti.Model(_TinyModel())
    block_sites = ti.resolve_layer_sites(model, [0, 2], hook_point="block")
    layernorm_sites = ti.resolve_layer_sites(model, [1], hook_point="layernorm")

    assert [site.path for site in block_sites] == ["model.layers.0", "model.layers.2"]
    assert layernorm_sites[0].path == "model.layers.1.input_layernorm"

    model.close()


def test_stream_collect_respects_batch_size_and_token_budget() -> None:
    model = ti.Model(_TinyModel())
    requests = [
        _token_request(6, 0),
        _token_request(4, 10),
        _token_request(3, 20),
    ]
    site = ti.resolve_layer_sites(model, [1])[0]

    batches = list(
        ti.stream_collect(
            model,
            requests,
            get=[site],
            batch_size=3,
            batch_token_budget=8,
            sort_by_length=True,
        )
    )

    assert [batch.indices for batch in batches] == [[0], [1, 2]]
    assert [len(batch.indices) for batch in batches] == [1, 2]
    assert [int(batch.rows[0]["attention_mask"].sum().item()) for batch in batches] == [6, 4]
    assert [int(batch.batch["attention_mask"].sum().item()) for batch in batches] == [6, 7]

    for batch in batches:
        activation = batch[site]
        assert isinstance(activation, torch.Tensor)
        assert activation.shape[0] == len(batch.indices)
        batch.release()
    model.close()
