"""Repo-local collect benchmark and stress harness.

This script is intentionally not part of the public ``mirin`` CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from mirin import Model, renames, resolve_layer_sites


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("toy tokenizer only supports return_tensors='pt'.")
        texts = [text] if isinstance(text, str) else list(text)
        rows: list[list[int]] = []
        for item in texts:
            encoded = [1]
            encoded.extend(((ord(char) % 23) + 3) for char in item)
            rows.append(encoded[:64])
        max_len = max((len(row) for row in rows), default=1)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for idx, row in enumerate(rows):
            values = torch.tensor(row, dtype=torch.long)
            input_ids[idx, : len(row)] = values
            attention_mask[idx, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _ToyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(hidden_states)


class _ToyBackbone(nn.Module):
    layers: nn.ModuleList

    def __init__(self, width: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(width) for _ in range(n_layers)])


class _ToyLlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        width: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.model = _ToyBackbone(width, n_layers)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            hidden_size=width,
            intermediate_size=width * 2,
            eos_token_id=2,
            pad_token_id=0,
            _attn_implementation="eager",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        del attention_mask
        hidden = self.embed(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        use_cache: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        del attention_mask, do_sample, use_cache
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(tokens).logits
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokens


def _parse_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_name!r}.") from exc


def _load_bundle(
    model_name: str,
    *,
    rename_name: str,
    dtype_name: str,
) -> tuple[nn.Module, Any | None, dict[str, str] | None]:
    rename = renames.llm if rename_name == "llm" else None
    if model_name == "toy-llama":
        return _ToyLlamaModel(), _ToyTokenizer(), rename
    from mirin.model import _load_model, _maybe_load_tokenizer

    load_kwargs: dict[str, Any] = {}
    torch_dtype = _parse_dtype(dtype_name)
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    wrapped = _load_model(model_name, **load_kwargs)
    tokenizer = _maybe_load_tokenizer(model_name)
    return wrapped, tokenizer, rename


def _synthetic_requests(
    *,
    count: int,
    min_len: int,
    max_len: int,
    vocab_size: int,
) -> list[dict[str, torch.Tensor]]:
    requests: list[dict[str, torch.Tensor]] = []
    width = max(max_len - min_len, 1)
    for idx in range(count):
        length = min_len + ((idx * 7) % width)
        tokens = torch.arange(idx, idx + length, dtype=torch.long).unsqueeze(0)
        tokens = tokens % max(vocab_size - 1, 1)
        requests.append(
            {
                "input_ids": tokens + 1,
                "attention_mask": torch.ones_like(tokens, dtype=torch.long),
            }
        )
    return requests


def _infer_vocab_size(model: Model) -> int:
    config = getattr(model.wrapped, "config", None)
    value = getattr(config, "vocab_size", None)
    if isinstance(value, int) and value > 1:
        return value
    tokenizer = getattr(model, "tokenizer", None)
    token_value = getattr(tokenizer, "vocab_size", None)
    if isinstance(token_value, int) and token_value > 1:
        return token_value
    embed = getattr(model.wrapped, "embed", None)
    num_embeddings = getattr(embed, "num_embeddings", None)
    if isinstance(num_embeddings, int) and num_embeddings > 1:
        return num_embeddings
    return 32


def _parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _cast_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor activation, got {type(value).__name__}.")
    return value


def _run_collect_pass(
    model: Model,
    requests: list[dict[str, torch.Tensor]],
    *,
    layers: list[int],
    hook_point: str,
    batch_size: int,
    batch_token_budget: int | None,
    sort_by_length: bool,
) -> dict[str, Any]:
    sites = resolve_layer_sites(model, layers, hook_point=hook_point)
    paths = [site.path for site in sites]
    pooled: list[list[torch.Tensor] | None] = [None] * len(requests)
    total_tokens = 0
    max_batch_rows = 0
    started = time.perf_counter()
    for step in model.collect(
        requests,
        get=sites,
        process=lambda step: step,
        max_items=batch_size,
        max_tokens=batch_token_budget,
        sort=sort_by_length,
        stop_at_last_get=True,
    ):
        max_batch_rows = max(max_batch_rows, len(step.indices))
        for local_idx, (row, idx) in enumerate(zip(step.rows, step.indices, strict=True)):
            length = int(row["attention_mask"].sum().item())
            total_tokens += length
            pooled[idx] = [
                _cast_tensor(step[path])[local_idx, :length].mean(dim=0).detach().cpu()
                for path in paths
            ]
        step.release()
    elapsed_s = max(time.perf_counter() - started, 1e-9)
    return {
        "elapsed_s": elapsed_s,
        "requests_per_second": len(requests) / elapsed_s,
        "tokens_per_second": total_tokens / elapsed_s,
        "total_tokens": total_tokens,
        "max_batch_rows": max_batch_rows,
        "pooled": [
            [value.tolist() for value in values] if values is not None else None
            for values in pooled
        ],
    }


def _emit_report(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_compact_report(report), indent=2, sort_keys=True))
        return
    print(f"model: {report['model']}")
    print(f"requests: {report['requests']}")
    print(f"local elapsed_s: {report['local']['elapsed_s']:.4f}")
    print(f"local tokens_per_second: {report['local']['tokens_per_second']:.2f}")


def _compact_collect_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "pooled"}


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    compact = dict(report)
    if isinstance(compact.get("local"), dict):
        compact["local"] = _compact_collect_summary(compact["local"])
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect-path benchmark and stress harness")
    parser.add_argument("--model", default="toy-llama")
    parser.add_argument("--rename", choices=["none", "llm"], default="llm")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--hook-point", choices=["block", "layernorm"], default="block")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-token-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sort-by-length", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    wrapped, tokenizer, rename = _load_bundle(
        args.model,
        rename_name=args.rename,
        dtype_name=args.dtype,
    )
    local_model = Model(wrapped, rename=rename, tokenizer=tokenizer)
    if args.device is not None:
        local_model.wrapped.to(args.device)
    try:
        requests = _synthetic_requests(
            count=args.requests,
            min_len=args.min_len,
            max_len=args.max_len,
            vocab_size=_infer_vocab_size(local_model),
        )
        layers = _parse_layers(args.layers)
        local_report = _run_collect_pass(
            local_model,
            requests,
            layers=layers,
            hook_point=args.hook_point,
            batch_size=args.batch_size,
            batch_token_budget=args.batch_token_budget,
            sort_by_length=not args.no_sort_by_length,
        )
        report: dict[str, Any] = {
            "model": args.model,
            "requests": args.requests,
            "layers": layers,
            "local": local_report,
        }
    finally:
        local_model.close()

    _emit_report(report, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
