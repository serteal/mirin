"""Run local OOM/admission guardrail checks for `mirin.Model.collect(...)`."""

from __future__ import annotations

import argparse
import json
from typing import Any

from mirin import Model, renames, resolve_layer_sites

from testbed_collect import _ToyLlamaModel, _parse_layers, _synthetic_requests


def _run_cases(
    *,
    device: str | None,
    layers: list[int],
    batch_size: int,
    min_len: int,
    max_len: int,
    batch_token_budget: int | None,
) -> dict[str, Any]:
    wrapped = _ToyLlamaModel()
    model = Model(wrapped, rename=renames.llm)
    if device is not None:
        model.wrapped.to(device)
    try:
        runtime = model._executor._shared_runtime()
        sites = resolve_layer_sites(model, layers)
        rows = _synthetic_requests(
            count=batch_size,
            min_len=min_len,
            max_len=max_len,
            vocab_size=32,
        )
        cases = {
            "activation_budget": False,
            "cpu_budget": False,
            "kv_cache_budget": False,
        }
        try:
            _ = model.collect(
                rows,
                get=sites,
                max_tokens=batch_token_budget,
                activation_budget_bytes=1,
            )
        except MemoryError:
            cases["activation_budget"] = True
        original_cpu_capacity = runtime.capacity.cpu_capacity_bytes
        try:
            runtime.capacity.cpu_capacity_bytes = 1
            _ = model.collect(
                rows,
                get=sites,
                out="cpu",
                max_tokens=batch_token_budget,
            )
        except MemoryError:
            cases["cpu_budget"] = True
        finally:
            runtime.capacity.cpu_capacity_bytes = original_cpu_capacity
        original_kv_scheduler = runtime._scheduler.max_kv_cache_bytes
        original_kv_capacity = runtime.capacity.kv_cache_bytes
        try:
            runtime._scheduler.max_kv_cache_bytes = 1
            runtime.capacity.kv_cache_bytes = 1
            _ = model.collect(
                rows,
                get=sites,
                max_tokens=batch_token_budget,
            )
        except MemoryError:
            cases["kv_cache_budget"] = True
        finally:
            runtime._scheduler.max_kv_cache_bytes = original_kv_scheduler
            runtime.capacity.kv_cache_bytes = original_kv_capacity
        return {
            "cases": cases,
            "passed": all(cases.values()),
        }
    finally:
        model.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mirin local OOM/admission guardrail harness")
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-token-budget", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {
        "layers": _parse_layers(args.layers),
        "workload": _run_cases(
            device=args.device,
            layers=_parse_layers(args.layers),
            batch_size=args.batch_size,
            min_len=args.min_len,
            max_len=args.max_len,
            batch_token_budget=args.batch_token_budget,
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"guardrails passed: {report['workload']['passed']}")
        print(report["workload"])
    return 0 if bool(report["workload"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
