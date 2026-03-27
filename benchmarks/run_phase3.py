"""CLI entrypoint for the tinyinterp Phase 3 benchmark harness."""

from __future__ import annotations

import argparse

from .phase3 import (
    DEFAULT_MODEL_NAMES,
    BenchmarkConfig,
    format_suite_report,
    run_phase3_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", action="append")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--micro-warmup", type=int, default=5)
    parser.add_argument("--micro-trials", type=int, default=20)
    parser.add_argument("--throughput-warmup", type=int, default=1)
    parser.add_argument("--throughput-runs", type=int, default=5)
    parser.add_argument("--sweep-width", type=int, default=8)
    parser.add_argument(
        "--get-one-stop-at-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-output")
    args = parser.parse_args()

    model_names = [None] if args.synthetic else (args.model_name or list(DEFAULT_MODEL_NAMES))
    configs = [
        BenchmarkConfig(
            model_name=model_name,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            layers=args.layers,
            width=args.width,
            n_heads=args.n_heads,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            micro_warmup=args.micro_warmup,
            micro_trials=args.micro_trials,
            throughput_warmup=args.throughput_warmup,
            throughput_runs=args.throughput_runs,
            sweep_width=args.sweep_width,
            get_one_stop_at_last=args.get_one_stop_at_last,
        )
        for model_name in model_names
    ]
    suite = run_phase3_suite(configs, json_output=args.json_output)
    print(format_suite_report(suite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
