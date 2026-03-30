"""CLI entrypoint for the local/remote comparison benchmark harness."""

from __future__ import annotations

import argparse

from .remote_compare import (
    RemoteCompareConfig,
    format_remote_compare_report,
    run_remote_compare_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-family", default="custom")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = run_remote_compare_benchmarks(
        RemoteCompareConfig(
            model_name=args.model_name,
            model_family=args.model_family,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            trials=args.trials,
            json_output=args.json_output,
        )
    )
    print(format_remote_compare_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
