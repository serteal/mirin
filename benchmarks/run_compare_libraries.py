"""CLI entrypoint for the cross-library benchmark harness."""

from __future__ import annotations

import argparse

from .compare_libraries import (
    CompareConfig,
    format_compare_report,
    run_compare_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-family", default="custom")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--hf-block-path", default="model.layers.0")
    parser.add_argument("--lens-hook-name", default="blocks.0.hook_resid_post")
    parser.add_argument("--lens-stop-at-layer", type=int, default=1)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = run_compare_benchmarks(
        CompareConfig(
            model_name=args.model,
            model_family=args.model_family,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            trials=args.trials,
            hf_block_path=args.hf_block_path,
            lens_hook_name=args.lens_hook_name,
            lens_stop_at_layer=args.lens_stop_at_layer,
            json_output=args.json_output,
        )
    )
    print(format_compare_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
