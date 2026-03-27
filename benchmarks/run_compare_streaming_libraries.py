"""CLI entrypoint for the large-workload dataset-loop comparison."""

from __future__ import annotations

import argparse

from .compare_streaming_libraries import (
    StreamingCompareConfig,
    format_streaming_compare_report,
    run_streaming_compare_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--source-batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dataset-batches", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--hf-block-path", default="model.layers.0")
    parser.add_argument("--lens-hook-name", default="blocks.0.hook_resid_post")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = run_streaming_compare_benchmarks(
        StreamingCompareConfig(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            chunk_batch_size=args.chunk_batch_size,
            source_batch_size=args.source_batch_size,
            seq_len=args.seq_len,
            dataset_batches=args.dataset_batches,
            warmup=args.warmup,
            trials=args.trials,
            hf_block_path=args.hf_block_path,
            lens_hook_name=args.lens_hook_name,
            json_output=args.json_output,
        )
    )
    print(format_streaming_compare_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
