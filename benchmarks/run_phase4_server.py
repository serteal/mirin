"""CLI entrypoint for the Phase 4 inference-server benchmark harness."""

from __future__ import annotations

import argparse

from .phase4_server import (
    DEFAULT_MODEL_NAMES,
    ServerBenchmarkConfig,
    format_phase4_server_suite,
    run_phase4_server_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", action="append")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dataset-batch-size", type=int, default=4)
    parser.add_argument("--dataset-batches", type=int, default=8)
    parser.add_argument("--generate-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    model_names = [None] if args.synthetic else (args.model_name or list(DEFAULT_MODEL_NAMES))
    configs = [
        ServerBenchmarkConfig(
            model_name=model_name,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            seq_len=args.seq_len,
            dataset_batch_size=args.dataset_batch_size,
            dataset_batches=args.dataset_batches,
            generate_batch_size=args.generate_batch_size,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            trials=args.trials,
            layers=args.layers,
            width=args.width,
            n_heads=args.n_heads,
            vocab_size=args.vocab_size,
        )
        for model_name in model_names
    ]
    suite = run_phase4_server_suite(configs, json_output=args.json_output)
    print(format_phase4_server_suite(suite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
