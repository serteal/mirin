"""CLI entrypoint for the real-model benchmark matrix."""

from __future__ import annotations

import argparse

from .matrix import MatrixConfig, format_matrix_report, run_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--local-seq-len", type=int, default=64)
    parser.add_argument("--local-warmup", type=int, default=0)
    parser.add_argument("--local-trials", type=int, default=1)
    parser.add_argument("--remote-batch-size", type=int, default=1)
    parser.add_argument("--remote-seq-len", type=int, default=64)
    parser.add_argument("--remote-max-new-tokens", type=int, default=4)
    parser.add_argument("--remote-warmup", type=int, default=0)
    parser.add_argument("--remote-trials", type=int, default=1)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = run_matrix(
        MatrixConfig(
            model_names=args.model_name,
            device=args.device,
            dtype=args.dtype,
            local_batch_size=args.local_batch_size,
            local_seq_len=args.local_seq_len,
            local_warmup=args.local_warmup,
            local_trials=args.local_trials,
            remote_batch_size=args.remote_batch_size,
            remote_seq_len=args.remote_seq_len,
            remote_max_new_tokens=args.remote_max_new_tokens,
            remote_warmup=args.remote_warmup,
            remote_trials=args.remote_trials,
            json_output=args.json_output,
        )
    )
    print(format_matrix_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
