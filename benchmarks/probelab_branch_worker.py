"""Profile one probelab tree against a real HF model.

This worker is meant to be launched in a subprocess with the target probelab
environment active, for example:

    uv run --project /path/to/probelab python benchmarks/probelab_branch_worker.py \
        --probelab-root /path/to/probelab ...
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _git_value(root: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", os.fspath(root), *args],
        text=True,
    ).strip()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}.") from exc


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gpu_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return None


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "std_s": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_s": min(values),
        "max_s": max(values),
    }


def _build_dataset(pl: Any, *, n_samples: int, min_len: int, max_len: int) -> Any:
    width = max(max_len - min_len, 1)
    stems = (
        "Activation patching reveals whether a hidden state causally matters.",
        "Pooling over assistant tokens is a common probing workload.",
        "Dynamic batching should maximize utilization without hitting OOM.",
        "Interpretability pipelines often stream activations into downstream trainers.",
    )
    dialogues = []
    labels = []
    for idx in range(n_samples):
        repeat = min_len + ((idx * 11) % width)
        stem = stems[idx % len(stems)]
        user = f"User {idx}: " + " ".join([stem] * max(1, repeat // 12))
        assistant = f"Assistant {idx}: " + " ".join(reversed(stem.split())) * max(
            1, repeat // 20
        )
        dialogues.append(
            [
                pl.types.Message(role=pl.types.Role.USER, content=user),
                pl.types.Message(role=pl.types.Role.ASSISTANT, content=assistant),
            ]
        )
        labels.append(pl.types.Label(idx % 2))
    return pl.datasets.base.Dataset(
        dialogues=dialogues,
        labels=labels,
        name="synthetic_profile",
    )


def _summarize_tokens(tokens: Any) -> dict[str, Any]:
    lengths = tokens.lengths.detach().cpu().to(torch.int64)
    quantiles = torch.quantile(lengths.float(), torch.tensor([0.5, 0.9, 0.99])).tolist()
    return {
        "samples": int(len(tokens)),
        "total_tokens": int(tokens.total_tokens),
        "max_seq": int(tokens.seq_len),
        "min_len": int(lengths.min().item()) if lengths.numel() else 0,
        "max_len": int(lengths.max().item()) if lengths.numel() else 0,
        "mean_len": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
        "p50_len": float(quantiles[0]) if quantiles else 0.0,
        "p90_len": float(quantiles[1]) if len(quantiles) > 1 else 0.0,
        "p99_len": float(quantiles[2]) if len(quantiles) > 2 else 0.0,
    }


def _load_model(
    *,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=token,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            token=token,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def _measure_case(
    *,
    name: str,
    device: torch.device,
    warmup: int,
    trials: int,
    fn: Any,
) -> tuple[dict[str, Any], Any]:
    last_value: Any = None
    for _ in range(warmup):
        last_value = fn()
        _sync(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    times: list[float] = []
    peak_gpu_mb = 0.0
    peak_rss_mb = 0.0
    for _ in range(trials):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _reset_peak(device)
        rss_before = _rss_mb() or 0.0
        _sync(device)
        started = time.perf_counter()
        last_value = fn()
        _sync(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        peak_gpu_mb = max(peak_gpu_mb, _peak_gpu_mb(device) or 0.0)
        peak_rss_mb = max(peak_rss_mb, (_rss_mb() or rss_before) - rss_before)
    report = {
        "name": name,
        **_stats(times),
        "peak_gpu_mb": peak_gpu_mb if device.type == "cuda" else None,
        "peak_rss_delta_mb": peak_rss_mb,
    }
    return report, last_value


def _save_tensor(value: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu(), path)
    return os.fspath(path)


def _run_stream_mean(
    pl: Any,
    *,
    model: Any,
    tokens: Any,
    layer: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    pool_fn = pl.pool.mean
    n = len(tokens)
    out: torch.Tensor | None = None
    extract_times: list[float] = []
    pool_times: list[float] = []
    batch_tokens: list[int] = []
    batch_samples: list[int] = []

    iterator = iter(pl.processing.stream_activations(model, tokens, layers=[layer], batch_size=batch_size))
    while True:
        extract_started = time.perf_counter()
        try:
            flat_data, det, offsets, idx = next(iterator)
        except StopIteration:
            break
        extract_times.append(time.perf_counter() - extract_started)
        batch_tokens.append(int(offsets[-1].item()) if offsets.numel() else 0)
        batch_samples.append(int(offsets.shape[0] - 1))
        if out is None:
            out = torch.zeros(
                n,
                1,
                flat_data.shape[-1],
                dtype=flat_data.dtype,
                device=flat_data.device,
            )
        pool_started = time.perf_counter()
        pooled = pool_fn(flat_data[:, 0, :], det, offsets=offsets)
        out_idx = torch.tensor(idx, dtype=torch.long, device=pooled.device)
        out[out_idx, 0] = pooled
        pool_times.append(time.perf_counter() - pool_started)
    if out is None:
        out = torch.zeros(n, 0, dtype=torch.float32)
    else:
        out = out.squeeze(1).cpu()
    prepared = pl.Activations(data=out, dims="bh", layers=None)
    metrics = {
        "num_batches": len(extract_times),
        "extract_total_s": sum(extract_times),
        "pool_total_s": sum(pool_times),
        "extract_mean_s": statistics.mean(extract_times) if extract_times else 0.0,
        "pool_mean_s": statistics.mean(pool_times) if pool_times else 0.0,
        "mean_batch_tokens": statistics.mean(batch_tokens) if batch_tokens else 0.0,
        "max_batch_tokens": max(batch_tokens) if batch_tokens else 0,
        "mean_batch_samples": statistics.mean(batch_samples) if batch_samples else 0.0,
    }
    return prepared, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile one probelab tree on a real HF model")
    parser.add_argument("--probelab-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)

    probelab_root = Path(args.probelab_root).resolve()
    sys.path.insert(0, os.fspath(probelab_root))
    import probelab as pl  # type: ignore

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    model, tokenizer = _load_model(model_name=args.model, device=device, dtype=dtype)
    dataset = _build_dataset(pl, n_samples=args.samples, min_len=args.min_len, max_len=args.max_len)
    tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
    model_layer_count = int(getattr(model.config, "num_hidden_layers"))
    layer = (model_layer_count // 2) if args.layer < 0 else min(args.layer, model_layer_count - 1)
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        collect_flat_report, collect_flat = _measure_case(
            name="collect_flat",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: pl.collect_activations(
                model,
                tokens,
                layers=[layer],
                batch_size=args.batch_size,
            ),
        )
        collect_mean_report, collect_mean = _measure_case(
            name="collect_mean",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: pl.collect_activations(
                model,
                tokens,
                layers=[layer],
                batch_size=args.batch_size,
                pool="mean",
            ),
        )
        stream_mean_report, stream_mean_bundle = _measure_case(
            name="stream_mean",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: _run_stream_mean(
                pl,
                model=model,
                tokens=tokens,
                layer=layer,
                batch_size=args.batch_size,
            ),
        )
        stream_mean, stream_metrics = stream_mean_bundle

        collect_flat_report.update(
            {
                "dims": collect_flat.dims,
                "shape": list(collect_flat.data.shape),
                "total_rows": int(collect_flat.data.shape[0]),
                "samples_per_second": args.samples / collect_flat_report["median_s"],
                "tokens_per_second": int(tokens.total_tokens) / collect_flat_report["median_s"],
            }
        )
        collect_mean_report.update(
            {
                "dims": collect_mean.dims,
                "shape": list(collect_mean.data.shape),
                "samples_per_second": args.samples / collect_mean_report["median_s"],
                "tokens_per_second": int(tokens.total_tokens) / collect_mean_report["median_s"],
                "tensor_path": _save_tensor(collect_mean.data, result_dir / "collect_mean.pt"),
            }
        )
        stream_mean_report.update(
            {
                "dims": stream_mean.dims,
                "shape": list(stream_mean.data.shape),
                "samples_per_second": args.samples / stream_mean_report["median_s"],
                "tokens_per_second": int(tokens.total_tokens) / stream_mean_report["median_s"],
                "tensor_path": _save_tensor(stream_mean.data, result_dir / "stream_mean.pt"),
                "breakdown": stream_metrics,
            }
        )
        max_abs = float(
            (collect_mean.data.float().cpu() - stream_mean.data.float().cpu()).abs().max().item()
        )
        report = {
            "probelab_root": os.fspath(probelab_root),
            "git": {
                "branch": _git_value(probelab_root, ["branch", "--show-current"]),
                "commit": _git_value(probelab_root, ["rev-parse", "HEAD"]),
            },
            "model": args.model,
            "device": str(device),
            "dtype": args.dtype,
            "layer": layer,
            "tokens": _summarize_tokens(tokens),
            "cases": {
                "collect_flat": collect_flat_report,
                "collect_mean": collect_mean_report,
                "stream_mean": stream_mean_report,
            },
            "correctness": {
                "collect_mean_vs_stream_mean_max_abs_diff": max_abs,
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
