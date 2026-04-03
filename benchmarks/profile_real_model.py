"""End-to-end real-model profiling for mirin and the current probelab tree."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import torch

import mirin as ti
from mirin.collect import CollectStep

if __package__ in {None, ""}:
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from benchmarks.model_api import (
    _environment_report,
    _get_module,
    _load_hf_workload_model,
    _resolve_device,
    _resolve_dtype,
)


_PROMPT_STEMS = (
    "Explain why activation patching can reveal causal structure in transformer circuits.",
    "Describe the tradeoff between simplicity and speed in systems engineering.",
    "Summarize how a residual stream differs from an attention output.",
    "Give a short explanation of why benchmark methodology needs warmup and repeated trials.",
    "Explain the purpose of an MLP block inside a transformer layer.",
    "Describe one reason fixed prompts are useful when comparing benchmark runs.",
    "Explain how a forward hook can observe or change a module output.",
    "Summarize why exact environment reporting matters for performance claims.",
)


def _git_value(root: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", os.fspath(root), *args],
        text=True,
    ).strip()


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


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max().item())


def _variable_prompts(total: int, *, min_len: int, max_len: int) -> list[str]:
    width = max(max_len - min_len, 1)
    prompts: list[str] = []
    for idx in range(total):
        repeat = min_len + ((idx * 13) % width)
        stem = _PROMPT_STEMS[idx % len(_PROMPT_STEMS)]
        prompt = f"Prompt {idx}: " + " ".join([stem] * max(1, repeat // 10))
        prompts.append(prompt)
    return prompts


def _tokenize_groups(
    tokenizer: Any,
    prompts: list[str],
    *,
    group_size: int,
) -> list[dict[str, torch.Tensor]]:
    groups: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(prompts), group_size):
        chunk = prompts[start : start + group_size]
        batch = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
        )
        groups.append(
            {
                key: value
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
        )
    return groups


def _manual_capture(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _get_module(model, path)
    captured: dict[str, torch.Tensor] = {}

    def hook(_: Any, _args: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor activation at {path!r}.")
        captured["value"] = value.detach()

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(**inputs)
    finally:
        handle.remove()
    logits = cast(torch.Tensor, getattr(output, "logits"))
    return logits.detach(), captured["value"]


def _trim_batch_activation(value: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for idx in range(int(value.shape[0])):
        length = int(attention_mask[idx].sum().item())
        rows.append(value[idx : idx + 1, :length].detach().cpu())
    return rows


def _generate_rows(output: Any) -> list[torch.Tensor]:
    sequences = cast(torch.Tensor, output.sequences).detach().cpu()
    prompt_length = output.prompt_length
    generated_length = output.generated_length
    prompt_lengths = (
        [int(value) for value in prompt_length]
        if isinstance(prompt_length, list)
        else [int(prompt_length)] * int(sequences.shape[0])
    )
    generated_lengths = (
        [int(value) for value in generated_length]
        if isinstance(generated_length, list)
        else [int(generated_length)] * int(sequences.shape[0])
    )
    rows: list[torch.Tensor] = []
    for idx in range(int(sequences.shape[0])):
        total = prompt_lengths[idx] + generated_lengths[idx]
        rows.append(sequences[idx : idx + 1, :total])
    return rows


def _stack_output_activations(outputs: list[Any], site: Any) -> torch.Tensor:
    return torch.cat([cast(torch.Tensor, output[site]).detach() for output in outputs], dim=0)


def _step_activation_tensor(step: CollectStep, site: Any) -> torch.Tensor:
    return cast(torch.Tensor, step[site]).detach()


def _capture_tensor(value: Any, site: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return cast(torch.Tensor, value[site]).detach()


def _release_outputs(outputs: list[Any]) -> None:
    for output in outputs:
        releaser = getattr(output, "release", None)
        if callable(releaser):
            releaser()


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
    peak_rss_delta_mb = 0.0
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
        peak_rss_delta_mb = max(peak_rss_delta_mb, (_rss_mb() or rss_before) - rss_before)
    report = {
        "name": name,
        **_stats(times),
        "peak_gpu_mb": peak_gpu_mb if device.type == "cuda" else None,
        "peak_rss_delta_mb": peak_rss_delta_mb,
    }
    return report, last_value


def _consume_collect_iterator(iterator: Any) -> dict[str, Any]:
    rows = 0
    total_tokens = 0
    batches = 0
    for step in iterator:
        if isinstance(step, CollectStep):
            batches += 1
            rows += len(step.indices)
            attention_mask = step.batch.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                total_tokens += int(attention_mask.sum().item())
            step.release()
        else:
            batches += 1
            if isinstance(step, torch.Tensor):
                rows += int(step.shape[0])
    return {"rows": rows, "total_tokens": total_tokens, "batches": batches}


def _profile_case(fn: Any, *, device: torch.device) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        fn()
        _sync(device)
    events = prof.key_averages()
    by_cuda = sorted(
        events,
        key=lambda evt: float(getattr(evt, "self_cuda_time_total", 0.0)),
        reverse=True,
    )
    by_cpu = sorted(
        events,
        key=lambda evt: float(getattr(evt, "self_cpu_time_total", 0.0)),
        reverse=True,
    )

    def pack(values: list[Any]) -> list[dict[str, Any]]:
        packed: list[dict[str, Any]] = []
        for evt in values[:12]:
            packed.append(
                {
                    "name": evt.key,
                    "count": int(evt.count),
                    "self_cpu_ms": float(getattr(evt, "self_cpu_time_total", 0.0)) / 1000.0,
                    "self_cuda_ms": float(getattr(evt, "self_cuda_time_total", 0.0)) / 1000.0,
                    "cpu_mem_mb": float(getattr(evt, "self_cpu_memory_usage", 0.0)) / (1024 * 1024),
                    "cuda_mem_mb": float(getattr(evt, "self_cuda_memory_usage", 0.0)) / (1024 * 1024),
                }
            )
        return packed

    return {
        "top_cpu": pack(by_cpu),
        "top_cuda": pack(by_cuda),
    }


def _gather_processed_tensors(iterator: Any) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for value in iterator:
        if isinstance(value, torch.Tensor):
            rows.append(value.detach().cpu())
    if not rows:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(rows, dim=0)


def _run_probelab_worker(
    *,
    env_project_root: Path,
    probelab_root: Path,
    result_dir: Path,
    model_name: str,
    device: str,
    dtype: str,
    samples: int,
    min_len: int,
    max_len: int,
    batch_size: int,
    layer: int,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "--project",
        os.fspath(env_project_root),
        "python",
        os.fspath(Path(__file__).with_name("probelab_branch_worker.py")),
        "--probelab-root",
        os.fspath(probelab_root),
        "--model",
        model_name,
        "--device",
        device,
        "--dtype",
        dtype,
        "--samples",
        str(samples),
        "--min-len",
        str(min_len),
        "--max-len",
        str(max_len),
        "--batch-size",
        str(batch_size),
        "--layer",
        str(layer),
        "--warmup",
        str(warmup),
        "--trials",
        str(trials),
        "--result-dir",
        os.fspath(result_dir),
    ]
    output = subprocess.check_output(cmd, text=True)
    return cast(dict[str, Any], json.loads(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile mirin.Model and the current probelab integration")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--probelab-current", default="/root/probes/probelab")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    raw_model, tokenizer = _load_hf_workload_model(args.model, device=device, dtype=dtype)
    model = ti.Model(raw_model, tokenizer=tokenizer, rename=ti.renames.llm)
    prompts = _variable_prompts(args.samples, min_len=args.min_len, max_len=args.max_len)
    prompt_batches = _tokenize_groups(tokenizer, prompts, group_size=args.batch_size)
    first_batch_cpu = prompt_batches[0]
    first_batch = {
        key: value.to(device)
        for key, value in first_batch_cpu.items()
    }
    layer_count = int(getattr(raw_model.config, "num_hidden_layers"))
    layer = (layer_count // 2) if args.layer < 0 else min(args.layer, layer_count - 1)
    site = model.layers[layer]

    raw_logits, raw_activation = _manual_capture(raw_model, first_batch, path=site.path)
    raw_activation_rows = _trim_batch_activation(raw_activation, cast(torch.Tensor, first_batch["attention_mask"]))

    def manual_mean_from_outputs(outputs: list[Any], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values = _stack_output_activations(outputs, site)
        attention_mask = cast(torch.Tensor, batch["attention_mask"]).detach()
        pooled_rows: list[torch.Tensor] = []
        for idx in range(int(values.shape[0])):
            length = int(attention_mask[idx].sum().item())
            pooled_rows.append(values[idx, :length].mean(dim=0))
        return torch.stack(pooled_rows, dim=0)

    try:
        forward_plain_report, forward_plain = _measure_case(
            name="forward_plain",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: model(**first_batch),
        )
        forward_capture_report, forward_capture = _measure_case(
            name="forward_capture_only",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: model(**first_batch, get=[site], stop_at_last_get=True),
        )
        generate_report, generate_output = _measure_case(
            name="generate_requests",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=lambda: model.generate(
                prompts[: args.batch_size],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            ),
        )

        def _collect_requests_cpu() -> list[Any]:
            outputs = cast(
                list[Any],
                model.collect(
                    prompts,
                    get=[site],
                    out="cpu",
                    max_items=args.batch_size,
                    max_tokens=args.max_tokens,
                ),
            )
            return outputs

        collect_requests_cpu_report, collect_requests_cpu_outputs = _measure_case(
            name="collect_requests_cpu",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=_collect_requests_cpu,
        )
        _release_outputs(cast(list[Any], collect_requests_cpu_outputs))

        def _collect_batch_gpu() -> list[Any]:
            return cast(
                list[Any],
                model.collect(
                    first_batch_cpu,
                    get=[site],
                    out="gpu",
                    max_items=args.batch_size,
                    max_tokens=args.max_tokens,
                ),
            )

        collect_batch_gpu_report, collect_batch_gpu_outputs = _measure_case(
            name="collect_batch_gpu",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=_collect_batch_gpu,
        )
        collect_batch_gpu_outputs = cast(list[Any], collect_batch_gpu_outputs)
        collected_batch_activation = _stack_output_activations(collect_batch_gpu_outputs, site)
        manual_pooled = manual_mean_from_outputs(collect_batch_gpu_outputs, first_batch_cpu)
        _release_outputs(collect_batch_gpu_outputs)

        collect_batch_correct_outputs = cast(
            list[Any],
            model.collect(
                first_batch_cpu,
                get=[site],
                out="gpu",
                max_items=args.batch_size,
            ),
        )
        first_lengths = [
            int(value)
            for value in cast(torch.Tensor, first_batch_cpu["attention_mask"]).sum(dim=1).tolist()
        ]
        collect_batch_correct_rows = [
            cast(torch.Tensor, output[site]).detach().cpu()[:, : first_lengths[idx]]
            for idx, output in enumerate(collect_batch_correct_outputs)
        ]
        _release_outputs(collect_batch_correct_outputs)

        def _collect_dataset_cpu() -> dict[str, Any]:
            iterator = model.collect(
                prompt_batches,
                get=[site],
                out="cpu",
                max_items=args.batch_size,
                max_tokens=args.max_tokens,
            )
            return _consume_collect_iterator(iterator)

        collect_dataset_cpu_report, collect_dataset_cpu_meta = _measure_case(
            name="collect_dataset_cpu",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=_collect_dataset_cpu,
        )

        process_metrics: dict[str, float] = {"elapsed_s": 0.0, "calls": 0.0}

        def _process_mean(step: CollectStep) -> torch.Tensor:
            started = time.perf_counter()
            values = _step_activation_tensor(step, site)
            attention_mask = cast(torch.Tensor, step.batch["attention_mask"]).detach()
            mask = attention_mask.to(values.device, dtype=values.dtype).unsqueeze(-1)
            counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            result = ((values * mask).sum(dim=1) / counts.to(values.device, dtype=values.dtype)).detach().cpu()
            process_metrics["elapsed_s"] += time.perf_counter() - started
            process_metrics["calls"] += 1.0
            return result

        def _collect_dataset_process() -> dict[str, Any]:
            process_metrics["elapsed_s"] = 0.0
            process_metrics["calls"] = 0.0
            iterator = model.collect(
                prompt_batches,
                get=[site],
                out="gpu",
                process=_process_mean,
                max_items=args.batch_size,
                max_tokens=args.max_tokens,
            )
            rows = 0
            batches = 0
            for pooled in iterator:
                if isinstance(pooled, torch.Tensor):
                    rows += int(pooled.shape[0])
                batches += 1
            return {
                "rows": rows,
                "batches": batches,
                "process_elapsed_s": process_metrics["elapsed_s"],
                "process_calls": int(process_metrics["calls"]),
            }

        collect_dataset_process_report, collect_dataset_process_meta = _measure_case(
            name="collect_dataset_process_mean",
            device=device,
            warmup=args.warmup,
            trials=args.trials,
            fn=_collect_dataset_process,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir) / "export"

            def _collect_dataset_export() -> Any:
                return model.collect(
                    prompt_batches,
                    get=[site],
                    out=export_root,
                    max_items=args.batch_size,
                    max_tokens=args.max_tokens,
                )

            collect_dataset_export_report, collect_dataset_export_manifest = _measure_case(
                name="collect_dataset_export",
                device=device,
                warmup=args.warmup,
                trials=args.trials,
                fn=_collect_dataset_export,
            )

            profiler = {
                "collect_requests_cpu": _profile_case(_collect_requests_cpu, device=device),
                "collect_dataset_export": _profile_case(_collect_dataset_export, device=device),
            }
            manifest = cast(Any, collect_dataset_export_manifest)
            collect_dataset_export_report.update(
                {
                    "rows": int(manifest.rows),
                    "files_per_site": {path: len(files) for path, files in manifest.files.items()},
                }
            )

        forward_plain_report.update(
            {
                "shape": list(cast(torch.Tensor, forward_plain.logits).shape),
                "samples_per_second": int(first_batch["input_ids"].shape[0]) / forward_plain_report["median_s"],
                "tokens_per_second": int(first_batch["attention_mask"].sum().item()) / forward_plain_report["median_s"],
            }
        )
        forward_capture_report.update(
            {
                "shape": list(_capture_tensor(forward_capture, site).shape),
                "samples_per_second": int(first_batch["input_ids"].shape[0]) / forward_capture_report["median_s"],
                "tokens_per_second": int(first_batch["attention_mask"].sum().item()) / forward_capture_report["median_s"],
            }
        )
        generate_report.update(
            {
                "shape": list(cast(torch.Tensor, generate_output.sequences).shape),
                "samples_per_second": args.batch_size / generate_report["median_s"],
            }
        )
        collect_requests_cpu_report.update(
            {
                "rows": args.samples,
                "samples_per_second": args.samples / collect_requests_cpu_report["median_s"],
            }
        )
        collect_batch_gpu_report.update(
            {
                "rows": int(first_batch["input_ids"].shape[0]),
                "samples_per_second": int(first_batch["input_ids"].shape[0]) / collect_batch_gpu_report["median_s"],
                "tokens_per_second": int(first_batch["attention_mask"].sum().item()) / collect_batch_gpu_report["median_s"],
            }
        )
        collect_dataset_cpu_report.update(
            {
                **cast(dict[str, Any], collect_dataset_cpu_meta),
                "samples_per_second": args.samples / collect_dataset_cpu_report["median_s"],
            }
        )
        collect_dataset_process_report.update(
            {
                **cast(dict[str, Any], collect_dataset_process_meta),
                "samples_per_second": args.samples / collect_dataset_process_report["median_s"],
                "process_share_pct": (
                    100.0
                    * cast(dict[str, Any], collect_dataset_process_meta)["process_elapsed_s"]
                    / collect_dataset_process_report["median_s"]
                )
                if collect_dataset_process_report["median_s"] > 0
                else 0.0,
            }
        )

        raw_generate_rows: list[torch.Tensor] = []
        for prompt in prompts[: args.batch_size]:
            single = tokenizer(prompt, return_tensors="pt")
            single = {
                key: value.to(device)
                for key, value in single.items()
                if isinstance(value, torch.Tensor)
            }
            generated = cast(
                torch.Tensor,
                raw_model.generate(
                    **single,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                ),
            )
            raw_generate_rows.append(generated.detach().cpu())
        report_root = Path(args.json_output).resolve().parent if args.json_output else Path(tempfile.mkdtemp(prefix="real-profile-"))
        report_root.mkdir(parents=True, exist_ok=True)
        probelab_current_dir = report_root / "probelab-current"
        probelab_current = _run_probelab_worker(
            env_project_root=Path(args.probelab_current),
            probelab_root=Path(args.probelab_current),
            result_dir=probelab_current_dir,
            model_name=args.model,
            device=str(device),
            dtype=args.dtype,
            samples=args.samples,
            min_len=args.min_len,
            max_len=args.max_len,
            batch_size=args.batch_size,
            layer=layer,
            warmup=args.warmup,
            trials=args.trials,
        )

        report = {
            "environment": {
                **_environment_report(
                    raw_model,
                    model_name=args.model,
                    device=device,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    seq_len=args.max_len,
                ),
                "mirin_commit": _git_value(Path("/root/probes/mirin"), ["rev-parse", "HEAD"]),
                "probelab_current_commit": probelab_current["git"]["commit"],
            },
            "config": {
                "model": args.model,
                "device": str(device),
                "dtype": args.dtype,
                "samples": args.samples,
                "min_len": args.min_len,
                "max_len": args.max_len,
                "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
                "max_new_tokens": args.max_new_tokens,
                "layer": layer,
                "warmup": args.warmup,
                "trials": args.trials,
            },
            "mirin": {
                "cases": {
                    "forward_plain": forward_plain_report,
                    "forward_capture_only": forward_capture_report,
                    "generate_requests": generate_report,
                    "collect_requests_cpu": collect_requests_cpu_report,
                    "collect_batch_gpu": collect_batch_gpu_report,
                    "collect_dataset_cpu": collect_dataset_cpu_report,
                    "collect_dataset_process_mean": collect_dataset_process_report,
                    "collect_dataset_export": collect_dataset_export_report,
                },
                "correctness": {
                    "forward_logits_vs_raw_max_abs_diff": _max_abs_diff(
                        cast(torch.Tensor, forward_plain.logits),
                        raw_logits,
                    ),
                    "forward_capture_vs_raw_max_abs_diff": _max_abs_diff(
                        _capture_tensor(forward_capture, site),
                        raw_activation,
                    ),
                    "generate_vs_raw_max_abs_diff": _max_abs_diff(
                        torch.cat(
                            [
                                (left.detach().float() - right.detach().float()).abs().reshape(-1)
                                for left, right in zip(
                                    _generate_rows(generate_output),
                                    raw_generate_rows,
                                    strict=True,
                                )
                            ]
                        ).max().unsqueeze(0),
                        torch.zeros(1),
                    ),
                    "collect_batch_gpu_vs_raw_max_abs_diff": max(
                        _max_abs_diff(left, right)
                        for left, right in zip(collect_batch_correct_rows, raw_activation_rows, strict=True)
                    ),
                    "collect_process_mean_vs_manual_max_abs_diff": _max_abs_diff(
                        _gather_processed_tensors(
                            model.collect(
                                [first_batch_cpu],
                                get=[site],
                                out="gpu",
                                process=_process_mean,
                                max_items=args.batch_size,
                                max_tokens=args.max_tokens,
                            )
                        ),
                        manual_pooled.cpu(),
                    ),
                },
                "runtime_stats": model.stats(),
                "profiler": profiler,
            },
            "probelab_current": probelab_current,
        }
        if args.json_output is not None:
            output_path = Path(args.json_output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
