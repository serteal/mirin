"""Model API benchmarking harness for mirin."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median, pstdev
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

import mirin as ti
from mirin.hooks import _extract, _replace
from mirin.output import Output

from .tolerances import comparison_tolerances

DEFAULT_MODEL_NAMES = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3.5-4B",
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


@dataclass(slots=True)
class ModelApiBenchmarkConfig:
    """Configuration for the model API benchmark suite."""

    model_name: str
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 7
    seq_len: int = 256
    batch_size: int = 8
    micro_warmup: int = 5
    micro_trials: int = 20
    throughput_warmup: int = 1
    throughput_runs: int = 5
    sweep_width: int = 8
    get_one_stop_at_last: bool = True
    json_output: str | None = None

def run_model_api_benchmarks(config: ModelApiBenchmarkConfig) -> dict[str, Any]:
    """Run the full model API benchmark matrix and return a structured report."""

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)
    workload = _build_workload(config, device=device, dtype=dtype)
    raw_model = workload["model"]
    inputs = workload["inputs"]

    raw_case = _measure_case(
        "raw_forward",
        lambda: _run_raw(raw_model, inputs),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
    )
    raw_reference_output = _run_raw(raw_model, inputs)
    model = ti.Model(raw_model)

    get_proxies = _get_sites(model)
    map_proxies = _get_attn_sites(model, get_proxies)
    single_get = get_proxies[len(get_proxies) // 2]
    single_map = map_proxies[len(map_proxies) // 2]
    sweep_values = _sweep_values(config.sweep_width)

    correctness = {
        "passthrough": _compare_outputs(raw_reference_output, model(**inputs)),
        "get_one": _check_capture(raw_model, model, inputs, [single_get]),
        "get_many": _check_capture(raw_model, model, inputs, get_proxies),
        "map_one": _check_zero_map(raw_model, model, inputs, [single_map]),
        "map_many": _check_zero_map(raw_model, model, inputs, map_proxies),
        "batch_fused": _check_batch_fusion(model, inputs, single_map, sweep_values),
    }
    if config.get_one_stop_at_last:
        correctness["get_one_stop_at_last"] = _check_capture_only_stop(
            raw_model,
            model,
            inputs,
            single_get,
        )
    wrapped_case = _measure_case(
        "wrapped_passthrough",
        lambda: model(**inputs),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
        use_counters=True,
    )
    get_one_case = _measure_case(
        "get_one",
        lambda: model(**inputs, get=[single_get]),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
        use_counters=True,
    )
    get_many_case = _measure_case(
        "get_many",
        lambda: model(**inputs, get=get_proxies),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
        use_counters=True,
    )
    map_one_case = _measure_case(
        "map_one",
        lambda: model(**inputs, map={single_map: ti.zero()}),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
        use_counters=True,
    )
    map_many_case = _measure_case(
        "map_many",
        lambda: model(**inputs, map={proxy: ti.zero() for proxy in map_proxies}),
        warmup=config.micro_warmup,
        trials=config.micro_trials,
        device=device,
        use_counters=True,
    )
    batch_eager_case = _measure_case(
        "batch_eager",
        lambda: _run_batch_eager(model, inputs, single_map, sweep_values),
        warmup=config.throughput_warmup,
        trials=config.throughput_runs,
        device=device,
        use_counters=True,
    )
    batch_fused_case = _measure_case(
        "batch_fused",
        lambda: _run_batch_fused(model, inputs, single_map, sweep_values),
        warmup=config.throughput_warmup,
        trials=config.throughput_runs,
        device=device,
        use_counters=True,
    )

    cases = [
        raw_case,
        wrapped_case,
        get_one_case,
        get_many_case,
        map_one_case,
        map_many_case,
        batch_eager_case,
        batch_fused_case,
    ]
    if config.get_one_stop_at_last:
        cases.insert(
            3,
            _measure_case(
                "get_one_stop_at_last",
                lambda: model(**inputs, get=[single_get], stop_at_last_get=True)[single_get],
                warmup=config.micro_warmup,
                trials=config.micro_trials,
                device=device,
                use_counters=True,
            ),
        )
    _annotate_case_metrics(
        cases,
        raw_name="raw_forward",
        batch_names=("batch_eager", "batch_fused"),
    )

    report = {
        "environment": _environment_report(
            raw_model,
            model_name=config.model_name,
            device=device,
            dtype=dtype,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
        ),
        "config": asdict(config),
        "correctness": correctness,
        "cases": cases,
    }
    if config.json_output is not None:
        output_path = Path(config.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_model_api_suite(
    configs: list[ModelApiBenchmarkConfig],
    *,
    json_output: str | None = None,
) -> dict[str, Any]:
    """Run several model API benchmark configs and return an aggregate report."""

    reports = []
    for config in configs:
        config.json_output = None
        try:
            reports.append(run_model_api_benchmarks(config))
        except Exception as exc:
            reports.append(
                {
                    "ok": False,
                    "model_name": config.model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    suite = {"reports": reports}
    if json_output is not None:
        output_path = Path(json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8")
    return suite


def format_model_api_report(report: dict[str, Any]) -> str:
    """Render a structured report as readable plain text."""

    env = report["environment"]
    lines = [
        "Model API Benchmark Report",
        f"model: {env['model_name']}  attn={env['attention_impl']}",
        (
            f"device: {env['device']}  dtype={env['dtype']}  gpu={env['gpu_name']}  "
            f"torch={env['torch_version']}  cuda={env['cuda_version']}"
        ),
        (
            f"shape: layers={env['n_layers']} width={env['width']} heads={env['n_heads']} "
            f"batch={env['batch_size']} seq={env['seq_len']} vocab={env['vocab_size']}"
        ),
        "",
        "Correctness",
    ]
    for name, check in report["correctness"].items():
        status = "ok" if check["ok"] else "FAIL"
        lines.append(f"  {name:<14} {status}  max_abs_diff={check['max_abs_diff']:.6g}")

    lines.append("")
    lines.append("Cases")
    for case in report["cases"]:
        parts = [
            f"  {case['name']:<18}",
            f"median={case['median_ms']:.3f}ms",
            f"p90={case['p90_ms']:.3f}ms",
            f"std={case['std_ms']:.3f}ms",
        ]
        overhead = case.get("overhead_vs_raw_pct")
        if overhead is not None:
            parts.append(f"vs_raw={overhead:+.2f}%")
        speedup = case.get("speedup_vs_baseline")
        if speedup is not None:
            parts.append(f"speedup={speedup:.2f}x")
        examples_per_second = case.get("examples_per_second")
        if examples_per_second is not None:
            parts.append(f"ex/s={examples_per_second:.1f}")
        tokens_per_second = case.get("tokens_per_second")
        if tokens_per_second is not None:
            parts.append(f"tok/s={tokens_per_second:.1f}")
        user_calls = case.get("user_calls")
        forward_passes = case.get("forward_passes")
        if user_calls is not None and forward_passes is not None:
            parts.append(f"calls={user_calls}/{forward_passes}")
        gpu_peak_mb = case.get("gpu_peak_memory_mb")
        if gpu_peak_mb is not None:
            parts.append(f"gpu_peak={gpu_peak_mb:.1f}MB")
        lines.append("  ".join(parts))

    lines.append("")
    lines.append("Counters")
    for case in report["cases"]:
        summary = case.get("counters_summary")
        if summary is None:
            continue
        lines.append(f"  {case['name']}:")
        for line in summary.splitlines():
            lines.append(f"    {line}")
    return "\n".join(lines)


def format_model_api_suite(suite: dict[str, Any]) -> str:
    """Render several benchmark reports as one printable block."""

    reports = suite["reports"]
    blocks = []
    for report in reports:
        if report.get("ok", True):
            blocks.append(format_model_api_report(report))
            continue
        blocks.append(
            "\n".join(
                [
                    "Model API Benchmark Report",
                    f"model: {report['model_name']}",
                    f"status: FAILED ({report['error_type']})",
                    f"error: {report['error']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _prepare_model(model: nn.Module, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model.eval()
    if device.type == "cpu":
        return model.to(device=device, dtype=dtype)
    return model.to(device=device, dtype=dtype)


def _run_raw(model: nn.Module, inputs: dict[str, torch.Tensor]) -> Any:
    with torch.no_grad():
        return model(**inputs)


def _get_sites(model: ti.Model) -> list[Any]:
    indices = sorted({0, len(model.layers) // 2, len(model.layers) - 1})
    return [model.layers[idx] for idx in indices]


def _get_attn_sites(model: ti.Model, layer_sites: list[Any]) -> list[Any]:
    attn_sites: list[Any] = []
    for layer_site in layer_sites:
        attn = ti.find(layer_site, "attn")
        if attn is None:
            raise RuntimeError(f"Could not find attention module under {layer_site.path}.")
        attn_sites.append(attn)
    return attn_sites


def _check_capture(
    raw_model: nn.Module,
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxies: list[Any],
) -> dict[str, Any]:
    manual: dict[str, torch.Tensor] = {}
    handles = []
    try:
        for proxy in proxies:
            module = _get_module(raw_model, proxy.path)

            def capture(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: object,
                *,
                _path: str = proxy.path,
            ) -> None:
                manual[_path] = _extract(output).detach()

            handles.append(module.register_forward_hook(capture))
        raw_output = _run_raw(raw_model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    wrapped_output = model(**inputs, get=proxies)
    max_abs_diff = _compare_logits(raw_output, wrapped_output)
    for proxy in proxies:
        diff = _max_abs_diff(wrapped_output[proxy], manual[proxy.path])
        max_abs_diff = max(max_abs_diff, diff)
    tolerance = max(
        _tolerance_for(wrapped_output.logits),
        max(_tolerance_for(wrapped_output[proxy]) for proxy in proxies),
    )
    return {"ok": max_abs_diff <= tolerance, "max_abs_diff": max_abs_diff}


def _check_capture_only_stop(
    raw_model: nn.Module,
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxy: Any,
) -> dict[str, Any]:
    manual: dict[str, torch.Tensor] = {}
    module = _get_module(raw_model, proxy.path)

    def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        manual["act"] = _extract(output).detach()

    handle = module.register_forward_hook(capture)
    try:
        _ = _run_raw(raw_model, inputs)
    finally:
        handle.remove()

    wrapped_output = model(**inputs, get=[proxy], stop_at_last_get=True)
    max_abs_diff = _max_abs_diff(wrapped_output[proxy], manual["act"])
    tolerance = _tolerance_for(wrapped_output[proxy])
    return {"ok": max_abs_diff <= tolerance, "max_abs_diff": max_abs_diff}


def _check_zero_map(
    raw_model: nn.Module,
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxies: list[Any],
) -> dict[str, Any]:
    handles = []
    try:
        for proxy in proxies:
            module = _get_module(raw_model, proxy.path)

            def zero_hook(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: object,
            ) -> object:
                return _replace(output, torch.zeros_like(_extract(output)))

            handles.append(module.register_forward_hook(zero_hook))
        raw_output = _run_raw(raw_model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    wrapped_output = model(**inputs, map={proxy: ti.zero() for proxy in proxies})
    max_abs_diff = _compare_logits(raw_output, wrapped_output)
    return {
        "ok": max_abs_diff <= _tolerance_for(wrapped_output.logits),
        "max_abs_diff": max_abs_diff,
    }


def _check_batch_fusion(
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxy: Any,
    sweep_values: tuple[float, ...],
) -> dict[str, Any]:
    eager_outputs = _run_batch_eager(model, inputs, proxy, sweep_values)
    fused_outputs = _run_batch_fused(model, inputs, proxy, sweep_values)
    max_abs_diff = 0.0
    for eager, fused in zip(eager_outputs, fused_outputs, strict=True):
        max_abs_diff = max(max_abs_diff, _compare_logits(eager, fused))
    return {
        "ok": max_abs_diff <= _tolerance_for(eager_outputs[0].logits),
        "max_abs_diff": max_abs_diff,
    }


def _run_batch_eager(
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxy: Any,
    sweep_values: tuple[float, ...],
) -> list[Output]:
    return [model(**inputs, map={proxy: ti.add(value)}) for value in sweep_values]


def _run_batch_fused(
    model: ti.Model,
    inputs: dict[str, torch.Tensor],
    proxy: Any,
    sweep_values: tuple[float, ...],
) -> list[Output]:
    with ti.batch():
        outputs = [model(**inputs, map={proxy: ti.add(value)}) for value in sweep_values]
    return outputs


def _measure_case(
    name: str,
    fn: Any,
    *,
    warmup: int,
    trials: int,
    device: torch.device,
    use_counters: bool = False,
) -> dict[str, Any]:
    for _ in range(warmup):
        result = fn()
        _sync(device)
        del result

    if use_counters:
        ti.Counters.reset()
    _reset_peak_memory(device)

    rss_after = 0
    times_ms: list[float] = []
    for _ in range(trials):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        rss_after = max(rss_after, _current_rss_bytes())
        del result

    counters = _snapshot_counters() if use_counters else None
    return {
        "name": name,
        "times_ms": [round(value, 6) for value in times_ms],
        "median_ms": median(times_ms),
        "p90_ms": _percentile(times_ms, 0.90),
        "std_ms": pstdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "gpu_peak_memory_mb": _gpu_peak_memory_mb(device),
        "rss_after_mb": rss_after / (1024 * 1024) if rss_after else None,
        "counters": counters,
        "counters_summary": ti.Counters.summary() if counters is not None else None,
        "user_calls": counters["calls"] if counters is not None else None,
        "forward_passes": counters["forward_passes"] if counters is not None else None,
    }


def _annotate_case_metrics(
    cases: list[dict[str, Any]],
    *,
    raw_name: str,
    batch_names: tuple[str, str],
) -> None:
    by_name = {case["name"]: case for case in cases}
    raw_case = by_name[raw_name]
    for case in cases:
        if case["name"] != raw_name:
            case["overhead_vs_raw_pct"] = (
                (case["median_ms"] / raw_case["median_ms"]) - 1.0
            ) * 100.0

    eager_case = by_name[batch_names[0]]
    fused_case = by_name[batch_names[1]]
    fused_case["speedup_vs_baseline"] = eager_case["median_ms"] / fused_case["median_ms"]


def _environment_report(
    raw_model: nn.Module,
    *,
    model_name: str | None,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    config = getattr(raw_model, "config", SimpleNamespace())
    return {
        "model_name": model_name or getattr(config, "model_type", type(raw_model).__name__),
        "architecture": type(raw_model).__name__,
        "attention_impl": _config_value(config, "_attn_implementation", "attn_implementation")
        or "unknown",
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "n_layers": _config_value(config, "n_layer", "num_hidden_layers", "num_layers"),
        "n_heads": _config_value(config, "n_head", "num_attention_heads"),
        "width": _config_value(config, "n_embd", "hidden_size", "d_model"),
        "vocab_size": _config_value(config, "vocab_size"),
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


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
        choices = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose from: {choices}.") from exc


def _snapshot_counters() -> dict[str, int]:
    fields = (
        "calls",
        "forward_passes",
        "total_time_ns",
        "forward_time_ns",
        "hook_overhead_ns",
        "activations_captured",
        "activations_bytes",
        "buffer_pool_hits",
        "buffer_pool_misses",
        "maps_applied",
        "batch_groups",
        "batch_fusions",
        "prefix_layers_saved",
    )
    return {field: int(getattr(ti.Counters, field)) for field in fields}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return 0
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return 0


def _compare_outputs(left: Any, right: Any) -> dict[str, Any]:
    diff = _compare_logits(left, right)
    logits = _logits_tensor(left)
    return {"ok": diff <= _tolerance_for(logits), "max_abs_diff": diff}


def _compare_logits(left: Any, right: Any) -> float:
    return _max_abs_diff(_logits_tensor(left), _logits_tensor(right))


def _logits_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Output):
        return value.logits
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"Cannot extract logits from {type(value).__name__}.")


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    left_cpu = left.detach().float().to("cpu")
    right_cpu = right.detach().float().to("cpu")
    return float((left_cpu - right_cpu).abs().max().item())


def _tolerance_for(tensor: torch.Tensor) -> float:
    return comparison_tolerances(tensor, mode="same_impl")[0]


def _get_module(model: nn.Module, path: str) -> nn.Module:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Path {path!r} does not resolve to a module.")
    return current


def _reshape_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    d_head = tensor.shape[-1] // n_heads
    return tensor.view(*tensor.shape[:-1], n_heads, d_head)


def _sweep_values(width: int) -> tuple[float, ...]:
    if width < 1:
        raise ValueError("sweep_width must be at least 1.")
    if width == 1:
        return (0.5,)
    return tuple(-1.0 + (2.0 * idx / (width - 1)) for idx in range(width))


def _build_workload(
    config: ModelApiBenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model, tokenizer = _load_hf_workload_model(config.model_name, device=device, dtype=dtype)
    prompts = _prompt_pool(config.batch_size)
    return {
        "model": model,
        "inputs": _tokenize_prompts(
            tokenizer,
            prompts[: config.batch_size],
            seq_len=config.seq_len,
            device=device,
        ),
    }


def _load_hf_workload_model(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Real-model benchmarks require `transformers`.\n"
            "Install with: uv sync --extra transformers"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    try:
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
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load benchmark model {model_name!r}. "
            "Check HuggingFace access, token permissions, and model availability."
        ) from exc

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return _prepare_model(model, device=device, dtype=dtype), tokenizer


def _prompt_pool(total: int) -> list[str]:
    return [f"Prompt {idx}: {_PROMPT_STEMS[idx % len(_PROMPT_STEMS)]}" for idx in range(total)]


def _tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return {
        key: value.to(device=device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sample.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _config_value(config: Any, *names: str) -> Any | None:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    text_config = _text_config(config)
    if text_config is not None:
        for name in names:
            if hasattr(text_config, name):
                return getattr(text_config, name)
    return None


def _text_config(config: Any) -> Any | None:
    getter = getattr(config, "get_text_config", None)
    if not callable(getter):
        return None
    for kwargs in ({"decoder": True}, {}):
        try:
            return getter(**kwargs)
        except TypeError:
            continue
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None
