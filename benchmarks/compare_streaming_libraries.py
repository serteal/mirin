"""Large-workload dataset-loop comparison for tinyinterp, TransformerLens, and nnterp."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import tinyinterp as ti

from .compare_libraries import (
    _compare_tensors,
    _import_transformers,
    _library_versions,
    _load_hf_model,
    _load_nnterp_model,
    _load_transformerlens_model,
    _manual_capture,
    _release_models,
    _run_lens_capture,
    _run_nnterp_capture,
)
from .model_api import (
    _environment_report,
    _measure_case,
    _resolve_device,
    _resolve_dtype,
)

DEFAULT_STREAMING_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_STREAMING_HF_BLOCK_PATH = "model.layers.0"
DEFAULT_STREAMING_LENS_HOOK_NAME = "blocks.0.hook_resid_post"


@dataclass(slots=True)
class StreamingCompareConfig:
    """Configuration for the large-workload dataset-loop comparison."""

    model_name: str = DEFAULT_STREAMING_MODEL_NAME
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 17
    chunk_batch_size: int = 8
    source_batch_size: int = 16
    seq_len: int = 512
    dataset_batches: int = 16
    warmup: int = 1
    trials: int = 5
    hf_block_path: str = DEFAULT_STREAMING_HF_BLOCK_PATH
    lens_hook_name: str = DEFAULT_STREAMING_LENS_HOOK_NAME
    json_output: str | None = None


def run_streaming_compare_benchmarks(config: StreamingCompareConfig) -> dict[str, Any]:
    """Run a large dataset-style capture benchmark across libraries."""

    if config.source_batch_size % config.chunk_batch_size != 0:
        raise ValueError("source_batch_size must be divisible by chunk_batch_size.")

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)
    transformers = _import_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)

    raw_model = _load_hf_model(config.model_name, device=device, dtype=dtype)

    dataset = _make_dataset(
        vocab_size=int(raw_model.config.vocab_size),
        dataset_batches=config.dataset_batches,
        source_batch_size=config.source_batch_size,
        seq_len=config.seq_len,
        seed=config.seed + 1,
        device=device,
    )
    first_chunk = _iter_chunks(dataset[0], config.chunk_batch_size)[0]

    reference_activation = _manual_capture(raw_model, first_chunk, path=config.hf_block_path)
    env = _environment_report(
        raw_model,
        model_name=config.model_name,
        device=device,
        dtype=dtype,
        batch_size=config.chunk_batch_size,
        seq_len=config.seq_len,
    )
    raw_case = _tag_stream_case(
        _measure_case(
            "raw_hf_hook_loop",
            lambda model=raw_model: _run_raw_stream_loop(
                model,
                dataset,
                path=config.hf_block_path,
                chunk_batch_size=config.chunk_batch_size,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
        ),
        library="raw_hf",
    )
    del raw_model
    _release_models(device)

    tiny_wrapped = _load_hf_model(config.model_name, device=device, dtype=dtype)
    tiny_model = ti.Model(tiny_wrapped)
    tiny_site = tiny_model.layers[0]
    tiny_manual_first = tiny_model(**first_chunk, get=[tiny_site])[tiny_site]
    tiny_case = _tag_stream_case(
        _measure_case(
            "tinyinterp_manual_loop",
            lambda model=tiny_model, site=tiny_site: _run_tinyinterp_manual_loop(
                model,
                dataset,
                site,
                chunk_batch_size=config.chunk_batch_size,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        library="tinyinterp_manual",
    )
    tiny_site_path = tiny_site.path
    del tiny_model
    del tiny_wrapped
    _release_models(device)

    tl_source = _load_hf_model(config.model_name, device=device, dtype=dtype)
    lens_model = _load_transformerlens_model(
        config.model_name,
        hf_model=tl_source,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    del tl_source
    lens_first = _run_lens_activation(lens_model, first_chunk["input_ids"], config.lens_hook_name)
    lens_case = _tag_stream_case(
        _measure_case(
            "transformerlens_loop",
            lambda model=lens_model: _run_transformerlens_loop(
                model,
                dataset,
                hook_name=config.lens_hook_name,
                chunk_batch_size=config.chunk_batch_size,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
        ),
        library="transformerlens",
    )
    del lens_model
    _release_models(device)

    nn_model = _load_nnterp_model(config.model_name, device=device, dtype=dtype)
    nn_first = _run_nnterp_capture(nn_model, first_chunk)["activation"]
    nn_case = _tag_stream_case(
        _measure_case(
            "nnterp_loop",
            lambda model=nn_model: _run_nnterp_loop(
                model,
                dataset,
                chunk_batch_size=config.chunk_batch_size,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
        ),
        library="nnterp",
    )
    del nn_model
    _release_models(device)

    correctness = {
        "tinyinterp_manual": _compare_tensors(
            reference_activation.detach().cpu(),
            tiny_manual_first.detach().cpu(),
        ),
        "transformerlens": _compare_tensors(
            reference_activation.detach().cpu(),
            lens_first.detach().cpu(),
        ),
        "nnterp": _compare_tensors(
            reference_activation.detach().cpu(),
            nn_first.detach().cpu(),
        ),
    }

    total_examples = config.dataset_batches * config.source_batch_size
    cases = [raw_case, tiny_case, lens_case, nn_case]
    _annotate_stream_metrics(
        cases,
        raw_name="raw_hf_hook_loop",
        total_examples=total_examples,
        seq_len=config.seq_len,
    )

    report = {
        "config": asdict(config),
        "environment": {
            **env,
            "source_batch_size": config.source_batch_size,
            "dataset_batches": config.dataset_batches,
            "total_examples": total_examples,
            "site_mapping": {
                "hf_block_path": config.hf_block_path,
                "tinyinterp_site": tiny_site_path,
                "transformerlens_hook": config.lens_hook_name,
                "nnterp_site": "layers_output[0]",
            },
            "library_versions": _library_versions(),
        },
        "correctness": correctness,
        "cases": cases,
    }
    if config.json_output is not None:
        path = Path(config.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_streaming_compare_report(report: dict[str, Any]) -> str:
    """Format a large-workload dataset-loop comparison report."""

    env = report["environment"]
    lines = [
        f"Model: {env['model_name']}",
        f"Device: {env['device']} ({env['gpu_name']})",
        f"Dtype: {env['dtype']}",
        (
            f"Workload: dataset_batches={env['dataset_batches']} "
            f"source_batch={env['source_batch_size']} chunk_batch={env['batch_size']} "
            f"seq={env['seq_len']} total_examples={env['total_examples']}"
        ),
        "",
        "Correctness:",
    ]
    for name, result in report["correctness"].items():
        status = "ok" if result["ok"] else "FAIL"
        lines.append(f"- {name}: {status} (max_abs_diff={result['max_abs_diff']:.6f})")

    lines.append("")
    lines.append("Timing:")
    for case in report["cases"]:
        if case.get("skipped"):
            lines.append(f"- {case['library']}: skipped ({case['skipped']})")
            continue
        lines.append(
            f"- {case['library']}: {case['median_ms']:.3f}ms "
            f"ex/s={case['examples_per_second']:.1f} "
            f"tok/s={case['tokens_per_second']:.1f}"
        )
    return "\n".join(lines)


def _compare_config_from_streaming(config: StreamingCompareConfig) -> Any:
    from .compare_libraries import CompareConfig

    return CompareConfig(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        batch_size=config.chunk_batch_size,
        seq_len=config.seq_len,
        warmup=1,
        trials=1,
        hf_block_path=config.hf_block_path,
        lens_hook_name=config.lens_hook_name,
        json_output=None,
    )


def _make_dataset(
    *,
    vocab_size: int,
    dataset_batches: int,
    source_batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(
                0,
                vocab_size,
                (source_batch_size, seq_len),
                generator=generator,
                device=device,
            ),
        }
        for _ in range(dataset_batches)
    ]


def _iter_chunks(batch: dict[str, Any], chunk_batch_size: int) -> list[dict[str, Any]]:
    total = int(batch["input_ids"].shape[0])
    return [
        {key: _slice_value(value, start, end) for key, value in batch.items()}
        for start in range(0, total, chunk_batch_size)
        for end in [min(total, start + chunk_batch_size)]
    ]


def _slice_value(value: Any, start: int, end: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[start:end]
    return value


def _run_raw_stream_loop(
    model: Any,
    dataset: list[dict[str, Any]],
    *,
    path: str,
    chunk_batch_size: int,
) -> list[float]:
    return [
        _summarize_activation(
            _manual_capture(model, chunk, path=path).detach().to("cpu", non_blocking=True)
        )
        for batch in dataset
        for chunk in _iter_chunks(batch, chunk_batch_size)
    ]


def _run_tinyinterp_manual_loop(
    model: ti.Model,
    dataset: list[dict[str, Any]],
    site: Any,
    *,
    chunk_batch_size: int,
) -> list[float]:
    summaries: list[float] = []
    for batch in dataset:
        for chunk in _iter_chunks(batch, chunk_batch_size):
            output = model(**chunk, get=[site])
            activation = output[site].detach().to("cpu", non_blocking=True)
            summaries.append(_summarize_activation(activation))
    return summaries


def _run_transformerlens_loop(
    model: Any,
    dataset: list[dict[str, Any]],
    *,
    hook_name: str,
    chunk_batch_size: int,
) -> list[float]:
    summaries: list[float] = []
    for batch in dataset:
        for chunk in _iter_chunks(batch, chunk_batch_size):
            activation = _run_lens_activation(model, chunk["input_ids"], hook_name)
            cpu_activation = activation.detach().to("cpu", non_blocking=True)
            summaries.append(_summarize_activation(cpu_activation))
    return summaries


def _run_nnterp_loop(
    model: Any,
    dataset: list[dict[str, Any]],
    *,
    chunk_batch_size: int,
) -> list[float]:
    summaries: list[float] = []
    for batch in dataset:
        for chunk in _iter_chunks(batch, chunk_batch_size):
            activation = _run_nnterp_capture(model, chunk)["activation"]
            cpu_activation = activation.detach().to("cpu", non_blocking=True)
            summaries.append(_summarize_activation(cpu_activation))
    return summaries


def _run_lens_activation(
    model: Any,
    input_ids: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    # Avoid ActivationCache object construction for the one-site capture path.
    result = _run_lens_capture(model, input_ids, hook_name=hook_name)
    return result["activation"]


def _summarize_activation(activation: torch.Tensor) -> float:
    return float(activation.float().mean(dim=-1).sum().item())


def _tag_stream_case(case: dict[str, Any], *, library: str) -> dict[str, Any]:
    case["library"] = library
    return case


def _annotate_stream_metrics(
    cases: list[dict[str, Any]],
    *,
    raw_name: str,
    total_examples: int,
    seq_len: int,
) -> None:
    by_name = {case["name"]: case for case in cases}
    raw_case = by_name[raw_name]
    for case in cases:
        if case.get("skipped"):
            continue
        median_ms = case.get("median_ms")
        raw_median_ms = raw_case.get("median_ms")
        if not isinstance(median_ms, (int, float)) or median_ms <= 0:
            continue
        if case["name"] != raw_name:
            if not isinstance(raw_median_ms, (int, float)) or raw_median_ms <= 0:
                continue
            case["overhead_vs_raw_pct"] = ((median_ms / raw_median_ms) - 1.0) * 100.0
        total_seconds = median_ms / 1000.0
        case["examples_per_second"] = total_examples / total_seconds
        case["tokens_per_second"] = (total_examples * seq_len) / total_seconds
