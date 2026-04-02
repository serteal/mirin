"""Cross-library benchmark harness for mirin, TransformerLens, and nnterp."""

from __future__ import annotations

import gc
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

import mirin as ti
from mirin.hooks import _extract, _replace

from .model_api import (
    _config_value,
    _environment_report,
    _measure_case,
    _resolve_device,
    _resolve_dtype,
)
from .support import local_support_map, runnable_support
from .tolerances import compare_tensors

DEFAULT_COMPARE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_HF_BLOCK_PATH = "model.layers.0"
DEFAULT_LENS_HOOK_NAME = "blocks.0.hook_resid_post"
DEFAULT_LENS_STOP_AT_LAYER = 1


@dataclass(slots=True)
class CompareConfig:
    """Configuration for the cross-library benchmark suite."""

    model_name: str = DEFAULT_COMPARE_MODEL_NAME
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 11
    batch_size: int = 4
    seq_len: int = 128
    warmup: int = 5
    trials: int = 20
    model_family: str = "custom"
    hf_block_path: str | None = DEFAULT_HF_BLOCK_PATH
    lens_hook_name: str = DEFAULT_LENS_HOOK_NAME
    lens_stop_at_layer: int = DEFAULT_LENS_STOP_AT_LAYER
    json_output: str | None = None


def run_compare_benchmarks(config: CompareConfig) -> dict[str, Any]:
    """Run the cross-library benchmark suite and return a structured report."""

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)
    tokenizer, inputs = _prepare_compare_inputs(config, device=device, dtype=dtype)
    raw_inputs = dict(inputs)
    support = local_support_map(config.model_family)
    probe_model = ti.Model(_load_hf_model(config.model_name, device=device, dtype=dtype))
    probe_site = probe_model.layers[0]
    manual_path = config.hf_block_path or probe_site.path
    del probe_model
    _release_models(device)

    raw_model = _load_hf_model(config.model_name, device=device, dtype=dtype)
    env = _environment_report(
        raw_model,
        model_name=config.model_name,
        device=device,
        dtype=dtype,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    raw_case = _tag_case(
        _measure_case(
            "raw_hf_forward",
            lambda model=raw_model: _run_raw_forward(model, raw_inputs),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
        ),
        library="raw_hf",
        operation="forward",
    )
    raw_logits = _run_raw_forward(raw_model, raw_inputs)
    manual_capture = _manual_capture(raw_model, raw_inputs, path=manual_path)
    manual_zero_logits = _manual_zero(raw_model, raw_inputs, path=manual_path)
    del raw_model
    _release_models(device)

    tiny_wrapped = _load_hf_model(config.model_name, device=device, dtype=dtype)
    tiny_model = ti.Model(tiny_wrapped)
    tiny_site = tiny_model.layers[0]
    tiny_forward = tiny_model(**raw_inputs)
    tiny_capture = tiny_model(**raw_inputs, get=[tiny_site])
    tiny_capture_only = tiny_model(**raw_inputs, get=[tiny_site], stop_at_last_get=True)[tiny_site]
    tiny_zero = tiny_model(**raw_inputs, map={tiny_site: ti.zero()})
    tiny_cases = [
        _tag_case(
            _measure_case(
                "mirin_forward",
                lambda model=tiny_model: model(**raw_inputs),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
                use_counters=True,
            ),
            library="mirin",
            operation="forward",
        ),
        _tag_case(
            _measure_case(
                "mirin_get_one",
                lambda model=tiny_model, site=tiny_site: model(**raw_inputs, get=[site]),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
                use_counters=True,
            ),
            library="mirin",
            operation="get_one",
        ),
        _tag_case(
            _measure_case(
                "mirin_capture_only",
                lambda model=tiny_model, site=tiny_site: model(
                    **raw_inputs,
                    get=[site],
                    stop_at_last_get=True,
                )[site],
                warmup=config.warmup,
                trials=config.trials,
                device=device,
                use_counters=True,
            ),
            library="mirin",
            operation="capture_only",
        ),
        _tag_case(
            _measure_case(
                "mirin_map_one",
                lambda model=tiny_model, site=tiny_site: model(
                    **raw_inputs,
                    map={site: ti.zero()},
                ),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
                use_counters=True,
            ),
            library="mirin",
            operation="map_one",
        ),
    ]
    tiny_site_path = tiny_site.path
    del tiny_model
    del tiny_wrapped
    _release_models(device)

    lens_input_ids = inputs["input_ids"]
    lens_cases, lens_correctness = _run_transformerlens_compare(
        config,
        tokenizer=tokenizer,
        inputs=raw_inputs,
        lens_input_ids=lens_input_ids,
        raw_logits=raw_logits,
        manual_capture=manual_capture,
        manual_zero_logits=manual_zero_logits,
        device=device,
        dtype=dtype,
    )
    _release_models(device)

    nn_cases, nn_correctness = _run_nnterp_compare(
        config,
        inputs=raw_inputs,
        raw_logits=raw_logits,
        manual_capture=manual_capture,
        manual_zero_logits=manual_zero_logits,
        device=device,
        dtype=dtype,
    )
    _release_models(device)

    correctness = {
        "mirin_forward": _compare_tensors(raw_logits, tiny_forward.logits),
        "mirin_get_one_logits": _compare_tensors(raw_logits, tiny_capture.logits),
        "mirin_get_one_activation": _compare_tensors(manual_capture, tiny_capture[tiny_site]),
        "mirin_capture_only_activation": _compare_tensors(manual_capture, tiny_capture_only),
        "mirin_map_one": _compare_tensors(manual_zero_logits, tiny_zero.logits),
    }
    correctness.update(lens_correctness)
    correctness.update(nn_correctness)

    cases = [raw_case, *tiny_cases, *lens_cases, *nn_cases]
    _annotate_against_raw(cases, raw_name="raw_hf_forward")

    report = {
        "config": asdict(config),
        "environment": {
            **env,
            "site_mapping": {
                "hf_block_path": manual_path,
                "mirin_site": tiny_site_path,
                "transformerlens_hook": config.lens_hook_name,
                "transformerlens_stop_at_layer": config.lens_stop_at_layer,
                "nnterp_site": "layers_output[0]",
            },
            "library_versions": _library_versions(),
        },
        "support": support,
        "correctness": correctness,
        "cases": cases,
    }
    if config.json_output is not None:
        path = Path(config.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_compare_report(report: dict[str, Any]) -> str:
    """Format a cross-library benchmark report for CLI output."""

    env = report["environment"]
    lines = [
        f"Model: {env['model_name']}",
        f"Device: {env['device']} ({env['gpu_name']})",
        f"Dtype: {env['dtype']}",
        f"Shape: batch={env['batch_size']} seq={env['seq_len']}",
        "",
        "Support:",
    ]
    for entry in report.get("support", {}).values():
        state = "ok" if entry["runnable"] else "skip"
        reason = f" ({entry['reason']})" if entry["reason"] else ""
        lines.append(f"- {entry['name']}: {state}{reason}")
    lines.extend(
        [
        "",
        "Correctness:",
        ]
    )
    for name, result in report["correctness"].items():
        if result.get("skipped"):
            lines.append(f"- {name}: skipped ({result['reason']})")
            continue
        status = "ok" if result["ok"] else "FAIL"
        lines.append(f"- {name}: {status} (max_abs_diff={result['max_abs_diff']:.6f})")

    lines.append("")
    lines.append("Timing:")
    by_operation: dict[str, list[dict[str, Any]]] = {
        "forward": [],
        "get_one": [],
        "capture_only": [],
        "map_one": [],
    }
    for case in report["cases"]:
        by_operation.setdefault(case["operation"], []).append(case)
    for operation in ("forward", "get_one", "capture_only", "map_one"):
        lines.append(f"- {operation}:")
        for case in by_operation[operation]:
            if case.get("skipped"):
                lines.append(f"  {case['library']}: skipped ({case['skipped']})")
                continue
            suffix = ""
            overhead = case.get("overhead_vs_raw_pct")
            if overhead is not None:
                suffix = f" ({overhead:+.2f}% vs raw)"
            lines.append(f"  {case['library']}: {case['median_ms']:.3f}ms{suffix}")
    return "\n".join(lines)


def _prepare_compare_inputs(
    config: CompareConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, dict[str, Any]]:
    transformers = _import_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    raw_model = _load_hf_model(config.model_name, device=device, dtype=dtype)

    input_ids = _make_input_ids(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        vocab_size=int(_config_value(raw_model.config, "vocab_size") or 256),
        seed=config.seed + 1,
        device=device,
    )
    del raw_model
    _release_models(device)
    return tokenizer, {"input_ids": input_ids, "use_cache": False}


def _release_models(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "The comparison benchmark requires `transformers`.\n"
            "Install with: `uv sync --extra transformers --group bench`."
        ) from exc
    return transformers


def _import_transformerlens() -> Any:
    try:
        import transformer_lens
    except ImportError as exc:
        raise ImportError(
            "The comparison benchmark requires `transformer-lens`.\n"
            "Install with: `uv sync --extra transformers --group bench`."
        ) from exc
    return transformer_lens


def _import_nnterp() -> Any:
    try:
        import nnterp
    except ImportError as exc:
        raise ImportError(
            "The comparison benchmark requires `nnterp`.\n"
            "Install with: `uv sync --extra transformers --group bench`."
        ) from exc
    return nnterp


def _load_hf_model(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    transformers = _import_transformers()
    model = cast(
        nn.Module,
        transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype),
    )
    model.to(device)
    model.eval()
    return model


def _load_transformerlens_model(
    model_name: str,
    *,
    hf_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    transformer_lens = _import_transformerlens()
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        move_to_device=True,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        default_prepend_bos=False,
        dtype=dtype,
    )
    model.eval()
    return model


def _load_nnterp_model(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    nnterp = _import_nnterp()
    model = _load_hf_model(model_name, device=device, dtype=dtype)
    return nnterp.StandardizedTransformer(
        model,
        check_renaming=False,
        allow_dispatch=False,
    )


def _make_input_ids(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device=device)


def _run_raw_forward(model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        return cast(torch.Tensor, model(**inputs).logits)


def _manual_capture(
    model: nn.Module,
    inputs: dict[str, Any],
    *,
    path: str,
) -> torch.Tensor:
    module = _get_module(model, path)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        captured["activation"] = _extract(output).detach()

    handle = module.register_forward_hook(capture)
    try:
        _ = _run_raw_forward(model, inputs)
    finally:
        handle.remove()
    return captured["activation"]


def _manual_zero(
    model: nn.Module,
    inputs: dict[str, Any],
    *,
    path: str,
) -> torch.Tensor:
    module = _get_module(model, path)

    def zero(_module: nn.Module, _args: tuple[object, ...], output: object) -> object:
        return _replace(output, torch.zeros_like(_extract(output)))

    handle = module.register_forward_hook(zero)
    try:
        return _run_raw_forward(model, inputs)
    finally:
        handle.remove()


def _run_lens_forward(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return cast(torch.Tensor, model(input_ids))


def _run_lens_capture(
    model: Any,
    input_ids: torch.Tensor,
    *,
    hook_name: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda name: name == hook_name,
            return_cache_object=False,
        )
    return {
        "logits": cast(torch.Tensor, logits),
        "activation": cast(torch.Tensor, cache[hook_name]),
    }


def _run_lens_capture_only(
    model: Any,
    input_ids: torch.Tensor,
    *,
    hook_name: str,
    stop_at_layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        _residual, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda name: name == hook_name,
            return_cache_object=False,
            stop_at_layer=stop_at_layer,
        )
    return cast(torch.Tensor, cache[hook_name])


def _run_lens_zero(
    model: Any,
    input_ids: torch.Tensor,
    *,
    hook_name: str,
) -> torch.Tensor:
    with torch.no_grad():
        return cast(
            torch.Tensor,
            model.run_with_hooks(
                input_ids,
                fwd_hooks=[(hook_name, lambda activation, hook: torch.zeros_like(activation))],
            ),
        )


def _run_nnterp_forward(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    with model.trace(**inputs) as tracer:
        logits = model.logits.save()
        tracer.stop()
    return _saved_tensor(logits)


def _run_nnterp_capture(model: Any, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    with model.trace(**inputs) as tracer:
        activation = model.layers_output[0].save()
        logits = model.logits.save()
        tracer.stop()
    return {"logits": _saved_tensor(logits), "activation": _saved_tensor(activation)}


def _run_nnterp_capture_only(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    with model.trace(**inputs) as tracer:
        activation = model.layers_output[0].save()
        tracer.stop()
    return _saved_tensor(activation)


def _run_nnterp_zero(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    with model.trace(**inputs) as tracer:
        model.layers_output[0] = torch.zeros_like(model.layers_output[0])
        logits = model.logits.save()
        tracer.stop()
    return _saved_tensor(logits)


def _saved_tensor(value: Any) -> torch.Tensor:
    actual = getattr(value, "value", value)
    if not isinstance(actual, torch.Tensor):
        raise TypeError(f"Expected a tensor, got {type(actual).__name__}.")
    return actual


def _compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return compare_tensors(left, right, mode="cross_impl")


def _tag_case(case: dict[str, Any], *, library: str, operation: str) -> dict[str, Any]:
    case["library"] = library
    case["operation"] = operation
    return case


def _annotate_against_raw(cases: list[dict[str, Any]], *, raw_name: str) -> None:
    by_name = {case["name"]: case for case in cases}
    raw_case = by_name[raw_name]
    raw_median_ms = raw_case.get("median_ms")
    if not isinstance(raw_median_ms, (int, float)) or raw_median_ms <= 0:
        return
    for case in cases:
        if case["name"] == raw_name or case.get("skipped"):
            continue
        median_ms = case.get("median_ms")
        if not isinstance(median_ms, (int, float)) or median_ms <= 0:
            continue
        case["overhead_vs_raw_pct"] = ((median_ms / raw_median_ms) - 1.0) * 100.0


def _library_versions() -> dict[str, str]:
    names = ("transformers", "transformer-lens", "nnterp", "nnsight")
    return {name: _package_version(name) for name in names}


def _run_transformerlens_compare(
    config: CompareConfig,
    *,
    tokenizer: Any,
    inputs: dict[str, Any],
    lens_input_ids: torch.Tensor,
    raw_logits: torch.Tensor,
    manual_capture: torch.Tensor,
    manual_zero_logits: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support = local_support_map(config.model_family)
    runnable, reason = runnable_support(support, "transformerlens")
    if not runnable:
        return _skipped_library("transformerlens", reason)
    try:
        tl_source = _load_hf_model(config.model_name, device=device, dtype=dtype)
        lens_model = _load_transformerlens_model(
            config.model_name,
            hf_model=tl_source,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
        del tl_source
        lens_forward = _run_lens_forward(lens_model, lens_input_ids)
        lens_capture = _run_lens_capture(
            lens_model,
            lens_input_ids,
            hook_name=config.lens_hook_name,
        )
        lens_capture_only = _run_lens_capture_only(
            lens_model,
            lens_input_ids,
            hook_name=config.lens_hook_name,
            stop_at_layer=config.lens_stop_at_layer,
        )
        lens_zero = _run_lens_zero(lens_model, lens_input_ids, hook_name=config.lens_hook_name)
        cases = [
            _tag_case(
                _measure_case(
                    "transformerlens_forward",
                    lambda model=lens_model, ids=lens_input_ids: _run_lens_forward(model, ids),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="transformerlens",
                operation="forward",
            ),
            _tag_case(
                _measure_case(
                    "transformerlens_get_one",
                    lambda model=lens_model, ids=lens_input_ids: _run_lens_capture(
                        model,
                        ids,
                        hook_name=config.lens_hook_name,
                    ),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="transformerlens",
                operation="get_one",
            ),
            _tag_case(
                _measure_case(
                    "transformerlens_capture_only",
                    lambda model=lens_model, ids=lens_input_ids: _run_lens_capture_only(
                        model,
                        ids,
                        hook_name=config.lens_hook_name,
                        stop_at_layer=config.lens_stop_at_layer,
                    ),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="transformerlens",
                operation="capture_only",
            ),
            _tag_case(
                _measure_case(
                    "transformerlens_map_one",
                    lambda model=lens_model, ids=lens_input_ids: _run_lens_zero(
                        model,
                        ids,
                        hook_name=config.lens_hook_name,
                    ),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="transformerlens",
                operation="map_one",
            ),
        ]
        correctness = {
            "transformerlens_forward": _compare_tensors(raw_logits, lens_forward),
            "transformerlens_get_one_logits": _compare_tensors(raw_logits, lens_capture["logits"]),
            "transformerlens_get_one_activation": _compare_tensors(
                manual_capture,
                lens_capture["activation"],
            ),
            "transformerlens_capture_only_activation": _compare_tensors(
                manual_capture,
                lens_capture_only,
            ),
            "transformerlens_map_one": _compare_tensors(manual_zero_logits, lens_zero),
        }
        del lens_model
        return cases, correctness
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _skipped_library("transformerlens", reason)


def _run_nnterp_compare(
    config: CompareConfig,
    *,
    inputs: dict[str, Any],
    raw_logits: torch.Tensor,
    manual_capture: torch.Tensor,
    manual_zero_logits: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support = local_support_map(config.model_family)
    runnable, reason = runnable_support(support, "nnterp")
    if not runnable:
        return _skipped_library("nnterp", reason)
    try:
        nn_model = _load_nnterp_model(config.model_name, device=device, dtype=dtype)
        nn_forward = _run_nnterp_forward(nn_model, inputs)
        nn_capture = _run_nnterp_capture(nn_model, inputs)
        nn_capture_only = _run_nnterp_capture_only(nn_model, inputs)
        nn_zero = _run_nnterp_zero(nn_model, inputs)
        cases = [
            _tag_case(
                _measure_case(
                    "nnterp_forward",
                    lambda model=nn_model: _run_nnterp_forward(model, inputs),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="nnterp",
                operation="forward",
            ),
            _tag_case(
                _measure_case(
                    "nnterp_get_one",
                    lambda model=nn_model: _run_nnterp_capture(model, inputs),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="nnterp",
                operation="get_one",
            ),
            _tag_case(
                _measure_case(
                    "nnterp_capture_only",
                    lambda model=nn_model: _run_nnterp_capture_only(model, inputs),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="nnterp",
                operation="capture_only",
            ),
            _tag_case(
                _measure_case(
                    "nnterp_map_one",
                    lambda model=nn_model: _run_nnterp_zero(model, inputs),
                    warmup=config.warmup,
                    trials=config.trials,
                    device=device,
                ),
                library="nnterp",
                operation="map_one",
            ),
        ]
        correctness = {
            "nnterp_forward": _compare_tensors(raw_logits, nn_forward),
            "nnterp_get_one_logits": _compare_tensors(raw_logits, nn_capture["logits"]),
            "nnterp_get_one_activation": _compare_tensors(
                manual_capture,
                nn_capture["activation"],
            ),
            "nnterp_capture_only_activation": _compare_tensors(manual_capture, nn_capture_only),
            "nnterp_map_one": _compare_tensors(manual_zero_logits, nn_zero),
        }
        del nn_model
        return cases, correctness
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _skipped_library("nnterp", reason)


def _skipped_library(library: str, reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    operations = ("forward", "get_one", "capture_only", "map_one")
    cases = [
        {
            "name": f"{library}_{operation}",
            "library": library,
            "operation": operation,
            "skipped": reason,
        }
        for operation in operations
    ]
    correctness = {
        f"{library}_forward": {"skipped": True, "reason": reason},
        f"{library}_get_one_logits": {"skipped": True, "reason": reason},
        f"{library}_get_one_activation": {"skipped": True, "reason": reason},
        f"{library}_capture_only_activation": {"skipped": True, "reason": reason},
        f"{library}_map_one": {"skipped": True, "reason": reason},
    }
    return cases, correctness


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


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
