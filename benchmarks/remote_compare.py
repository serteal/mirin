"""User-visible local/remote comparison harness for tinyinterp."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

import tinyinterp as ti

from .model_api import _environment_report, _measure_case, _resolve_device, _resolve_dtype
from .runtime_internals import (
    _clear_cuda,
    _hf_generate,
    _load_model,
    _make_dataset,
    _manual_hook_once,
    _model_collect_loop,
    _open_remote_model,
    _pad_batch_sequences,
    _pad_token_id,
    _requests_from_batch,
    _resolve_proxy,
    _site_path,
    _vocab_size,
)
from .support import remote_support_map
from .tolerances import compare_tensors


@dataclass(slots=True)
class RemoteCompareConfig:
    """Configuration for the user-visible local/remote comparison matrix."""

    model_name: str
    model_family: str = "custom"
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 7
    batch_size: int = 4
    seq_len: int = 128
    max_new_tokens: int = 16
    warmup: int = 1
    trials: int = 5
    json_output: str | None = None


def run_remote_compare_benchmarks(config: RemoteCompareConfig) -> dict[str, Any]:
    """Benchmark the same user-visible collect/generate calls across local and remote paths."""

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)
    support = remote_support_map(config.model_family)

    hf_model = _load_model(_load_cfg(config.model_name), device=device, dtype=dtype)
    environment = _environment_report(
        hf_model,
        model_name=config.model_name,
        device=device,
        dtype=dtype,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )
    site_path = _site_path(config.model_name, hf_model)
    batch = _make_dataset(
        batch_size=config.batch_size,
        batches=1,
        seq_len=config.seq_len,
        vocab_size=_vocab_size(hf_model),
        device=device,
    )[0]
    requests = _requests_from_batch(batch)
    prompt_sequences = _hf_generate(hf_model, batch, max_new_tokens=config.max_new_tokens)
    manual_capture = _manual_hook_once(hf_model, site_path, batch)
    hf_collect_case = _measure_case(
        "hf_collect_manual",
        lambda model=hf_model, path=site_path, row=batch: _manual_hook_once(model, path, row),
        warmup=config.warmup,
        trials=config.trials,
        device=device,
    )
    hf_generate_case = _measure_case(
        "hf_generate_batched",
        lambda model=hf_model, row=batch: _hf_generate(
            model,
            row,
            max_new_tokens=config.max_new_tokens,
        ),
        warmup=config.warmup,
        trials=config.trials,
        device=device,
    )
    del hf_model
    _clear_cuda(device)

    local_model = _load_model(_load_cfg(config.model_name), device=device, dtype=dtype)
    local_ti = ti.Model(local_model)
    local_proxy = _resolve_proxy(local_ti, site_path)
    local_collect = local_ti.collect(requests, get=[local_proxy])
    local_collect_tensor = torch.cat(
        [cast(torch.Tensor, output[local_proxy]).detach().cpu() for output in local_collect],
        dim=0,
    )
    local_generate = _pad_sequences(
        local_ti.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        ),
        pad_token_id=_pad_token_id(local_model),
    )
    local_generate_get_all = _as_generate_output_list(
        local_ti.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            get=[local_proxy],
            capture="all",
        )
    )
    local_generate_get_generated = _as_generate_output_list(
        local_ti.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            get=[local_proxy],
            capture="generated",
        )
    )
    local_generate_get_all_sequences = _pad_generate_sequences(
        local_generate_get_all,
        pad_token_id=_pad_token_id(local_model),
    )
    local_generate_get_generated_sequences = _pad_generate_sequences(
        local_generate_get_generated,
        pad_token_id=_pad_token_id(local_model),
    )
    local_generate_get_all_activations = _pad_generate_activations(
        local_generate_get_all,
        local_proxy,
    )
    local_generate_get_generated_activations = _pad_generate_activations(
        local_generate_get_generated,
        local_proxy,
    )
    local_cases = [
        _measure_case(
            "tinyinterp_collect_local",
            lambda model=local_ti, proxy=local_proxy, row=batch: _model_collect_loop(
                model,
                proxy,
                [row],
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_local",
            lambda model=local_ti, reqs=requests: model.generate(
                reqs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_get_all_local",
            lambda model=local_ti, reqs=requests, proxy=local_proxy: model.generate(
                reqs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                get=[proxy],
                capture="all",
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_get_generated_local",
            lambda model=local_ti, reqs=requests, proxy=local_proxy: model.generate(
                reqs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                get=[proxy],
                capture="generated",
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
    ]
    del local_ti, local_model
    _clear_cuda(device)

    runtime_model = _load_model(_load_cfg(config.model_name), device=device, dtype=dtype)
    runtime = ti.Server(runtime_model)
    remote_client, remote_proxy = _open_tinyinterp_remote(runtime, site_path)
    remote_collect = remote_client.collect(requests, get=[remote_proxy])
    remote_collect_tensor = torch.cat(
        [cast(torch.Tensor, output[remote_proxy]).detach().cpu() for output in remote_collect],
        dim=0,
    )
    remote_generate = _pad_sequences(
        remote_client.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        ),
        pad_token_id=_pad_token_id(runtime_model),
    )
    remote_generate_get_all = _as_generate_output_list(
        remote_client.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            get=[remote_proxy],
            capture="all",
        )
    )
    remote_generate_get_generated = _as_generate_output_list(
        remote_client.generate(
            requests,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            get=[remote_proxy],
            capture="generated",
        )
    )
    remote_generate_get_all_sequences = _pad_generate_sequences(
        remote_generate_get_all,
        pad_token_id=_pad_token_id(runtime_model),
    )
    remote_generate_get_generated_sequences = _pad_generate_sequences(
        remote_generate_get_generated,
        pad_token_id=_pad_token_id(runtime_model),
    )
    remote_generate_get_all_activations = _pad_generate_activations(
        remote_generate_get_all,
        remote_proxy,
    )
    remote_generate_get_generated_activations = _pad_generate_activations(
        remote_generate_get_generated,
        remote_proxy,
    )
    remote_cases = [
        _measure_case(
            "tinyinterp_collect_remote",
            lambda: _model_collect_loop(remote_client, remote_proxy, [batch]),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_remote",
            lambda: remote_client.generate(
                requests,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_get_all_remote",
            lambda: remote_client.generate(
                requests,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                get=[remote_proxy],
                capture="all",
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
        _measure_case(
            "tinyinterp_generate_get_generated_remote",
            lambda: remote_client.generate(
                requests,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                get=[remote_proxy],
                capture="generated",
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        ),
    ]

    correctness = {
        "collect_local_vs_hf": _compare_tensors(local_collect_tensor, manual_capture),
        "collect_remote_vs_hf": _compare_tensors(remote_collect_tensor, manual_capture),
        "generate_local_vs_hf": {"ok": torch.equal(local_generate.cpu(), prompt_sequences.cpu())},
        "generate_remote_vs_hf": {"ok": torch.equal(remote_generate.cpu(), prompt_sequences.cpu())},
        "generate_get_all_sequences_remote_vs_local": {
            "ok": torch.equal(
                remote_generate_get_all_sequences.cpu(),
                local_generate_get_all_sequences.cpu(),
            )
        },
        "generate_get_all_activations_remote_vs_local": _compare_tensors(
            remote_generate_get_all_activations,
            local_generate_get_all_activations,
        ),
        "generate_get_generated_sequences_remote_vs_local": {
            "ok": torch.equal(
                remote_generate_get_generated_sequences.cpu(),
                local_generate_get_generated_sequences.cpu(),
            )
        },
        "generate_get_generated_activations_remote_vs_local": _compare_tensors(
            remote_generate_get_generated_activations,
            local_generate_get_generated_activations,
        ),
    }

    cases = [
        hf_collect_case,
        hf_generate_case,
        *local_cases,
        *remote_cases,
    ]
    _annotate_cases(
        cases,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        max_new_tokens=config.max_new_tokens,
    )
    performance = _performance_checks(cases)

    report = {
        "config": asdict(config),
        "environment": {
            **environment,
            "site_path": site_path,
        },
        "support": support,
        "correctness": correctness,
        "performance": performance,
        "cases": cases,
    }
    if config.json_output is not None:
        path = Path(config.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        remote_client.close()
    finally:
        runtime.close()
        del runtime_model
        _clear_cuda(device)
    return report


def format_remote_compare_report(report: dict[str, Any]) -> str:
    """Render the user-visible local/remote comparison report."""

    env = report["environment"]
    lines = [
        f"Model: {env['model_name']}",
        f"Device: {env['device']} ({env['gpu_name']})",
        f"Dtype: {env['dtype']}",
        f"Site: {env['site_path']}",
        "",
        "Support:",
    ]
    for entry in report["support"].values():
        state = "ok" if entry["runnable"] else "skip"
        reason = f" ({entry['reason']})" if entry["reason"] else ""
        lines.append(f"- {entry['name']}: {state}{reason}")

    lines.extend(["", "Correctness:"])
    for name, check in report["correctness"].items():
        if check.get("skipped"):
            lines.append(f"- {name}: skipped ({check['reason']})")
            continue
        status = "ok" if check["ok"] else "FAIL"
        diff = check.get("max_abs_diff")
        detail = f" (max_abs_diff={diff:.6f})" if diff is not None else ""
        lines.append(f"- {name}: {status}{detail}")

    lines.extend(["", "Performance:"])
    for name, check in report.get("performance", {}).items():
        status = "ok" if check["ok"] else "SLOWER"
        lines.append(
            f"- {name}: {status} "
            f"(baseline={check['baseline_ms']:.3f}ms, candidate={check['candidate_ms']:.3f}ms, "
            f"delta={check['delta_pct']:+.2f}%)"
        )

    lines.extend(["", "Timing:"])
    for case in report["cases"]:
        if case.get("skipped"):
            lines.append(f"- {case['name']}: skipped ({case['skipped']})")
            continue
        extras = []
        if case.get("examples_per_second") is not None:
            extras.append(f"{case['examples_per_second']:.1f} ex/s")
        if case.get("tokens_per_second") is not None:
            extras.append(f"{case['tokens_per_second']:.1f} tok/s")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {case['name']}: {case['median_ms']:.3f}ms{suffix}")
    return "\n".join(lines)


def _annotate_cases(
    cases: list[dict[str, Any]],
    *,
    batch_size: int,
    seq_len: int,
    max_new_tokens: int,
) -> None:
    for case in cases:
        if case.get("skipped"):
            continue
        name = case["name"]
        if "collect" in name:
            case["examples_per_second"] = batch_size / (case["median_ms"] / 1000.0)
            case["tokens_per_second"] = (batch_size * seq_len) / (case["median_ms"] / 1000.0)
            continue
        case["examples_per_second"] = batch_size / (case["median_ms"] / 1000.0)
        case["tokens_per_second"] = (batch_size * max_new_tokens) / (case["median_ms"] / 1000.0)


def _performance_checks(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name = {case["name"]: case for case in cases}
    return {
        "collect_remote_vs_local": _performance_check(
            baseline=by_name["tinyinterp_collect_local"],
            candidate=by_name["tinyinterp_collect_remote"],
        ),
        "generate_remote_vs_local": _performance_check(
            baseline=by_name["tinyinterp_generate_local"],
            candidate=by_name["tinyinterp_generate_remote"],
        ),
        "generate_get_all_remote_vs_local": _performance_check(
            baseline=by_name["tinyinterp_generate_get_all_local"],
            candidate=by_name["tinyinterp_generate_get_all_remote"],
        ),
        "generate_get_generated_remote_vs_local": _performance_check(
            baseline=by_name["tinyinterp_generate_get_generated_local"],
            candidate=by_name["tinyinterp_generate_get_generated_remote"],
        ),
    }


def _performance_check(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_ms = float(baseline["median_ms"])
    candidate_ms = float(candidate["median_ms"])
    return {
        "ok": candidate_ms <= baseline_ms,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "delta_pct": ((candidate_ms / baseline_ms) - 1.0) * 100.0,
    }


def _open_tinyinterp_remote(
    server: ti.Server,
    site_path: str,
) -> tuple[Any, Any]:
    sock_path = f"/tmp/tinyinterp-server-compare-{uuid.uuid4().hex}.sock"
    thread = threading.Thread(target=server.serve, args=(sock_path,), daemon=True)
    thread.start()
    remote_client = _open_remote_model(sock_path)
    remote_proxy = _resolve_proxy(remote_client, site_path)
    return remote_client, remote_proxy


def _load_cfg(model_name: str) -> Any:
    return SimpleNamespace(model_name=model_name)


def _compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return compare_tensors(left, right, mode="same_impl")


def _pad_sequences(value: Any, *, pad_token_id: int) -> torch.Tensor:
    if isinstance(value, ti.GenerateOutput):
        return cast(torch.Tensor, value.sequences)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, ti.GenerateOutput) for item in value)
    ):
        return _pad_generate_sequences(
            cast(list[ti.GenerateOutput], value),
            pad_token_id=pad_token_id,
        )
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, list) or not value:
        raise TypeError(
            f"Expected GenerateOutput or list[GenerateOutput], got {type(value).__name__}."
        )
    if not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(
            f"Expected GenerateOutput or list[GenerateOutput], got {type(value).__name__}."
        )
    return _pad_batch_sequences(cast(list[torch.Tensor], value), pad_token_id=pad_token_id)


def _pad_generate_sequences(outputs: list[ti.GenerateOutput], *, pad_token_id: int) -> torch.Tensor:
    return _pad_batch_sequences(
        [cast(torch.Tensor, output.sequences) for output in outputs],
        pad_token_id=pad_token_id,
    )


def _pad_generate_activations(outputs: list[ti.GenerateOutput], proxy: Any) -> torch.Tensor:
    values = [cast(torch.Tensor, output[proxy]).detach().cpu() for output in outputs]
    max_tokens = max(int(value.shape[1]) for value in values)
    padded: list[torch.Tensor] = []
    for value in values:
        if int(value.shape[1]) == max_tokens:
            padded.append(value)
            continue
        pad_shape = (value.shape[0], max_tokens - value.shape[1], *value.shape[2:])
        pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        padded.append(torch.cat([value, pad], dim=1))
    return torch.cat(padded, dim=0)


def _as_generate_output_list(value: Any) -> list[ti.GenerateOutput]:
    if isinstance(value, ti.GenerateOutput):
        return [value]
    if isinstance(value, list) and all(isinstance(item, ti.GenerateOutput) for item in value):
        return cast(list[ti.GenerateOutput], value)
    raise TypeError(f"Expected GenerateOutput or list[GenerateOutput], got {type(value).__name__}.")
