"""Server runtime benchmark harness for the inference server."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn

import tinyinterp as ti
from tinyinterp.hooks import _extract
from tinyinterp.server.runtime import (
    contains_eos,
    eos_token_ids,
    extract_last_token_logits,
    filter_supported_kwargs,
)

from .model_api import (
    _config_value,
    _environment_report,
    _measure_case,
    _resolve_device,
    _resolve_dtype,
)

DEFAULT_MODEL_NAMES = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3.5-4B",
)


@dataclass(slots=True)
class RuntimeInternalsBenchmarkConfig:
    """Configuration for the runtime-internals benchmarks."""

    model_name: str
    device: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 7
    seq_len: int = 128
    dataset_batch_size: int = 4
    dataset_batches: int = 8
    generate_batch_size: int = 4
    max_new_tokens: int = 16
    warmup: int = 1
    trials: int = 5
    json_output: str | None = None


def run_runtime_internals_benchmarks(config: RuntimeInternalsBenchmarkConfig) -> dict[str, Any]:
    """Run the runtime-internals benchmark matrix for one model."""

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)

    hf_model = _load_model(config, device=device, dtype=dtype)
    environment = _environment_report(
        hf_model,
        model_name=config.model_name,
        device=device,
        dtype=dtype,
        batch_size=config.dataset_batch_size,
        seq_len=config.seq_len,
    )
    site_path = _site_path(config.model_name, hf_model)
    dataset = _make_dataset(
        batch_size=config.dataset_batch_size,
        batches=config.dataset_batches,
        seq_len=config.seq_len,
        vocab_size=_vocab_size(hf_model),
        device=device,
    )
    prompts = _make_dataset(
        batch_size=config.generate_batch_size,
        batches=1,
        seq_len=config.seq_len,
        vocab_size=_vocab_size(hf_model),
        device=device,
    )[0]
    prompt_requests = _requests_from_batch(prompts)
    single_prompt = {
        "input_ids": prompts["input_ids"][:1],
        "attention_mask": prompts["attention_mask"][:1],
    }

    correctness = _correctness_checks(
        config,
        device=device,
        dtype=dtype,
        site_path=site_path,
        batch=dataset[0],
        prompts=prompts,
        single_prompt=single_prompt,
    )

    cases: list[dict[str, Any]] = []

    manual_runner = _ManualHookLoop(hf_model, site_path)
    try:
        cases.append(
            _measure_case(
                "hf_hook_loop",
                lambda runner=manual_runner, rows=dataset: runner.run(rows),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
            )
        )
        cases.append(
            _measure_case(
                "hf_generate_single",
                lambda model=hf_model, row=single_prompt: _hf_generate(
                    model,
                    row,
                    max_new_tokens=config.max_new_tokens,
                ),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
            )
        )
        cases.append(
            _measure_case(
                "hf_generate_multi_sequential",
                lambda model=hf_model, rows=prompts: _hf_generate_sequential(
                    model,
                    rows,
                    max_new_tokens=config.max_new_tokens,
                ),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
            )
        )
        cases.append(
            _measure_case(
                "hf_generate_multi_batched",
                lambda model=hf_model, rows=prompts: _hf_generate(
                    model,
                    rows,
                    max_new_tokens=config.max_new_tokens,
                ),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
            )
        )
    finally:
        manual_runner.close()
        del manual_runner, hf_model
        _clear_cuda(device)

    local_model = _load_model(config, device=device, dtype=dtype)
    local_ti = ti.Model(local_model)
    local_proxy = _resolve_proxy(local_ti, site_path)
    cases.append(
        _measure_case(
            "tinyinterp_capture_loop",
            lambda model=local_ti, proxy=local_proxy, rows=dataset: _tinyinterp_capture_loop(
                model,
                proxy,
                rows,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "model_collect_local",
            lambda model=local_ti, proxy=local_proxy, rows=dataset: _model_collect_loop(
                model,
                proxy,
                rows,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    _clear_cuda(device)

    server_model = _load_model(config, device=device, dtype=dtype)
    server = ti.Server(server_model)
    collector_plan = server.compile(
        get=[site_path],
        output={"logits": False, "activations": True},
    )
    collector = server.open_collector(plan=collector_plan, stop_at_last_get=True)
    empty_plan = server.compile(output={"logits": True, "activations": False})
    inspect_plan = server.compile(get=[site_path], output={"logits": False, "activations": True})
    steer_plan = server.compile(
        mapping={site_path: ti.zero()}, output={"logits": True, "activations": False}
    )
    static_supported = _supports_static_cache(server_model)
    cases.append(
        _measure_case(
            "server_collector",
            lambda: _server_collector_loop(collector, dataset, site_path),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_call_concurrent_stateless",
            lambda: _server_call_concurrent(
                server,
                empty_plan,
                _requests_from_batch(dataset[0])[:2],
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_single",
            lambda: server.generate(
                input_ids=single_prompt["input_ids"],
                attention_mask=single_prompt["attention_mask"],
                plan=empty_plan,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_batched_plain",
            lambda: server.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                plan=empty_plan,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_many_plain",
            lambda: server.generate_many(
                prompt_requests,
                plan=empty_plan,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_multi_session",
            lambda: _server_multi_session(
                server,
                empty_plan,
                prompts,
                max_new_tokens=config.max_new_tokens,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_batched_map",
            lambda: server.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                plan=steer_plan,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_generate_many_map",
            lambda: server.generate_many(
                prompt_requests,
                plan=steer_plan,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "tinyinterp_generate_map_batched",
            lambda model=local_ti, rows=prompts: _tinyinterp_generate_batched(
                model,
                rows,
                max_new_tokens=config.max_new_tokens,
                mapping={local_proxy: ti.zero()},
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "tinyinterp_decode_get_batched",
            lambda model=local_ti, rows=prompts: _tinyinterp_decode_batched(
                model,
                rows,
                max_new_tokens=config.max_new_tokens,
                proxy=local_proxy,
            )[1],
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_decode_get_multi_session",
            lambda: _server_multi_session_stepwise(
                server,
                inspect_plan,
                prompts,
                max_new_tokens=config.max_new_tokens,
                site_path=site_path,
            )[1],
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "tinyinterp_decode_map_batched",
            lambda model=local_ti, rows=prompts: _tinyinterp_decode_batched(
                model,
                rows,
                max_new_tokens=config.max_new_tokens,
                mapping={local_proxy: ti.zero()},
            )[1],
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "server_decode_map_multi_session",
            lambda: _server_multi_session_stepwise(
                server,
                steer_plan,
                prompts,
                max_new_tokens=config.max_new_tokens,
                site_path=None,
            )[1],
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    if static_supported:
        cases.append(
            _measure_case(
                "server_generate_multi_session_static",
                lambda: _server_multi_session(
                    server,
                    empty_plan,
                    prompts,
                    max_new_tokens=config.max_new_tokens,
                    cache="static",
                ),
                warmup=config.warmup,
                trials=config.trials,
                device=device,
                use_counters=True,
            )
        )
    else:
        cases.append(
            {
                "name": "server_generate_multi_session_static",
                "skipped": "unsupported_static_cache",
            }
        )
    remote_sock = f"/tmp/tinyinterp-bench-{uuid.uuid4().hex}.sock"
    remote_thread = threading.Thread(target=server.serve, args=(remote_sock,), daemon=True)
    remote_thread.start()
    remote_client = _open_remote_model(remote_sock)
    remote_proxy = _resolve_proxy(remote_client, site_path)
    cases.append(
        _measure_case(
            "model_collect_remote",
            lambda: _model_collect_loop(remote_client, remote_proxy, dataset),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    cases.append(
        _measure_case(
            "model_generate_remote",
            lambda: remote_client.generate(
                prompt_requests,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            ),
            warmup=config.warmup,
            trials=config.trials,
            device=device,
            use_counters=True,
        )
    )
    remote_client.close()
    server.close()
    del local_ti, local_model
    del server_model
    if os.path.exists(remote_sock):
        os.unlink(remote_sock)
    server_stats = server.stats()

    _annotate_server_metrics(cases, config=config)
    report = {
        "config": asdict(config),
        "environment": {
            **environment,
            "site_path": site_path,
        },
        "correctness": correctness,
        "cases": cases,
        "server_stats": server_stats,
    }
    if config.json_output is not None:
        path = Path(config.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    return report


def run_runtime_internals_suite(
    configs: list[RuntimeInternalsBenchmarkConfig],
    *,
    json_output: str | None = None,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for config in configs:
        try:
            reports.append(run_runtime_internals_benchmarks(config))
        except Exception as exc:
            reports.append(
                {
                    "config": asdict(config),
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
            )
    suite = {"reports": reports}
    if json_output is not None:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(suite, indent=2))
    return suite


def format_runtime_internals_report(report: Mapping[str, Any]) -> str:
    env = cast(dict[str, Any], report["environment"])
    lines = [
        f"Model: {env['model_name']} ({env['architecture']})",
        f"Device: {env['device']} {env['dtype']}",
        f"Site: {env['site_path']}",
        "Correctness:",
    ]
    for name, check in cast(dict[str, Any], report["correctness"]).items():
        if check.get("skipped"):
            lines.append(f"  - {name}: skipped ({check['reason']})")
            continue
        status = "ok" if check["ok"] else "FAIL"
        detail = f" max_abs_diff={check['max_abs_diff']:.6g}" if "max_abs_diff" in check else ""
        lines.append(f"  - {name}: {status}{detail}")
    lines.append("Cases:")
    for case in cast(list[dict[str, Any]], report["cases"]):
        if case.get("skipped"):
            lines.append(f"  - {case['name']}: skipped ({case['skipped']})")
            continue
        extras = []
        if case.get("tokens_per_sec") is not None:
            extras.append(f"{case['tokens_per_sec']:.1f} tok/s")
        if case.get("examples_per_sec") is not None:
            extras.append(f"{case['examples_per_sec']:.1f} ex/s")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"  - {case['name']}: {case['median_ms']:.3f}ms{suffix}")
    return "\n".join(lines)


def format_runtime_internals_suite(suite: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    for report in cast(list[dict[str, Any]], suite["reports"]):
        if "load_error" in report:
            config = cast(dict[str, Any], report["config"])
            blocks.append(f"Model: {config['model_name']}\n  load_error: {report['load_error']}")
            continue
        blocks.append(format_runtime_internals_report(report))
    return "\n\n".join(blocks)


def _correctness_checks(
    config: RuntimeInternalsBenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    site_path: str,
    batch: Mapping[str, torch.Tensor],
    prompts: Mapping[str, torch.Tensor],
    single_prompt: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    hf_model = _load_model(config, device=device, dtype=dtype)
    manual = _manual_hook_once(hf_model, site_path, batch)
    del hf_model
    _clear_cuda(device)

    local_model = _load_model(config, device=device, dtype=dtype)
    local_ti = ti.Model(local_model)
    local_proxy = _resolve_proxy(local_ti, site_path)
    local_output = local_ti(**batch, get=[local_proxy], use_cache=False, stop_at_last_get=True)
    local_act = cast(torch.Tensor, local_output[local_proxy]).cpu()
    checks["collector_local_vs_hf"] = {
        "ok": torch.allclose(local_act, manual, atol=1e-4, rtol=1e-4),
        "max_abs_diff": float((local_act - manual).abs().max().item()),
    }
    del local_ti, local_model
    _clear_cuda(device)

    server_model = _load_model(config, device=device, dtype=dtype)
    server = ti.Server(server_model)
    plan = server.compile(get=[site_path], output={"logits": False, "activations": True})
    collector = server.open_collector(plan=plan, stop_at_last_get=True)
    server_result = collector.collect_batch(batch)
    server_act = cast(torch.Tensor, server_result.activations[site_path]).cpu()
    checks["collector_server_vs_hf"] = {
        "ok": torch.allclose(server_act, manual, atol=1e-4, rtol=1e-4),
        "max_abs_diff": float((server_act - manual).abs().max().item()),
    }
    remote_sock = f"/tmp/tinyinterp-correctness-{uuid.uuid4().hex}.sock"
    remote_thread = threading.Thread(target=server.serve, args=(remote_sock,), daemon=True)
    remote_thread.start()
    remote_client = _open_remote_model(remote_sock)
    remote_proxy = _resolve_proxy(remote_client, site_path)
    api_collect = remote_client.collect(_requests_from_batch(batch), get=[remote_proxy])
    api_collect_act = torch.cat(
        [cast(torch.Tensor, output[remote_proxy]).cpu() for output in api_collect],
        dim=0,
    )
    checks["collect_api_remote_vs_hf"] = {
        "ok": torch.allclose(api_collect_act, manual, atol=1e-4, rtol=1e-4),
        "max_abs_diff": float((api_collect_act - manual).abs().max().item()),
    }
    empty_plan = server.compile(output={"logits": True, "activations": False})
    steer_plan = server.compile(
        mapping={site_path: ti.zero()}, output={"logits": True, "activations": False}
    )
    server_single = server.generate(
        input_ids=single_prompt["input_ids"],
        attention_mask=single_prompt["attention_mask"],
        plan=empty_plan,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
    )
    server_batched = server.generate(
        input_ids=prompts["input_ids"],
        attention_mask=prompts["attention_mask"],
        plan=empty_plan,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
    )
    server_many_plain = _pad_batch_sequences(
        server.generate_many(
            _requests_from_batch(prompts),
            plan=empty_plan,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        ),
        pad_token_id=_pad_token_id(server._model.wrapped),
    )
    server_many_map = _pad_batch_sequences(
        server.generate_many(
            _requests_from_batch(prompts),
            plan=steer_plan,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        ),
        pad_token_id=_pad_token_id(server._model.wrapped),
    )
    model_generate_remote = _normalize_generated_batch(
        remote_client.generate(
            _requests_from_batch(prompts),
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        ),
        pad_token_id=_pad_token_id(server._model.wrapped),
    )
    server_multi = _server_multi_session(
        server,
        empty_plan,
        prompts,
        max_new_tokens=config.max_new_tokens,
    )
    remote_client.close()
    server.close()
    if os.path.exists(remote_sock):
        os.unlink(remote_sock)
    del server
    _clear_cuda(device)

    hf_generate_model = _load_model(config, device=device, dtype=dtype)
    hf_single = _hf_generate(hf_generate_model, single_prompt, max_new_tokens=config.max_new_tokens)
    hf_batched = _hf_generate(hf_generate_model, prompts, max_new_tokens=config.max_new_tokens)
    hf_multi = _hf_generate_sequential(
        hf_generate_model,
        prompts,
        max_new_tokens=config.max_new_tokens,
    )
    checks["generate_single"] = {
        "ok": torch.equal(server_single, hf_single),
    }
    checks["generate_batched_plain"] = {
        "ok": torch.equal(server_batched, hf_batched),
    }
    checks["generate_many_plain"] = {
        "ok": torch.equal(server_many_plain, hf_batched),
    }
    checks["model_generate_remote"] = {
        "ok": torch.equal(model_generate_remote.to(hf_batched.device), hf_batched),
    }
    checks["generate_multi"] = {
        "ok": torch.equal(server_multi, hf_multi),
    }
    del hf_generate_model
    _clear_cuda(device)

    inspect_local_model = _load_model(config, device=device, dtype=dtype)
    inspect_local_ti = ti.Model(inspect_local_model)
    inspect_proxy = _resolve_proxy(inspect_local_ti, site_path)
    local_inspect_tokens, local_inspect_total = _tinyinterp_decode_batched(
        inspect_local_ti,
        prompts,
        max_new_tokens=config.max_new_tokens,
        proxy=inspect_proxy,
    )
    local_steer_generate = _tinyinterp_generate_batched(
        inspect_local_ti,
        prompts,
        max_new_tokens=config.max_new_tokens,
        mapping={inspect_proxy: ti.zero()},
    )
    _local_steer_tokens, _ = _tinyinterp_decode_batched(
        inspect_local_ti,
        prompts,
        max_new_tokens=config.max_new_tokens,
        mapping={inspect_proxy: ti.zero()},
    )
    del inspect_local_ti, inspect_local_model
    _clear_cuda(device)

    inspect_server_model = _load_model(config, device=device, dtype=dtype)
    inspect_server = ti.Server(inspect_server_model)
    inspect_plan = inspect_server.compile(
        get=[site_path], output={"logits": False, "activations": True}
    )
    steer_plan = inspect_server.compile(
        mapping={site_path: ti.zero()}, output={"logits": True, "activations": False}
    )
    server_steer_batched = inspect_server.generate(
        input_ids=prompts["input_ids"],
        attention_mask=prompts["attention_mask"],
        plan=steer_plan,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
    )
    server_inspect_tokens, server_inspect_total = _server_multi_session_stepwise(
        inspect_server,
        inspect_plan,
        prompts,
        max_new_tokens=config.max_new_tokens,
        site_path=site_path,
    )
    server_steer_tokens, _ = _server_multi_session_stepwise(
        inspect_server,
        steer_plan,
        prompts,
        max_new_tokens=config.max_new_tokens,
        site_path=None,
    )
    checks["inspect_multi"] = {
        "ok": torch.equal(local_inspect_tokens, server_inspect_tokens)
        and abs(local_inspect_total - server_inspect_total) < 1e-3,
        "activation_sum_diff": abs(local_inspect_total - server_inspect_total),
    }
    checks["steer_multi"] = {
        "ok": torch.equal(local_steer_generate, server_steer_tokens),
    }
    checks["generate_batched_map"] = {
        "ok": torch.equal(local_steer_generate, server_steer_batched),
    }
    checks["generate_many_map"] = {
        "ok": torch.equal(local_steer_generate, server_many_map.to(local_steer_generate.device)),
    }
    del inspect_server
    _clear_cuda(device)
    return checks


def _tinyinterp_capture_loop(
    model: ti.Model,
    proxy: Any,
    dataset: list[dict[str, torch.Tensor]],
) -> float:
    total = 0.0
    for batch in dataset:
        output = model(**batch, use_cache=False, get=[proxy], stop_at_last_get=True)
        total += float(cast(torch.Tensor, output[proxy]).detach().cpu().float().sum().item())
    return total


def _server_collector_loop(
    collector: Any,
    dataset: list[dict[str, torch.Tensor]],
    site_path: str,
) -> float:
    total = 0.0
    for batch in dataset:
        result = collector.collect_batch(batch)
        total += float(cast(torch.Tensor, result.activations[site_path]).float().sum().item())
    return total


def _model_collect_loop(
    model: Any,
    proxy: Any,
    dataset: list[dict[str, torch.Tensor]],
) -> float:
    total = 0.0
    for batch in dataset:
        outputs = model.collect(_requests_from_batch(batch), get=[proxy])
        total += sum(
            float(cast(torch.Tensor, output[proxy]).detach().cpu().float().sum().item())
            for output in outputs
        )
    return total


def _server_multi_session(
    server: ti.Server,
    plan: Any,
    prompts: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    cache: str = "dynamic",
) -> torch.Tensor:
    sessions = [
        server.open_session(
            plan=plan,
            cache=cache,
            limits={"max_total_tokens": int(prompts["input_ids"].shape[-1]) + max_new_tokens},
        )
        for _ in range(prompts["input_ids"].shape[0])
    ]
    try:
        _ = server.prefill_many(
            sessions,
            input_ids=prompts["input_ids"],
            attention_mask=prompts["attention_mask"],
        )
        steps = server.decode(sessions, max_new_tokens=max_new_tokens, do_sample=False)
        outputs = []
        for idx, step in enumerate(steps):
            generated = step.token_ids
            if generated is None:
                generated = torch.empty(
                    (1, 0),
                    dtype=torch.long,
                    device=prompts["input_ids"].device,
                )
            outputs.append(
                torch.cat(
                    [
                        prompts["input_ids"][idx : idx + 1],
                        generated.to(prompts["input_ids"].device),
                    ],
                    dim=-1,
                )
            )
        return _pad_batch_sequences(
            outputs,
            pad_token_id=_pad_token_id(server._model.wrapped),
        )
    finally:
        for session in sessions:
            server.close_session(session)


def _server_multi_session_stepwise(
    server: ti.Server,
    plan: Any,
    prompts: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    site_path: str | None,
    cache: str = "dynamic",
) -> tuple[torch.Tensor, float]:
    sessions = [
        server.open_session(
            plan=plan,
            cache=cache,
            limits={"max_total_tokens": int(prompts["input_ids"].shape[-1]) + max_new_tokens},
        )
        for _ in range(prompts["input_ids"].shape[0])
    ]
    total = 0.0
    try:
        prefills = server.prefill_many(
            sessions,
            input_ids=prompts["input_ids"],
            attention_mask=prompts["attention_mask"],
        )
        if site_path is not None:
            total += sum(
                float(cast(torch.Tensor, result.activations[site_path]).float().sum().item())
                for result in prefills
            )
        for _ in range(max_new_tokens):
            steps = server.decode(sessions, max_new_tokens=1, do_sample=False)
            if site_path is not None:
                total += sum(
                    float(cast(torch.Tensor, step.activations[site_path]).float().sum().item())
                    for step in steps
                    if site_path in step.activations
                )
        outputs = []
        for idx, session in enumerate(sessions):
            generated = torch.tensor(
                [session.generated_cpu],
                dtype=torch.long,
                device=prompts["input_ids"].device,
            )
            outputs.append(
                torch.cat(
                    [prompts["input_ids"][idx : idx + 1], generated],
                    dim=-1,
                )
            )
        return _pad_batch_sequences(
            outputs, pad_token_id=_pad_token_id(server._model.wrapped)
        ), total
    finally:
        for session in sessions:
            server.close_session(session)


def _server_call_concurrent(
    server: ti.Server,
    plan: Any,
    requests: list[dict[str, torch.Tensor]],
) -> float:
    rows = [
        {
            key: (
                value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim == 1 else value
            )
            for key, value in request.items()
        }
        for request in requests
    ]
    totals = [0.0 for _ in rows]
    errors: list[BaseException] = []

    def worker(idx: int, request: Mapping[str, torch.Tensor]) -> None:
        try:
            result = server.call(plan, **request)
            logits = cast(torch.Tensor, result.logits)
            totals[idx] = float(logits.detach().cpu().float().sum().item())
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(idx, request), daemon=True)
        for idx, request in enumerate(rows)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("Concurrent stateless server.call benchmark failed.") from errors[0]
    return sum(totals)


def _tinyinterp_decode_batched(
    model: ti.Model,
    prompts: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    proxy: Any | None = None,
    mapping: Mapping[Any, Any] | None = None,
) -> tuple[torch.Tensor, float]:
    total = 0.0
    pad_token_id = _pad_token_id(model.wrapped)
    eos_ids = eos_token_ids(model.wrapped)
    active_indices = list(range(prompts["input_ids"].shape[0]))
    active_attention = prompts["attention_mask"]
    cache: Any | None = None
    cache_enabled = callable(getattr(model.wrapped, "prepare_inputs_for_generation", None))
    incremental_input: torch.Tensor | None = None
    generated_by_row: list[list[torch.Tensor]] = [[] for _ in range(prompts["input_ids"].shape[0])]

    for step_idx in range(max_new_tokens):
        if not active_indices:
            break
        if cache_enabled and cache is not None and incremental_input is not None:
            prepared = _prepare_generation_inputs(
                model.wrapped,
                incremental_input,
                active_attention,
                cache,
            )
        else:
            active_tokens = []
            for row_idx in active_indices:
                pieces = [prompts["input_ids"][row_idx : row_idx + 1], *generated_by_row[row_idx]]
                active_tokens.append(torch.cat(pieces, dim=-1))
            full_tokens = torch.cat(active_tokens, dim=0)
            active_attention = torch.ones_like(full_tokens)
            prepared = {
                "input_ids": full_tokens,
                "attention_mask": active_attention,
                "use_cache": cache_enabled,
            }
            cache = None
            incremental_input = None

        output = model(
            get=[proxy] if proxy is not None else None,
            map=dict(mapping) if mapping is not None else None,
            **filter_supported_kwargs(
                model.wrapped,
                prepared,
            ),
        )
        if proxy is not None:
            total += float(cast(torch.Tensor, output[proxy]).detach().cpu().float().sum().item())
        cache = getattr(
            output._model_output if isinstance(output, ti.Output) else output,
            "past_key_values",
            cache,
        )
        logits = extract_last_token_logits(output)
        next_token = logits.argmax(dim=-1, keepdim=True)
        next_active_indices: list[int] = []
        next_incremental: list[torch.Tensor] = []
        for local_idx, row_idx in enumerate(active_indices):
            token = next_token[local_idx : local_idx + 1]
            generated_by_row[row_idx].append(token)
            if not contains_eos(token, eos_ids):
                next_active_indices.append(row_idx)
                next_incremental.append(token)
        if step_idx == max_new_tokens - 1 or not next_active_indices:
            break
        if len(next_active_indices) == len(active_indices):
            incremental_input = torch.cat(next_incremental, dim=0)
            active_attention = torch.cat(
                [
                    active_attention,
                    torch.ones(
                        (active_attention.shape[0], 1),
                        dtype=active_attention.dtype,
                        device=active_attention.device,
                    ),
                ],
                dim=-1,
            )
        else:
            incremental_input = None
            cache = None
        active_indices = next_active_indices

    rows = []
    for row_idx in range(prompts["input_ids"].shape[0]):
        generated = (
            torch.cat(generated_by_row[row_idx], dim=-1)
            if generated_by_row[row_idx]
            else torch.empty((1, 0), dtype=torch.long, device=prompts["input_ids"].device)
        )
        rows.append(torch.cat([prompts["input_ids"][row_idx : row_idx + 1], generated], dim=-1))
    return _pad_batch_sequences(rows, pad_token_id=pad_token_id), total


def _tinyinterp_generate_batched(
    model: ti.Model,
    prompts: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    mapping: Mapping[Any, Any] | None = None,
) -> torch.Tensor:
    output = model.generate(
        input_ids=prompts["input_ids"],
        attention_mask=prompts["attention_mask"],
        map=dict(mapping) if mapping is not None else None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if isinstance(output, ti.GenerateOutput):
        return cast(torch.Tensor, output.sequences)
    return cast(torch.Tensor, output.sequences)


def _hf_generate(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    with torch.no_grad():
        return cast(
            torch.Tensor,
            cast(Any, model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            ),
        )


def _hf_generate_sequential(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    outputs = [
        _hf_generate(
            model,
            {
                "input_ids": batch["input_ids"][idx : idx + 1],
                "attention_mask": batch["attention_mask"][idx : idx + 1],
            },
            max_new_tokens=max_new_tokens,
        )
        for idx in range(batch["input_ids"].shape[0])
    ]
    return _pad_batch_sequences(outputs, pad_token_id=_pad_token_id(model))


def _requests_from_batch(batch: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    return [
        {
            "input_ids": input_ids[idx],
            "attention_mask": attention_mask[idx],
        }
        for idx in range(input_ids.shape[0])
    ]


def _prepare_generation_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any | None,
) -> dict[str, Any]:
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if not callable(prepare):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": True,
        }
    prepare_kwargs = {
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "use_cache": True,
    }
    cache_position = _cache_position(cache, input_ids)
    if cache_position is not None:
        prepare_kwargs["cache_position"] = cache_position
    prepared = cast(Mapping[str, Any], prepare(input_ids, **prepare_kwargs))
    return {**prepared, "use_cache": True}


def _cache_position(cache: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    if cache is None:
        start = 0
    elif callable(getattr(cache, "get_seq_length", None)):
        start = int(cache.get_seq_length())
    else:
        return None
    return torch.arange(start, start + input_ids.shape[-1], device=input_ids.device)


class _ManualHookLoop:
    def __init__(self, model: nn.Module, site_path: str) -> None:
        self.model = model
        self.module = _resolve_module(model, site_path)
        self.captured: dict[str, torch.Tensor] = {}
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        self.captured["act"] = _extract(output).detach().cpu()

    def run(self, dataset: list[dict[str, torch.Tensor]]) -> float:
        total = 0.0
        with torch.no_grad():
            for batch in dataset:
                self.captured.clear()
                _ = self.model(**batch, use_cache=False)
                total += float(self.captured["act"].float().sum().item())
        return total

    def close(self) -> None:
        self.handle.remove()


def _manual_hook_once(
    model: nn.Module,
    site_path: str,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> None:
        captured["act"] = _extract(output).detach().cpu()

    handle = _resolve_module(model, site_path).register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**batch, use_cache=False)
    finally:
        handle.remove()
    return captured["act"]


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return cast(nn.Module, current)


def _resolve_proxy(model: ti.Model, path: str) -> Any:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _open_remote_model(sock_path: str) -> Any:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if os.path.exists(sock_path):
            try:
                return ti.Model(f"unix://{sock_path}")
            except OSError:
                time.sleep(0.02)
                continue
        time.sleep(0.02)
    raise RuntimeError(f"Remote benchmark server did not open {sock_path}.")


def _supports_static_cache(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        config = get_text(decoder=True)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return True
    supported = {"full_attention", "sliding_attention", "chunked_attention"}
    return all(layer_type in supported for layer_type in layer_types)


def _load_model(
    config: RuntimeInternalsBenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Server benchmarks require `transformers`.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
    )
    return model.to(device=device).eval()


def _make_dataset(
    *,
    batch_size: int,
    batches: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    outputs: list[dict[str, torch.Tensor]] = []
    for _ in range(batches):
        input_ids = torch.randint(
            low=3,
            high=max(vocab_size, 4),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        outputs.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return outputs


def _site_path(model_name: str, model: nn.Module) -> str:
    wrapper = ti.Model(model, rename=ti.renames.llm)
    site = ti.find(wrapper.layers[0], "linear_attn")
    if site is None:
        site = ti.find(wrapper.layers[0], "self_attn")
    if site is None:
        site = ti.find(wrapper.layers[0], "attn")
    if site is None:
        raise RuntimeError("Could not infer a benchmark site path.")
    return site.path


def _annotate_server_metrics(
    cases: list[dict[str, Any]],
    *,
    config: RuntimeInternalsBenchmarkConfig,
) -> None:
    examples = config.dataset_batch_size * config.dataset_batches
    single_tokens = config.max_new_tokens
    multi_tokens = config.generate_batch_size * config.max_new_tokens
    for case in cases:
        if case.get("skipped"):
            continue
        name = case["name"]
        if name in {"hf_hook_loop", "tinyinterp_capture_loop", "server_collector"}:
            case["examples_per_sec"] = examples / (case["median_ms"] / 1000.0)
            case["tokens_per_sec"] = (examples * config.seq_len) / (case["median_ms"] / 1000.0)
        elif name in {"hf_generate_single", "server_generate_single"}:
            case["examples_per_sec"] = 1.0 / (case["median_ms"] / 1000.0)
            case["tokens_per_sec"] = single_tokens / (case["median_ms"] / 1000.0)
        elif name in {
            "server_generate_batched_plain",
            "server_generate_many_plain",
            "hf_generate_multi_sequential",
            "hf_generate_multi_batched",
            "server_generate_multi_session",
            "server_generate_multi_session_static",
            "server_generate_batched_map",
            "server_generate_many_map",
            "tinyinterp_generate_map_batched",
            "tinyinterp_decode_get_batched",
            "server_decode_get_multi_session",
            "tinyinterp_decode_map_batched",
            "server_decode_map_multi_session",
        }:
            case["examples_per_sec"] = config.generate_batch_size / (case["median_ms"] / 1000.0)
            case["tokens_per_sec"] = multi_tokens / (case["median_ms"] / 1000.0)


def _vocab_size(model: nn.Module) -> int:
    config = getattr(model, "config", SimpleNamespace(vocab_size=256))
    return int(_config_value(config, "vocab_size") or 256)


def _reshape_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    d_head = tensor.shape[-1] // n_heads
    return tensor.view(*tensor.shape[:-1], n_heads, d_head)


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pad_batch_sequences(sequences: list[torch.Tensor], *, pad_token_id: int) -> torch.Tensor:
    max_len = max(sequence.shape[1] for sequence in sequences)
    padded: list[torch.Tensor] = []
    for sequence in sequences:
        if sequence.shape[1] == max_len:
            padded.append(sequence)
            continue
        pad = torch.full(
            (sequence.shape[0], max_len - sequence.shape[1]),
            pad_token_id,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        padded.append(torch.cat([sequence, pad], dim=-1))
    return torch.cat(padded, dim=0)


def _normalize_generated_batch(value: Any, *, pad_token_id: int) -> torch.Tensor:
    if isinstance(value, ti.GenerateOutput):
        return cast(torch.Tensor, value.sequences)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, ti.GenerateOutput) for item in value)
    ):
        return _pad_batch_sequences(
            [cast(torch.Tensor, item.sequences) for item in cast(list[ti.GenerateOutput], value)],
            pad_token_id=pad_token_id,
        )
    if not isinstance(value, list) or not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(
            f"Expected GenerateOutput or list[GenerateOutput], got {type(value).__name__}."
        )
    return _pad_batch_sequences(cast(list[torch.Tensor], value), pad_token_id=pad_token_id)


def _pad_token_id(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return 0
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(config, "eos_token_id", 0)
    if isinstance(eos_token_id, int):
        return eos_token_id
    if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
        return int(eos_token_id[0])
    return 0
