"""Server runtime benchmark harness for the inference server."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

import tinyinterp as ti

from .model_api import (
    _environment_report,
    _measure_case,
    _resolve_device,
    _resolve_dtype,
)
from .runtime_internals_shared import (
    _clear_cuda,
    _load_model,
    _make_dataset,
    _ManualHookLoop,
    _open_remote_model,
    _requests_from_batch,
    _resolve_proxy,
    _site_path,
    _supports_static_cache,
    _vocab_size,
)
from .runtime_internals_workloads import (
    _annotate_server_metrics,
    _correctness_checks,
    _hf_generate,
    _hf_generate_sequential,
    _model_collect_loop,
    _server_call_concurrent,
    _server_collector_loop,
    _server_multi_session,
    _server_multi_session_stepwise,
    _tinyinterp_capture_loop,
    _tinyinterp_decode_batched,
    _tinyinterp_generate_batched,
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
