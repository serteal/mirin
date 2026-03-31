"""Workload helpers for runtime-internals benchmarks."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Mapping
from typing import Any, cast

import torch

import tinyinterp as ti

from .runtime_internals_shared import (
    _clear_cuda,
    _load_model,
    _manual_hook_once,
    _normalize_generated_batch,
    _open_remote_model,
    _pad_batch_sequences,
    _pad_token_id,
    _prepare_generation_inputs,
    _requests_from_batch,
    _resolve_proxy,
    contains_eos,
    eos_token_ids,
    extract_last_token_logits,
    filter_supported_kwargs,
)


def _correctness_checks(
    config: Any,
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
    return cast(torch.Tensor, output.sequences)


def _hf_generate(
    model: torch.nn.Module,
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
    model: torch.nn.Module,
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


def _annotate_server_metrics(
    cases: list[dict[str, Any]],
    *,
    config: Any,
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
