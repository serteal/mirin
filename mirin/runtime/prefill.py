"""Collection helpers for bounded local activation extraction."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .collector import Collector, MmapActivationRef, MmapSegment, _split_batch, _torch_dtype_name
from .results import PlanResult
from .util import (
    batch_size_from_mapping,
    filter_supported_kwargs,
    move_tensors_to,
    to_cpu,
    to_cpu_dict,
)


def collect_batch(
    runtime: Any,
    collector: Collector,
    batch: Mapping[str, Any],
) -> PlanResult:
    batch_tokens = int(batch.get("attention_mask", batch.get("input_ids")).sum().item()) if isinstance(
        batch.get("attention_mask", batch.get("input_ids")), torch.Tensor
    ) else 0
    chunks = _split_batch(batch, collector.token_budget)
    if len(chunks) > 1:
        runtime._record_split(
            "collect_batch",
            reason="collect_token_budget",
            original_items=batch_size_from_mapping(batch),
            produced_chunks=len(chunks),
        )
        return _concat_batch_results([collect_batch(runtime, collector, chunk) for chunk in chunks])
    budget_chunks = runtime._auto_chunk_batch(collector.plan, batch)
    if budget_chunks is not None:
        runtime._record_split(
            "collect_batch",
            reason="capacity_batch_size",
            original_items=batch_size_from_mapping(batch),
            produced_chunks=len(budget_chunks),
        )
        return _concat_batch_results(
            [collect_batch(runtime, collector, chunk) for chunk in budget_chunks]
        )
    estimate = runtime._estimate_collection(
        collector.plan,
        batch=batch,
        activation_budget_bytes=collector.activation_budget_bytes,
    )
    with runtime._scheduled(
        "collect_batch",
        estimate=estimate,
        batch_size=batch_size_from_mapping(batch),
        batch_tokens=batch_tokens,
        cpu_bytes=estimate.activation_bytes if collector.activations_to_cpu or collector.uses_mmap else 0,
    ):
        kwargs = cast(dict[str, Any], move_tensors_to(dict(batch), runtime._primary_device()))
        runtime._record_physical_batch(
            "collect_batch",
            batch_size=batch_size_from_mapping(batch),
            batch_tokens=batch_tokens,
            context_tokens=batch_tokens,
        )
        if not collector.use_cache:
            kwargs.setdefault("use_cache", False)
        result = runtime._execute_plan(
            collector.plan,
            kwargs=filter_supported_kwargs(runtime._model.wrapped, kwargs),
            stop_at_last_get=collector.stop_at_last_get,
        )
        plan_result = runtime._build_plan_result(
            collector.plan,
            result,
            logits_slice=False,
            activations_to_cpu=False,
            logits_to_cpu=False,
        )
        if collector.uses_mmap:
            mmap_refs, mmap_files = _write_mmap_batch(collector, plan_result)
            plan_result.activations = mmap_refs
            plan_result.metadata["mmap_files"] = mmap_files
            plan_result.metadata["activation_output"] = "mmap"
        else:
            plan_result.activations = to_cpu_dict(
                plan_result.activations,
                enabled=collector.activations_to_cpu,
                pin_memory=collector.pin_memory,
            )
            plan_result.metadata["activation_output"] = collector.activation_output
        if plan_result.logits is not None:
            plan_result.logits = to_cpu(
                plan_result.logits,
                enabled=collector.plan.output.logits_to_cpu,
                pin_memory=collector.pin_memory,
            )
        return plan_result


def _write_mmap_batch(
    collector: Collector,
    result: PlanResult,
) -> tuple[dict[str, MmapActivationRef], dict[str, list[str]]]:
    root = Path(collector.mmap_path or "")
    root.mkdir(parents=True, exist_ok=True)
    refs: dict[str, MmapActivationRef] = {}
    written: dict[str, list[str]] = {}
    for path, value in result.activations.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        storage = tensor
        storage_dtype = _torch_dtype_name(tensor.dtype)
        if tensor.dtype == torch.bfloat16:
            storage = tensor.view(torch.uint16)
            storage_dtype = "uint16"
        array = storage.numpy()
        suffix = collector.mmap_state.get("counter", 0)
        collector.mmap_state["counter"] = suffix + 1
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path)
        filename = root / f"{suffix:06d}-{safe}.mmap"
        mem = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
        mem[...] = array
        mem.flush()
        segment = MmapSegment(
            filename=str(filename),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=_torch_dtype_name(tensor.dtype),
            storage_dtype=storage_dtype,
        )
        refs[path] = MmapActivationRef(path=path, segments=(segment,))
        written[path] = [str(filename)]
    return refs, written


def _concat_batch_results(results: Sequence[PlanResult]) -> PlanResult:
    if not results:
        return PlanResult()
    first = results[0]
    activations: dict[str, Any] = {}
    for path in first.activations:
        values = [result.activations[path] for result in results if path in result.activations]
        if values and all(isinstance(value, torch.Tensor) for value in values):
            activations[path] = torch.cat(cast(list[torch.Tensor], values), dim=0)
        elif values and all(isinstance(value, MmapActivationRef) for value in values):
            segments = tuple(
                segment
                for value in cast(list[MmapActivationRef], values)
                for segment in value.segments
            )
            activations[path] = MmapActivationRef(path=path, segments=segments)
        elif values:
            activations[path] = values[-1]
    logits: torch.Tensor | None = None
    logit_values = [result.logits for result in results if isinstance(result.logits, torch.Tensor)]
    if logit_values:
        logits = torch.cat(cast(list[torch.Tensor], logit_values), dim=0)
    metadata: dict[str, Any] = {}
    mmap_values = [result.metadata.get("mmap_files") for result in results if result.metadata]
    if mmap_values:
        merged: dict[str, list[str]] = defaultdict(list)
        for value in mmap_values:
            if not isinstance(value, Mapping):
                continue
            for path, filename in value.items():
                if isinstance(filename, Sequence) and not isinstance(filename, str):
                    merged[str(path)].extend(str(item) for item in filename)
                else:
                    merged[str(path)].append(str(filename))
        metadata["mmap_files"] = dict(merged)
    return PlanResult(
        activations=activations,
        logits=logits,
        completed_forward=all(result.completed_forward for result in results),
        metadata=metadata,
    )
