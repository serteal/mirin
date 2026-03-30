"""Prefill and collection engine for the inference server."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..output import Output
from .cache import select_cache_adapter
from .collector import Collector
from .results import PlanResult
from .runtime import (
    batch_size_from_mapping,
    default_attention_mask,
    extract_last_token_logits,
    filter_supported_kwargs,
    model_dtype,
    move_tensors_to,
    prompt_tokens_from_mapping,
    split_activation_dict,
    split_batch_tensor,
    to_cpu,
    to_cpu_dict,
)
from .scheduler import bucket_length, estimate_admission

if TYPE_CHECKING:
    from .decode_engine import DecodeEngine
    from .inference import Server
    from .sessions import Session


class PrefillEngine:
    """Own prefill workloads: collector jobs and prompt ingestion."""

    def __init__(self, server: Server, decode_engine: DecodeEngine) -> None:
        self.server = server
        self.decode_engine = decode_engine

    def collect_batch(self, collector: Collector, batch: Mapping[str, Any]) -> PlanResult:
        batch_tokens = prompt_tokens_from_mapping(batch)
        if collector.token_budget is not None and batch_tokens > collector.token_budget:
            message = (
                "collect_batch rejected by token budget: "
                f"{batch_tokens} > {collector.token_budget}."
            )
            raise MemoryError(message)
        estimate = self._estimate_collection(collector, batch=batch)
        with self.server._scheduled(
            "collect_batch",
            estimate=estimate,
            batch_size=batch_size_from_mapping(batch),
            batch_tokens=batch_tokens,
        ):
            kwargs = cast(
                dict[str, Any], move_tensors_to(dict(batch), self.server._primary_device())
            )
            if not collector.use_cache:
                kwargs.setdefault("use_cache", False)
            result = self.server._execute_plan(
                collector.plan,
                kwargs=filter_supported_kwargs(self.server._model.wrapped, kwargs),
                stop_at_last_get=collector.stop_at_last_get,
            )
            plan_result = self.server._build_plan_result(
                collector.plan,
                result,
                logits_slice=False,
                activations_to_cpu=False,
                logits_to_cpu=False,
            )
            if collector.reducer is not None or collector.token_index is not None:
                plan_result.activations = _apply_reducer(
                    plan_result.activations,
                    reducer=collector.reducer,
                    token_index=collector.token_index,
                )
            plan_result.activations = to_cpu_dict(
                plan_result.activations,
                enabled=collector.activations_to_cpu,
                pin_memory=collector.pin_memory,
            )
            if plan_result.logits is not None:
                plan_result.logits = to_cpu(
                    plan_result.logits,
                    enabled=collector.plan.output.logits_to_cpu,
                    pin_memory=collector.pin_memory,
                )
            if collector.mmap_path is not None:
                plan_result.metadata["mmap_files"] = self._write_mmap_batch(collector, plan_result)
            return plan_result

    def prefill(
        self,
        session: Session,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> PlanResult:
        return self.prefill_many(
            [session],
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
            **kwargs,
        )[0]

    def prefill_many(
        self,
        sessions: Sequence[Session],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[PlanResult]:
        if not sessions:
            return []
        if input_ids.ndim != 2:
            raise ValueError("prefill_many() expects [batch, seq] input_ids.")
        if int(input_ids.shape[1]) < 1:
            raise ValueError("prefill_many() expects at least one prompt token.")
        if len(sessions) != int(input_ids.shape[0]):
            raise ValueError("prefill_many() expects one session per batch row.")

        resolved = list(sessions)
        batch_attention = default_attention_mask(
            attention_mask,
            like=input_ids,
            device=self.server._primary_device(),
        )
        batch_input_ids = cast(
            torch.Tensor, move_tensors_to(input_ids, self.server._primary_device())
        )
        batch_attention = cast(
            torch.Tensor, move_tensors_to(batch_attention, self.server._primary_device())
        )
        batch_prompt_tokens = int(batch_input_ids.shape[0] * batch_input_ids.shape[1])

        for idx, session in enumerate(resolved):
            if (
                session.input_ids is not None
                or session.last_logits is not None
                or session.pending_input_ids is not None
            ):
                raise ValueError("prefill() is only valid on a fresh session.")
            session.extra_kwargs = cast(
                dict[str, Any],
                move_tensors_to(dict(kwargs), self.server._primary_device()),
            )
            session.max_total_tokens = self.server._resolve_session_max_total_tokens(
                session.max_total_tokens,
                prompt_len=int(batch_attention[idx : idx + 1].shape[-1]),
                max_new_tokens_hint=session.max_new_tokens_hint,
            )
            session.decode_bucket_len = bucket_length(
                max(
                    session.max_total_tokens or int(batch_attention.shape[-1]),
                    int(batch_attention.shape[-1]),
                ),
                self.server._scheduler.decode_bucket_multiple,
            )
            session.prompt_length = int(batch_attention[idx : idx + 1].sum().item())
            session.current_length = session.prompt_length
            session.history_cpu = batch_input_ids[idx].detach().cpu().tolist()
            session.generated_cpu.clear()

        estimate = self.server._estimate_prefill(
            resolved[0].plan,
            batch_size=int(batch_input_ids.shape[0]),
            prompt_tokens=batch_prompt_tokens,
            projected_decode_tokens=sum(
                max((session.max_total_tokens or 0) - session.prompt_length, 0)
                for session in resolved
            ),
        )
        with self.server._scheduled(
            "prefill",
            estimate=estimate,
            batch_size=int(batch_input_ids.shape[0]),
            batch_tokens=batch_prompt_tokens,
        ):
            grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for idx, session in enumerate(resolved):
                grouped[self.decode_engine.prefill_key(session)].append(idx)

            outputs: list[PlanResult | None] = [None] * len(resolved)
            for indices in grouped.values():
                if len(indices) == 1:
                    idx = indices[0]
                    outputs[idx] = self._prefill_one(
                        resolved[idx],
                        input_ids=batch_input_ids[idx : idx + 1],
                        attention_mask=batch_attention[idx : idx + 1],
                        chunk_size=chunk_size,
                    )
                    continue
                subset_sessions = [resolved[idx] for idx in indices]
                subset_input_ids = batch_input_ids[indices]
                subset_attention = batch_attention[indices]
                adapter = select_cache_adapter(
                    subset_sessions[0].cache,
                    subset_sessions[0].cache_mode,
                )
                if (
                    chunk_size is not None
                    or not subset_sessions[0].use_hf_cache
                    or not adapter.supports_batched_decode()
                ):
                    for local_idx, session in zip(indices, subset_sessions, strict=True):
                        outputs[local_idx] = self._prefill_one(
                            session,
                            input_ids=batch_input_ids[local_idx : local_idx + 1],
                            attention_mask=batch_attention[local_idx : local_idx + 1],
                            chunk_size=chunk_size,
                        )
                    continue
                batch_outputs = self._prefill_batched_hf(
                    subset_sessions,
                    input_ids=subset_input_ids,
                    attention_mask=subset_attention,
                )
                for local_idx, result in zip(indices, batch_outputs, strict=True):
                    outputs[local_idx] = result
        return [cast(PlanResult, output) for output in outputs]

    def _prefill_one(
        self,
        session: Session,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int | None,
    ) -> PlanResult:
        if session.use_hf_cache:
            cache = None
            result: PlanResult | None = None
            raw_result: Any = None
            if chunk_size is None and self.server._scheduler.prefill_token_budget is not None:
                per_example_budget = max(
                    self.server._scheduler.prefill_token_budget // max(input_ids.shape[0], 1),
                    1,
                )
                chunk_size = min(per_example_budget, int(input_ids.shape[-1]))
            if chunk_size is None or chunk_size <= 0:
                raw_result, result = self._execute_hf_chunk(
                    session,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache=cache,
                )
            else:
                for start in range(0, input_ids.shape[1], chunk_size):
                    end = min(start + chunk_size, input_ids.shape[1])
                    raw_result, chunk_result = self._execute_hf_chunk(
                        session,
                        input_ids=input_ids[:, start:end],
                        attention_mask=attention_mask[:, :end],
                        cache=cache,
                    )
                    result = _merge_chunk_result(result, chunk_result)
                    cache = session.cache
            if result is None:
                raise RuntimeError("prefill() did not produce a result.")
            session.input_ids = None
            session.attention_mask = None
            return self.decode_engine.register_prefill_result(
                session,
                raw_result=raw_result,
                plan_result=result,
                attention_mask=attention_mask,
            )

        model_result = self.server._execute_plan(
            session.plan,
            kwargs=filter_supported_kwargs(
                self.server._model.wrapped,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    **session.extra_kwargs,
                },
            ),
        )
        result = self.server._build_plan_result(
            session.plan,
            model_result,
            logits_slice=True,
            activations_to_cpu=session.plan.output.activations_to_cpu,
            logits_to_cpu=session.plan.output.logits_to_cpu,
        )
        session.last_logits = extract_last_token_logits(model_result)
        session.input_ids = None
        session.attention_mask = None
        return self.decode_engine.register_prefill_result(
            session,
            raw_result=model_result,
            plan_result=result,
            attention_mask=attention_mask,
        )

    def _prefill_batched_hf(
        self,
        sessions: Sequence[Session],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[PlanResult]:
        first = sessions[0]
        prepared = self.server._prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=None,
            extra_kwargs=first.extra_kwargs,
        )
        result = self.server._execute_plan(first.plan, kwargs=prepared)
        logits = split_batch_tensor(extract_last_token_logits(result), len(sessions))
        activations = split_activation_dict(
            self.server._extract_activations(first.plan, result),
            len(sessions),
        )
        batch_cache = getattr(
            result._model_output if isinstance(result, Output) else result, "past_key_values", None
        )
        if batch_cache is None:
            return [
                self.server._result_for_session_plan(
                    session,
                    PlanResult(
                        session_id=session.id,
                        logits=to_cpu(logits[idx], enabled=session.plan.output.logits_to_cpu),
                        activations=to_cpu_dict(
                            activations[idx],
                            enabled=session.plan.output.activations_to_cpu,
                        ),
                    ),
                )
                for idx, session in enumerate(sessions)
            ]
        adapter = select_cache_adapter(batch_cache, first.cache_mode)
        if not adapter.supports_batched_decode():
            fallback_outputs: list[PlanResult] = []
            for idx, session in enumerate(sessions):
                fallback_outputs.append(
                    self._prefill_one(
                        session,
                        input_ids=input_ids[idx : idx + 1],
                        attention_mask=attention_mask[idx : idx + 1],
                        chunk_size=None,
                    )
                )
            return fallback_outputs
        outputs: list[PlanResult] = []
        for idx, session in enumerate(sessions):
            session.cache = None
            plan_result = PlanResult(
                session_id=session.id,
                logits=to_cpu(logits[idx], enabled=session.plan.output.logits_to_cpu)
                if session.plan.output.logits
                else None,
                activations=to_cpu_dict(
                    activations[idx],
                    enabled=session.plan.output.activations_to_cpu,
                )
                if session.plan.output.activations
                else {},
            )
            outputs.append(
                self.decode_engine.register_prefill_result(
                    session,
                    raw_result=result,
                    plan_result=plan_result,
                    attention_mask=attention_mask[idx : idx + 1],
                    adapter=adapter,
                    shared_cache=batch_cache,
                    shared_index=idx,
                )
            )
        self.decode_engine.register_prefilled_family(
            list(sessions),
            shared_cache=batch_cache,
            adapter=adapter,
            attention_mask=attention_mask,
            raw_result=result,
        )
        return outputs

    def _execute_hf_chunk(
        self,
        session: Session,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Any | None,
    ) -> tuple[Any, PlanResult]:
        prepared = self.server._prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            extra_kwargs=session.extra_kwargs,
        )
        result = self.server._execute_plan(session.plan, kwargs=prepared)
        if isinstance(result, Output):
            model_output = result._model_output
        else:
            model_output = result
        session.cache = getattr(model_output, "past_key_values", session.cache)
        return (
            result,
            self.server._build_plan_result(
                session.plan,
                result,
                logits_slice=True,
                activations_to_cpu=session.plan.output.activations_to_cpu,
                logits_to_cpu=False,
            ),
        )

    def _estimate_collection(
        self,
        collector: Collector,
        *,
        batch: Mapping[str, Any],
    ) -> Any:
        activation_cap = (
            collector.activation_budget_bytes or self.server._scheduler.max_activation_capture_bytes
        )
        return estimate_admission(
            queue="collect_batch",
            wrapped=self.server._model.wrapped,
            plan=collector.plan,
            dtype=model_dtype(self.server._model.wrapped),
            batch_size=batch_size_from_mapping(batch),
            prompt_tokens=prompt_tokens_from_mapping(batch),
            projected_decode_tokens=0,
            bucket_multiple=self.server._scheduler.decode_bucket_multiple,
            max_kv_cache_bytes=self.server._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=activation_cap,
        )

    def _write_mmap_batch(self, collector: Collector, result: PlanResult) -> dict[str, str]:
        root = Path(collector.mmap_path or "")
        root.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for path, value in result.activations.items():
            if not isinstance(value, torch.Tensor):
                continue
            array = value.detach().cpu().contiguous().numpy()
            suffix = collector.mmap_state.get("counter", 0)
            collector.mmap_state["counter"] = suffix + 1
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path)
            filename = root / f"{suffix:06d}-{safe}.mmap"
            mem = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
            mem[...] = array
            mem.flush()
            written[path] = str(filename)
        return written


def _apply_reducer(
    activations: Mapping[str, Any],
    *,
    reducer: str | None,
    token_index: int | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path, value in activations.items():
        if not isinstance(value, torch.Tensor):
            output[path] = value
            continue
        reduced = value
        if token_index is not None and reduced.ndim >= 2:
            idx = token_index if token_index >= 0 else reduced.shape[1] + token_index
            reduced = reduced[:, idx : idx + 1]
        if reducer == "mean_tokens" and reduced.ndim >= 3:
            reduced = reduced.mean(dim=1)
        elif reducer == "last_token" and reduced.ndim >= 3:
            reduced = reduced[:, -1]
        output[path] = reduced
    return output


def _merge_chunk_result(current: PlanResult | None, update: PlanResult) -> PlanResult:
    if current is None:
        return update
    activations = dict(current.activations)
    for path, value in update.activations.items():
        prev = activations.get(path)
        if (
            isinstance(prev, torch.Tensor)
            and isinstance(value, torch.Tensor)
            and prev.ndim >= 2
            and value.ndim >= 2
            and prev.shape[0] == value.shape[0]
        ):
            activations[path] = torch.cat([prev, value], dim=1)
        else:
            activations[path] = value
    current.activations = activations
    current.logits = update.logits
    current.completed_forward = update.completed_forward
    current.metadata.update(update.metadata)
    return current
