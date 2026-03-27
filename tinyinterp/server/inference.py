"""In-process inference server for HuggingFace-style CausalLM execution."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch

from ..context import get_debug
from ..model import Model
from ..output import Output
from .collector import Collector
from .decode_engine import DecodeEngine
from .plans import CompiledPlan, OutputPolicyLike, SiteLike, compile_plan
from .prefill_engine import PrefillEngine
from .results import PlanResult
from .runtime import (
    contains_eos,
    default_attention_mask,
    eos_token_ids,
    extract_last_token_logits,
    filter_supported_kwargs,
    gpu_stats,
    is_static_cache,
    model_dtype,
    move_tensors_to,
    prompt_tokens_from_mapping,
    supports_static_cache_model,
    to_cpu,
    to_cpu_dict,
)
from .scheduler import (
    AdmissionEstimate,
    QueueMetrics,
    SchedulerConfig,
    estimate_admission,
)
from .sessions import SamplingConfig, Session, sample_next_token


@dataclass(slots=True)
class _OpStats:
    calls: int = 0
    errors: int = 0
    total_ns: int = 0
    inflight: int = 0
    peak_inflight: int = 0


@dataclass(slots=True)
class _RequestBatch:
    rows: list[dict[str, torch.Tensor]]
    batch: dict[str, torch.Tensor]


class Server:
    """Own one local ``ti.Model`` plus specialized prefill and decode engines."""

    def __init__(
        self,
        wrapped: torch.nn.Module | str,
        *,
        rename: Mapping[str, str] | None = None,
        tokenizer: Any | None = None,
        device: str | torch.device | None = None,
        attn_backend: str | None = None,
        decode_bucket_multiple: int = 64,
        decode_max_batch_tokens: int | None = None,
        prefill_token_budget: int | None = None,
        collect_token_budget: int | None = None,
        max_kv_cache_mb: float | None = None,
        max_activation_capture_mb: float | None = None,
        **load_kwargs: Any,
    ) -> None:
        if attn_backend is not None and "attn_implementation" not in load_kwargs:
            load_kwargs["attn_implementation"] = attn_backend
        self.model = Model(wrapped, rename=rename, tokenizer=tokenizer, **load_kwargs)
        if device is not None:
            self.model.wrapped.to(device)
        self._plans: dict[str, CompiledPlan] = {}
        self._sessions: dict[str, Session] = {}
        self._collectors: dict[str, Collector] = {}
        self._stats: dict[str, _OpStats] = defaultdict(_OpStats)
        self._queues: dict[str, QueueMetrics] = defaultdict(QueueMetrics)
        self._last_request_type = ""
        self._last_admission: dict[str, Any] | None = None
        self._server_started_ns = time.perf_counter_ns()
        self._scheduler = SchedulerConfig(
            decode_bucket_multiple=decode_bucket_multiple,
            decode_max_batch_tokens=decode_max_batch_tokens,
            prefill_token_budget=prefill_token_budget,
            collect_token_budget=collect_token_budget,
            max_kv_cache_bytes=(
                None if max_kv_cache_mb is None else int(max_kv_cache_mb * 1024 * 1024)
            ),
            max_activation_capture_bytes=(
                None
                if max_activation_capture_mb is None
                else int(max_activation_capture_mb * 1024 * 1024)
            ),
        )
        self._execution_lock = threading.Lock()
        self._decode_engine = DecodeEngine(self)
        self._prefill_engine = PrefillEngine(self, self._decode_engine)

    def compile(
        self,
        *,
        get: Sequence[SiteLike] | SiteLike | None = None,
        mapping: Mapping[SiteLike, Any] | None = None,
        output: OutputPolicyLike = None,
    ) -> CompiledPlan:
        with self._track("compile"):
            plan = compile_plan(self.model, get=get, mapping=mapping, output=output)
            self._plans[plan.id] = plan
            return plan

    def call(
        self,
        plan: CompiledPlan | str | None = None,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> PlanResult:
        compiled = self._resolve_plan(plan)
        with self._scheduled("call"):
            with torch.inference_mode():
                result = self.model(
                    *move_tensors_to(tuple(args), self._primary_device()),
                    get=list(compiled.get_proxies),
                    map=compiled.map_dict,
                    **cast(dict[str, Any], move_tensors_to(dict(kwargs), self._primary_device())),
                )
            return self._build_plan_result(
                compiled,
                result,
                logits_slice=False,
                activations_to_cpu=compiled.output.activations_to_cpu,
                logits_to_cpu=compiled.output.logits_to_cpu,
            )

    def call_many(
        self,
        requests: Sequence[Any],
        /,
        *,
        plan: CompiledPlan | str | None = None,
        **kwargs: Any,
    ) -> list[PlanResult]:
        compiled = self._resolve_plan(plan)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=False,
            pad_side="right",
        )
        batch_kwargs = self._merge_batch_kwargs(normalized.batch, kwargs)
        result = self.call(
            compiled,
            **filter_supported_kwargs(self.model.wrapped, batch_kwargs),
        )
        return self._split_plan_result(
            result,
            batch_size=len(normalized.rows),
        )

    def open_collector(
        self,
        *,
        plan: CompiledPlan | str | None = None,
        use_cache: bool = False,
        stop_at_last_get: bool | None = None,
        token_budget: int | None = None,
        activation_budget_bytes: int | None = None,
        activations_to_cpu: bool | None = None,
        pin_memory: bool = False,
        reducer: str | None = None,
        token_index: int | None = None,
        mmap_path: str | None = None,
    ) -> Collector:
        compiled = self._resolve_plan(plan)
        if stop_at_last_get is None:
            stop_at_last_get = (
                bool(compiled.get_proxies) and not compiled.output.logits and not compiled.map_dict
            )
        if stop_at_last_get and compiled.map_dict:
            raise ValueError("Collector fast path does not support map=.")
        if activations_to_cpu is None:
            activations_to_cpu = compiled.output.activations_to_cpu or True
        if token_budget is None:
            token_budget = self._scheduler.collect_token_budget
        collector = Collector(
            id=uuid.uuid4().hex,
            plan=compiled,
            use_cache=use_cache,
            stop_at_last_get=stop_at_last_get,
            token_budget=token_budget,
            activation_budget_bytes=activation_budget_bytes,
            activations_to_cpu=activations_to_cpu,
            pin_memory=pin_memory,
            reducer=reducer,
            token_index=token_index,
            mmap_path=mmap_path,
            server=self,
        )
        self._collectors[collector.id] = collector
        return collector

    def collect_batch(
        self,
        collector: Collector | str,
        batch: Mapping[str, Any],
    ) -> PlanResult:
        state = self._resolve_collector(collector)
        return self._prefill_engine.collect_batch(state, batch)

    def collect_many(
        self,
        collector: Collector | str,
        requests: Sequence[Any],
    ) -> list[PlanResult]:
        state = self._resolve_collector(collector)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=False,
            pad_side="right",
        )
        result = self.collect_batch(
            state,
            normalized.batch,
        )
        return self._split_plan_result(
            result,
            batch_size=len(normalized.rows),
        )

    def close_collector(self, collector: Collector | str) -> None:
        state = self._resolve_collector(collector)
        self._collectors.pop(state.id, None)

    def open_session(
        self,
        *,
        plan: CompiledPlan | str | None = None,
        cache: str = "dynamic",
        sampling: Mapping[str, Any] | None = None,
        limits: Mapping[str, Any] | None = None,
    ) -> Session:
        compiled = self._resolve_plan(plan)
        if cache not in {"dynamic", "static", "none"}:
            raise ValueError("Server supports cache='dynamic', 'static', or 'none'.")
        if cache == "static" and not supports_static_cache_model(self.model.wrapped):
            raise ValueError("Static cache is not supported for this model/configuration.")
        sample_cfg = SamplingConfig(
            do_sample=bool((sampling or {}).get("do_sample", False)),
            temperature=float((sampling or {}).get("temperature", 1.0)),
            top_k=cast(int | None, (sampling or {}).get("top_k")),
        )
        max_total_tokens = cast(int | None, (limits or {}).get("max_total_tokens"))
        max_new_tokens_hint = cast(int | None, (limits or {}).get("max_new_tokens"))
        session = Session(
            id=uuid.uuid4().hex,
            plan=compiled,
            cache_mode=cache,
            sampling=sample_cfg,
            use_hf_cache=cache != "none"
            and callable(getattr(self.model.wrapped, "prepare_inputs_for_generation", None)),
            max_total_tokens=max_total_tokens,
            max_new_tokens_hint=max_new_tokens_hint,
            cache=None,
        )
        self._sessions[session.id] = session
        return session

    def prefill(
        self,
        session: Session | str,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> PlanResult:
        state = self._resolve_session(session)
        result = self._prefill_engine.prefill(
            state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
            **kwargs,
        )
        result.session_id = state.id
        result.prompt_length = state.current_length
        return result

    def prefill_many(
        self,
        sessions: Sequence[Session | str],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[PlanResult]:
        resolved = [self._resolve_session(session) for session in sessions]
        results = self._prefill_engine.prefill_many(
            resolved,
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
            **kwargs,
        )
        for session, result in zip(resolved, results, strict=True):
            result.session_id = session.id
            result.prompt_length = session.current_length
        return results

    def decode(
        self,
        sessions: Sequence[Session | str],
        *,
        max_new_tokens: int = 1,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> list[PlanResult]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1.")
        resolved = [self._resolve_session(session) for session in sessions]
        if not resolved:
            return []
        total_prompt_tokens = sum(max(session.current_length, 1) for session in resolved)
        richest_plan = max(resolved, key=lambda session: len(session.plan.get_paths)).plan
        decode_estimate = self._estimate_prefill(
            richest_plan,
            batch_size=len(resolved),
            prompt_tokens=total_prompt_tokens,
            projected_decode_tokens=max_new_tokens * len(resolved),
        )
        with self._scheduled(
            "decode",
            estimate=decode_estimate,
            batch_size=len(resolved),
            batch_tokens=total_prompt_tokens,
        ):
            return self._decode_engine.decode(
                resolved,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )

    def close_session(self, session: Session | str) -> None:
        state = self._resolve_session(session)
        self._decode_engine.close_session(state)
        self._sessions.pop(state.id, None)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        plan: CompiledPlan | str | None = None,
        cache: str = "dynamic",
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if cache != "dynamic":
            raise ValueError(
                "Server.generate() supports cache='dynamic' only. "
                "Use open_session()/prefill()/decode() for explicit cache modes."
            )
        compiled = self._resolve_plan(plan)
        if self._is_plain_generate_plan(compiled):
            return self._generate_via_wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                **kwargs,
            )
        return self._generate_direct_batched(
            compiled,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            **kwargs,
        )

    def generate_many(
        self,
        requests: Sequence[Any],
        /,
        *,
        plan: CompiledPlan | str | None = None,
        cache: str = "dynamic",
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        if cache != "dynamic":
            raise ValueError(
                "Server.generate_many() supports cache='dynamic' only. "
                "Use open_session()/prefill()/decode() for explicit cache modes."
            )
        compiled = self._resolve_plan(plan)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=True,
            pad_side="left",
        )
        if not callable(getattr(self.model.wrapped, "prepare_inputs_for_generation", None)):
            lengths = {int(row["input_ids"].shape[-1]) for row in normalized.rows}
            if len(lengths) > 1:
                return [
                    self.generate(
                        input_ids=row["input_ids"],
                        attention_mask=row["attention_mask"],
                        plan=compiled,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        **kwargs,
                    )
                    for row in normalized.rows
                ]
        if self._is_plain_generate_plan(compiled):
            output = self._generate_via_wrapped(
                input_ids=normalized.batch["input_ids"],
                attention_mask=normalized.batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                **kwargs,
            )
        else:
            output = self._generate_direct_batched(
                compiled,
                input_ids=normalized.batch["input_ids"],
                attention_mask=normalized.batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                **kwargs,
            )
        generated = output[:, normalized.batch["input_ids"].shape[-1] :]
        eos_ids = eos_token_ids(self.model.wrapped)
        outputs: list[torch.Tensor] = []
        for idx, prompt in enumerate(normalized.rows):
            row_generated = generated[idx : idx + 1]
            outputs.append(
                torch.cat(
                    [
                        prompt["input_ids"],
                        self._trim_generated_tokens(row_generated, eos_ids),
                    ],
                    dim=-1,
                )
            )
        return outputs

    def stats(self) -> dict[str, Any]:
        total_calls = sum(entry.calls for entry in self._stats.values())
        total_errors = sum(entry.errors for entry in self._stats.values())
        total_time_ns = sum(entry.total_ns for entry in self._stats.values())
        mean_request_ms = 0.0 if total_calls == 0 else (total_time_ns / total_calls) / 1e6
        peak_inflight = max((entry.peak_inflight for entry in self._stats.values()), default=0)
        inflight = sum(entry.inflight for entry in self._stats.values())
        queued = sum(entry.current_depth for entry in self._queues.values())
        queue_peak = max((entry.peak_depth for entry in self._queues.values()), default=0)
        queue_wait_ns = sum(entry.total_queue_wait_ns for entry in self._queues.values())
        service_ns = sum(entry.total_service_ns for entry in self._queues.values())
        uptime_ns = max(time.perf_counter_ns() - self._server_started_ns, 1)
        return {
            "connected_clients": 0,
            "queued_requests": queued,
            "queue_peak": queue_peak,
            "requests_served": total_calls,
            "request_errors": total_errors,
            "mean_request_ms": mean_request_ms,
            "mean_queue_wait_ms": 0.0 if total_calls == 0 else (queue_wait_ns / total_calls) / 1e6,
            "scheduler_utilization": min(service_ns / uptime_ns, 1.0),
            "last_request_type": self._last_request_type,
            "active_sessions": len(self._sessions),
            "active_collectors": len(self._collectors),
            "last_admission": self._last_admission,
            "queues": {name: entry.snapshot() for name, entry in self._queues.items()},
            "peak_inflight": peak_inflight,
            "inflight_requests": inflight,
            **gpu_stats(self._primary_device()),
        }

    def _resolve_plan(self, plan: CompiledPlan | str | None) -> CompiledPlan:
        if plan is None:
            return self.compile()
        if isinstance(plan, CompiledPlan):
            return plan
        try:
            return self._plans[plan]
        except KeyError as exc:
            raise KeyError(f"Unknown plan id {plan!r}.") from exc

    def _resolve_session(self, session: Session | str) -> Session:
        if isinstance(session, Session):
            return session
        try:
            return self._sessions[session]
        except KeyError as exc:
            raise KeyError(f"Unknown session id {session!r}.") from exc

    def _resolve_collector(self, collector: Collector | str) -> Collector:
        if isinstance(collector, Collector):
            return collector
        try:
            return self._collectors[collector]
        except KeyError as exc:
            raise KeyError(f"Unknown collector id {collector!r}.") from exc

    def _primary_device(self) -> torch.device:
        device = self.model.device
        if isinstance(device, tuple):
            return device[0]
        return device

    def _estimate_collection(
        self,
        plan: CompiledPlan,
        *,
        batch: Mapping[str, Any],
    ) -> AdmissionEstimate:
        return estimate_admission(
            queue="collect_batch",
            wrapped=self.model.wrapped,
            plan=plan,
            dtype=model_dtype(self.model.wrapped),
            batch_size=_batch_size_from_mapping(batch),
            prompt_tokens=prompt_tokens_from_mapping(batch),
            projected_decode_tokens=0,
            bucket_multiple=self._scheduler.decode_bucket_multiple,
            max_kv_cache_bytes=self._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=self._scheduler.max_activation_capture_bytes,
        )

    def _estimate_prefill(
        self,
        plan: CompiledPlan,
        *,
        batch_size: int,
        prompt_tokens: int,
        projected_decode_tokens: int,
    ) -> AdmissionEstimate:
        return estimate_admission(
            queue="prefill",
            wrapped=self.model.wrapped,
            plan=plan,
            dtype=model_dtype(self.model.wrapped),
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            projected_decode_tokens=projected_decode_tokens,
            bucket_multiple=self._scheduler.decode_bucket_multiple,
            max_kv_cache_bytes=self._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=self._scheduler.max_activation_capture_bytes,
        )

    @contextmanager
    def _scheduled(
        self,
        op: str,
        *,
        estimate: AdmissionEstimate | None = None,
        batch_size: int = 1,
        batch_tokens: int = 0,
    ) -> Iterator[None]:
        queued_at = time.perf_counter_ns()
        queue = self._queues[op]
        queue.enqueued += 1
        queue.current_depth += 1
        queue.peak_depth = max(queue.peak_depth, queue.current_depth)
        if estimate is not None:
            self._last_admission = estimate.snapshot()
            if not estimate.admitted:
                queue.rejected += 1
                queue.current_depth = max(queue.current_depth - 1, 0)
                raise MemoryError(f"{op} rejected by admission control: {estimate.reason}.")
        with self._execution_lock:
            started_at = time.perf_counter_ns()
            queue.current_depth = max(queue.current_depth - 1, 0)
            queue.started += 1
            queue.total_queue_wait_ns += started_at - queued_at
            queue.total_tokens += batch_tokens
            queue.total_batches += 1
            queue.total_sessions += batch_size
            queue.max_batch_sessions = max(queue.max_batch_sessions, batch_size)
            with self._track(op):
                try:
                    yield
                finally:
                    queue.completed += 1
                    queue.total_service_ns += time.perf_counter_ns() - started_at

    @contextmanager
    def _track(self, op: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        entry = self._stats[op]
        entry.inflight += 1
        entry.peak_inflight = max(entry.peak_inflight, entry.inflight)
        self._last_request_type = op
        if get_debug() >= 1:
            print(f"[ti] server: op={op}")
        try:
            yield
        except Exception:
            entry.errors += 1
            raise
        finally:
            elapsed_ns = time.perf_counter_ns() - started
            entry.calls += 1
            entry.total_ns += elapsed_ns
            entry.inflight = max(entry.inflight - 1, 0)

    def _prepare_inputs_for_generation(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Any | None,
        extra_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        wrapped = self.model.wrapped
        prepare = getattr(wrapped, "prepare_inputs_for_generation", None)
        if not callable(prepare):
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "use_cache": True,
                **dict(extra_kwargs),
            }
        prepare_kwargs = dict(extra_kwargs)
        if (
            cache is not None
            and prepare_kwargs.get("cache_position") is None
            and is_static_cache(cache)
        ):
            prepare_kwargs["cache_position"] = torch.arange(
                int(cache.get_seq_length()),
                int(cache.get_seq_length()) + input_ids.shape[-1],
                device=input_ids.device,
            )
        prepared = cast(
            Mapping[str, Any],
            prepare(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                **prepare_kwargs,
            ),
        )
        normalized = {**prepared, "use_cache": True}
        if (
            cache is not None
            and normalized.get("cache_position") is None
            and is_static_cache(cache)
        ):
            start = int(cache.get_seq_length())
            normalized["cache_position"] = torch.arange(
                start,
                start + input_ids.shape[-1],
                device=input_ids.device,
            )
        return normalized

    def _extract_activations(self, plan: CompiledPlan, result: Any) -> dict[str, Any]:
        if not isinstance(result, Output):
            return {}
        return {
            path: result[proxy]
            for path, proxy in zip(plan.get_paths, plan.get_proxies, strict=True)
            if path in plan.get_paths
        }

    def _build_plan_result(
        self,
        plan: CompiledPlan,
        result: Any,
        *,
        logits_slice: bool,
        activations_to_cpu: bool,
        logits_to_cpu: bool,
    ) -> PlanResult:
        activations = to_cpu_dict(
            self._extract_activations(plan, result),
            enabled=activations_to_cpu,
        )
        logits: torch.Tensor | None = None
        if plan.output.logits:
            raw_logits = (
                extract_last_token_logits(result) if logits_slice else self._extract_logits(result)
            )
            logits = to_cpu(raw_logits, enabled=logits_to_cpu)
        completed_forward = result.completed_forward if isinstance(result, Output) else True
        return PlanResult(
            activations=activations if plan.output.activations else {},
            logits=logits,
            completed_forward=completed_forward,
        )

    def _result_for_session_plan(self, session: Session, result: PlanResult) -> PlanResult:
        return PlanResult(
            session_id=session.id,
            activations=result.activations if session.plan.output.activations else {},
            logits=result.logits if session.plan.output.logits else None,
            completed_forward=result.completed_forward,
            metadata=dict(result.metadata),
        )

    def _extract_logits(self, result: Any) -> torch.Tensor:
        model_output = result._model_output if isinstance(result, Output) else result
        logits = getattr(model_output, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits
        if isinstance(model_output, Mapping) and isinstance(
            model_output.get("logits"), torch.Tensor
        ):
            return cast(torch.Tensor, model_output["logits"])
        if isinstance(model_output, torch.Tensor):
            return model_output
        raise TypeError(f"Cannot extract logits from {type(model_output).__name__}.")

    @staticmethod
    def _resolve_session_max_total_tokens(
        current: int | None,
        *,
        prompt_len: int,
        max_new_tokens_hint: int | None = None,
    ) -> int:
        if current is not None:
            return max(current, prompt_len)
        if max_new_tokens_hint is not None:
            return prompt_len + max_new_tokens_hint
        return prompt_len

    def _normalize_requests(
        self,
        requests: Sequence[Any],
        *,
        add_generation_prompt: bool,
        pad_side: str,
    ) -> _RequestBatch:
        if not requests:
            raise ValueError("Expected at least one request.")
        rows = [
            self._normalize_request_row(
                request,
                add_generation_prompt=add_generation_prompt,
            )
            for request in requests
        ]
        devices = {row["input_ids"].device for row in rows}
        if len(devices) != 1:
            raise ValueError("All batched requests must live on the same device.")
        max_len = max(int(row["input_ids"].shape[-1]) for row in rows)
        device = rows[0]["input_ids"].device
        batch_input_ids = torch.full(
            (len(rows), max_len),
            self._pad_token_id(),
            dtype=torch.long,
            device=device,
        )
        batch_attention = torch.zeros(
            (len(rows), max_len),
            dtype=torch.long,
            device=device,
        )
        for idx, row in enumerate(rows):
            input_ids = row["input_ids"].view(-1)
            attention_mask = row["attention_mask"].view(-1)
            length = int(input_ids.shape[0])
            if pad_side == "left":
                batch_input_ids[idx, max_len - length :] = input_ids
                batch_attention[idx, max_len - length :] = attention_mask
            else:
                batch_input_ids[idx, :length] = input_ids
                batch_attention[idx, :length] = attention_mask
        return _RequestBatch(
            rows=rows,
            batch={
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention,
            },
        )

    def _normalize_request_row(
        self,
        request: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        if isinstance(request, str):
            return self._encode_text_request(request)
        if isinstance(request, Mapping):
            if "input_ids" in request:
                return self._normalize_token_request(request)
            if "text" in request:
                return self._encode_text_request(str(request["text"]))
            if "messages" in request:
                return self._encode_messages_request(
                    request["messages"],
                    add_generation_prompt=add_generation_prompt,
                )
            raise TypeError(
                "Request mappings must contain `input_ids`, `text`, or `messages`."
            )
        if (
            isinstance(request, Sequence)
            and request
            and all(isinstance(item, Mapping) for item in request)
        ):
            return self._encode_messages_request(
                request,
                add_generation_prompt=add_generation_prompt,
            )
        raise TypeError(
            "Requests must be strings, chat-message lists, or mappings with "
            "`input_ids`, `text`, or `messages`."
        )

    def _normalize_token_request(
        self,
        request: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        input_ids = self._coerce_token_tensor(request["input_ids"], name="input_ids")
        attention_value = request.get("attention_mask")
        if attention_value is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = self._coerce_token_tensor(
                attention_value,
                name="attention_mask",
            )
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape.")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _encode_text_request(self, text: str) -> dict[str, torch.Tensor]:
        tokenizer = self._require_tokenizer()
        encoded = tokenizer(text, return_tensors="pt")
        return self._normalize_token_request(cast(Mapping[str, Any], encoded))

    def _encode_messages_request(
        self,
        messages: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self._require_tokenizer()
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise TypeError("Server tokenizer does not support chat messages.")
        rendered = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return self._encode_text_request(str(rendered))

    def _require_tokenizer(self) -> Any:
        tokenizer = self.model.tokenizer
        if tokenizer is None:
            raise TypeError(
                "Server requires a tokenizer for string or chat-message requests."
            )
        return tokenizer

    def _coerce_token_tensor(self, value: Any, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            return tensor
        raise ValueError(f"{name} must be shape [seq] or [1, seq].")

    def _pad_token_id(self) -> int:
        tokenizer = self.model.tokenizer
        if tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if isinstance(pad_token_id, int):
                return pad_token_id
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                return eos_token_id
            if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
                return int(eos_token_id[0])
        config = getattr(self.model.wrapped, "config", None)
        if config is not None:
            pad_token_id = getattr(config, "pad_token_id", None)
            if isinstance(pad_token_id, int):
                return pad_token_id
            eos_token_id = getattr(config, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                return eos_token_id
            if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
                return int(eos_token_id[0])
        return 0

    @staticmethod
    def _is_plain_generate_plan(plan: CompiledPlan) -> bool:
        return not plan.get_paths and not plan.map_specs

    def _generate_via_wrapped(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        generate_fn = getattr(self.model.wrapped, "generate", None)
        if not callable(generate_fn):
            raise AttributeError(
                f"Wrapped model {type(self.model.wrapped).__name__} does not define generate()."
            )
        device = self._primary_device()
        batch_input_ids = cast(torch.Tensor, move_tensors_to(input_ids, device))
        batch_attention = default_attention_mask(
            attention_mask,
            like=batch_input_ids,
            device=device,
        )
        generate_kwargs: dict[str, Any] = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            **cast(dict[str, Any], move_tensors_to(dict(kwargs), device)),
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
        with self._scheduled(
            "generate",
            batch_size=int(batch_input_ids.shape[0]),
            batch_tokens=int(batch_attention.sum().item()),
        ):
            with torch.inference_mode():
                output = cast(
                    torch.Tensor,
                    generate_fn(**generate_kwargs),
                )
        return output.to(device=input_ids.device)

    def _generate_direct_batched(
        self,
        plan: CompiledPlan,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1.")
        source_device = input_ids.device
        device = self._primary_device()
        batch_input_ids = cast(torch.Tensor, move_tensors_to(input_ids, device))
        batch_attention = default_attention_mask(
            attention_mask,
            like=batch_input_ids,
            device=device,
        )
        extra_kwargs = cast(dict[str, Any], move_tensors_to(dict(kwargs), device))
        sampling = SamplingConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        use_cache = callable(getattr(self.model.wrapped, "prepare_inputs_for_generation", None))
        eos_ids = eos_token_ids(self.model.wrapped)
        pad_token_id = self._pad_token_id()
        batch_size = int(batch_input_ids.shape[0])
        with self._scheduled(
            "generate",
            batch_size=batch_size,
            batch_tokens=int(batch_attention.sum().item()),
        ):
            with torch.inference_mode():
                prepared = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention,
                    "use_cache": use_cache,
                    **extra_kwargs,
                }
                output = self.model(
                    get=list(plan.get_proxies),
                    map=plan.map_dict,
                    **filter_supported_kwargs(self.model.wrapped, prepared),
                )
                cache = getattr(
                    output._model_output if isinstance(output, Output) else output,
                    "past_key_values",
                    None,
                )
                logits = extract_last_token_logits(output)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
                steps: list[torch.Tensor] = []
                for step_idx in range(max_new_tokens):
                    next_token = sample_next_token(logits, sampling)
                    if finished.any():
                        filler = torch.full_like(next_token, pad_token_id)
                        next_token = torch.where(finished.unsqueeze(-1), filler, next_token)
                    steps.append(next_token)
                    if eos_ids:
                        for idx in range(batch_size):
                            if not bool(finished[idx]) and contains_eos(
                                next_token[idx : idx + 1], eos_ids
                            ):
                                finished[idx] = True
                    if step_idx == max_new_tokens - 1 or bool(finished.all()):
                        break
                    batch_attention = torch.cat(
                        [
                            batch_attention,
                            torch.ones(
                                (batch_size, 1),
                                dtype=batch_attention.dtype,
                                device=device,
                            ),
                        ],
                        dim=-1,
                    )
                    if use_cache:
                        prepared = self._prepare_inputs_for_generation(
                            input_ids=next_token,
                            attention_mask=batch_attention,
                            cache=cache,
                            extra_kwargs=extra_kwargs,
                        )
                    else:
                        full_tokens = torch.cat([batch_input_ids, *steps], dim=-1)
                        prepared = {
                            "input_ids": full_tokens,
                            "attention_mask": batch_attention,
                            **extra_kwargs,
                        }
                    output = self.model(
                        get=list(plan.get_proxies),
                        map=plan.map_dict,
                        **filter_supported_kwargs(self.model.wrapped, prepared),
                    )
                    cache = getattr(
                        output._model_output if isinstance(output, Output) else output,
                        "past_key_values",
                        cache,
                    )
                    logits = extract_last_token_logits(output)
        generated = (
            torch.cat(steps, dim=-1)
            if steps
            else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )
        return torch.cat([batch_input_ids, generated], dim=-1).to(device=source_device)

    def _trim_generated_tokens(
        self,
        generated: torch.Tensor,
        eos_ids: set[int],
    ) -> torch.Tensor:
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("Expected generated tokens with shape [1, steps].")
        if not eos_ids:
            return generated
        values = generated.view(-1).tolist()
        for idx, token_id in enumerate(values):
            if int(token_id) in eos_ids:
                return generated[:, : idx + 1]
        return generated

    @staticmethod
    def _merge_batch_kwargs(
        batch: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        overlap = set(batch).intersection(kwargs)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate batched kwargs: {joined}.")
        return {**batch, **kwargs}

    @staticmethod
    def _split_plan_result(result: PlanResult, *, batch_size: int) -> list[PlanResult]:
        logits: list[torch.Tensor | None]
        if isinstance(result.logits, torch.Tensor) and result.logits.shape[0] == batch_size:
            logits = [result.logits[idx : idx + 1] for idx in range(batch_size)]
        else:
            logits = [result.logits for _ in range(batch_size)]
        token_ids: list[torch.Tensor | None]
        if isinstance(result.token_ids, torch.Tensor) and result.token_ids.shape[0] == batch_size:
            token_ids = [result.token_ids[idx : idx + 1] for idx in range(batch_size)]
        else:
            token_ids = [result.token_ids for _ in range(batch_size)]
        activations: list[dict[str, Any]] = []
        for idx in range(batch_size):
            item: dict[str, Any] = {}
            for path, value in result.activations.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 1
                    and value.shape[0] == batch_size
                ):
                    item[path] = value[idx : idx + 1]
                else:
                    item[path] = value
            activations.append(item)
        return [
            PlanResult(
                activations=activations[idx],
                logits=logits[idx],
                token_ids=token_ids[idx],
                completed_forward=result.completed_forward,
                metadata=dict(result.metadata),
            )
            for idx in range(batch_size)
        ]


def _batch_size_from_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    return 1
