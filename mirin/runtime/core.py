"""Local runtime for compiled collection, batching, and generation."""

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
from ..output import (
    GenerateOutput,
    Output,
    generate_output_from_path_activations,
    generate_output_from_value,
    merge_generate_outputs,
    output_from_path_activations,
)
from ..requests import RequestBatch, normalize_requests
from .collector import Collector
from .plans import CompiledPlan, OutputPolicyLike, SiteLike, compile_plan
from .prefill import collect_batch
from .results import PlanResult
from .scheduler import (
    AdmissionEstimate,
    QueueMetrics,
    ResourceLedger,
    SchedulerConfig,
    estimate_admission,
)
from .util import (
    contains_eos,
    default_attention_mask,
    eos_token_ids,
    extract_last_token_logits,
    filter_supported_kwargs,
    gpu_stats,
    model_dtype,
    move_tensors_to,
    prompt_tokens_from_mapping,
    to_cpu,
    to_cpu_dict,
)


@dataclass(slots=True)
class _OpStats:
    calls: int = 0
    errors: int = 0
    total_ns: int = 0
    inflight: int = 0
    peak_inflight: int = 0


class _RuntimeCore:
    """Shared local runtime used by `mirin.Model`."""

    def __init__(
        self,
        wrapped: torch.nn.Module | str | Model,
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
        gpu_fraction: float = 0.9,
        cpu_fraction: float = 0.8,
        **load_kwargs: Any,
    ) -> None:
        if isinstance(wrapped, Model):
            if rename is not None or tokenizer is not None or load_kwargs:
                raise TypeError(
                    "_RuntimeCore(Model) does not accept rename=, tokenizer=, or loading kwargs."
                )
            self._model = wrapped
        else:
            if attn_backend is not None and "attn_implementation" not in load_kwargs:
                load_kwargs["attn_implementation"] = attn_backend
            self._model = Model(wrapped, rename=rename, tokenizer=tokenizer, **load_kwargs)
        if device is not None:
            self._model.wrapped.to(device)

        self._plans: dict[str, CompiledPlan] = {}
        self._collectors: dict[str, Collector] = {}
        self._stats: dict[str, _OpStats] = defaultdict(_OpStats)
        self._queues: dict[str, QueueMetrics] = defaultdict(QueueMetrics)
        self._metrics_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._resource_lock = threading.Lock()
        self._resources = ResourceLedger()
        self._last_request_type = ""
        self._last_admission: dict[str, Any] | None = None
        self._started_ns = time.perf_counter_ns()
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

        from .memory import RuntimeCapacity

        self._capacity = RuntimeCapacity.detect(
            self._model.wrapped,
            self._primary_device(),
            gpu_fraction=gpu_fraction,
            cpu_fraction=cpu_fraction,
            max_kv_cache_bytes=self._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=self._scheduler.max_activation_capture_bytes,
            prefill_token_budget=self._scheduler.prefill_token_budget,
            decode_max_batch_tokens=self._scheduler.decode_max_batch_tokens,
            collect_token_budget=self._scheduler.collect_token_budget,
        )
        self._scheduler.max_kv_cache_bytes = self._capacity.kv_cache_bytes
        self._scheduler.max_activation_capture_bytes = self._capacity.activation_capture_bytes
        self._scheduler.prefill_token_budget = self._capacity.max_prefill_tokens
        self._scheduler.decode_max_batch_tokens = self._capacity.max_decode_batch_tokens
        self._scheduler.collect_token_budget = self._capacity.collect_token_budget

    @property
    def capacity(self) -> Any:
        return self._capacity

    @property
    def budget(self) -> Any:
        return self._capacity

    def close(self) -> None:
        with self._state_lock:
            self._collectors.clear()
            self._plans.clear()
        with self._resource_lock:
            self._resources = ResourceLedger()

    def compile(
        self,
        *,
        get: Sequence[SiteLike] | SiteLike | None = None,
        mapping: Mapping[SiteLike, Any] | None = None,
        output: OutputPolicyLike = None,
    ) -> CompiledPlan:
        with self._track("compile"):
            plan = compile_plan(self._model, get=get, mapping=mapping, output=output)
            with self._state_lock:
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
        call_args = move_tensors_to(tuple(args), self._primary_device())
        call_kwargs = cast(dict[str, Any], move_tensors_to(dict(kwargs), self._primary_device()))
        chunked_batches = self._auto_chunk_batch(compiled, call_kwargs)
        if not call_args and chunked_batches is not None:
            return _concat_plan_results([self.call(compiled, **chunk) for chunk in chunked_batches])
        batch_size, batch_tokens = _batch_metrics_from_call(call_args, call_kwargs)
        estimate = self._estimate_call(compiled, args=call_args, kwargs=call_kwargs)
        with self._scheduled(
            "call",
            estimate=estimate,
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            cpu_bytes=estimate.activation_bytes if compiled.output.activations_to_cpu else 0,
        ):
            result = self._execute_plan(compiled, args=call_args, kwargs=call_kwargs)
            self._record_physical_batch(
                "call",
                batch_size=batch_size,
                batch_tokens=batch_tokens,
                context_tokens=batch_tokens,
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
            **filter_supported_kwargs(self._model.wrapped, batch_kwargs),
        )
        return self._split_plan_result(result, batch_size=len(normalized.rows))

    def open_collector(
        self,
        *,
        plan: CompiledPlan | str | None = None,
        use_cache: bool = False,
        stop_at_last_get: bool | None = None,
        token_budget: int | None = None,
        activation_budget_bytes: int | None = None,
        activation_output: str | None = None,
        pin_memory: bool = False,
        mmap_path: str | None = None,
    ) -> Collector:
        compiled = self._resolve_plan(plan)
        if stop_at_last_get is None:
            stop_at_last_get = (
                bool(compiled.get_proxies) and not compiled.output.logits and not compiled.map_dict
            )
        if stop_at_last_get and compiled.map_dict:
            raise ValueError("Collector fast path does not support map=.")
        if activation_output is None:
            activation_output = "cpu" if compiled.output.activations_to_cpu else "gpu"
        if activation_output not in {"gpu", "cpu", "mmap"}:
            raise ValueError("Collector activation_output must be 'gpu', 'cpu', or 'mmap'.")
        if activation_output == "mmap" and not mmap_path:
            raise ValueError("Collector activation_output='mmap' requires mmap_path.")
        if activation_output != "mmap" and mmap_path is not None:
            raise ValueError("Collector mmap_path requires activation_output='mmap'.")
        if token_budget is None:
            token_budget = self._scheduler.collect_token_budget
        collector = Collector(
            id=uuid.uuid4().hex,
            plan=compiled,
            use_cache=use_cache,
            stop_at_last_get=stop_at_last_get,
            token_budget=token_budget,
            activation_budget_bytes=activation_budget_bytes,
            activation_output=cast(Any, activation_output),
            pin_memory=pin_memory,
            mmap_path=mmap_path,
            runtime=self,
        )
        with self._state_lock:
            self._collectors[collector.id] = collector
        return collector

    def collect_batch(
        self,
        collector: Collector | str,
        batch: Mapping[str, Any],
    ) -> PlanResult:
        state = self._resolve_collector(collector)
        return collect_batch(self, state, batch)

    def collect_many(
        self,
        collector: Collector | str,
        requests: Sequence[Any],
        **kwargs: Any,
    ) -> list[PlanResult]:
        state = self._resolve_collector(collector)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=False,
            pad_side="right",
        )
        result = self.collect_batch(state, self._merge_batch_kwargs(normalized.batch, kwargs))
        return self._split_plan_result(result, batch_size=len(normalized.rows))

    def close_collector(self, collector: Collector | str) -> None:
        state = self._resolve_collector(collector)
        with self._state_lock:
            self._collectors.pop(state.id, None)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        plan: CompiledPlan | str | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        capture: str = "all",
        **kwargs: Any,
    ) -> torch.Tensor | GenerateOutput:
        compiled = self._resolve_plan(plan)
        if self._is_plain_generate_plan(compiled):
            return self._generate_via_wrapped(
                plan=compiled,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                **kwargs,
            )
        result = self._generate_direct_batched(
            compiled,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            capture=capture,
            **kwargs,
        )
        if not compiled.output.activations:
            assert isinstance(result.sequences, torch.Tensor)
            return result.sequences
        return self._generate_output_from_result(result)

    def generate_many(
        self,
        requests: Sequence[Any],
        /,
        *,
        plan: CompiledPlan | str | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        capture: str = "all",
        **kwargs: Any,
    ) -> GenerateOutput:
        compiled = self._resolve_plan(plan)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=True,
            pad_side="left",
        )
        prompt_lengths = [int(row["input_ids"].shape[-1]) for row in normalized.rows]
        if len(set(prompt_lengths)) > 1:
            outputs: list[GenerateOutput | None] = [None] * len(normalized.rows)
            groups: dict[int, list[int]] = {}
            for idx, prompt_length in enumerate(prompt_lengths):
                groups.setdefault(prompt_length, []).append(idx)
            for indices in groups.values():
                batch_input_ids = torch.cat(
                    [normalized.rows[idx]["input_ids"] for idx in indices],
                    dim=0,
                )
                batch_attention = torch.cat(
                    [normalized.rows[idx]["attention_mask"] for idx in indices],
                    dim=0,
                )
                result = self._generate_direct_batched(
                    compiled,
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    capture=capture,
                    **kwargs,
                )
                split = self._split_plan_result(result, batch_size=len(indices))
                for row_idx, item in zip(indices, split, strict=True):
                    outputs[row_idx] = self._generate_output_from_result(item)
            return merge_generate_outputs(cast(list[GenerateOutput], outputs))
        output = self._generate_direct_batched(
            compiled,
            input_ids=normalized.batch["input_ids"],
            attention_mask=normalized.batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            capture=capture,
            **kwargs,
        )
        if compiled.output.activations:
            output.metadata["left_padded"] = True
            split = self._split_plan_result(output, batch_size=len(normalized.rows))
            return merge_generate_outputs(
                [self._generate_output_from_result(item, left_padded=True) for item in split]
            )
        sequences = cast(torch.Tensor, output.sequences)
        generated = sequences[:, normalized.batch["input_ids"].shape[-1] :]
        eos_ids = eos_token_ids(self._model.wrapped)
        outputs: list[GenerateOutput] = []
        for idx, prompt in enumerate(normalized.rows):
            row_generated = self._trim_generated_tokens(generated[idx : idx + 1], eos_ids)
            outputs.append(
                generate_output_from_path_activations(
                    torch.cat([prompt["input_ids"], row_generated], dim=-1),
                    row_generated,
                    {},
                    prompt_length=int(prompt["input_ids"].shape[-1]),
                    generated_length=int(row_generated.shape[-1]),
                )
            )
        return merge_generate_outputs(outputs)

    def stats(self) -> dict[str, Any]:
        with self._metrics_lock:
            total_calls = sum(entry.calls for entry in self._stats.values())
            total_errors = sum(entry.errors for entry in self._stats.values())
            total_time_ns = sum(entry.total_ns for entry in self._stats.values())
            peak_inflight = max((entry.peak_inflight for entry in self._stats.values()), default=0)
            inflight = sum(entry.inflight for entry in self._stats.values())
            queued = sum(entry.current_depth for entry in self._queues.values())
            queue_peak = max((entry.peak_depth for entry in self._queues.values()), default=0)
            queue_wait_ns = sum(entry.total_queue_wait_ns for entry in self._queues.values())
            service_ns = sum(entry.total_service_ns for entry in self._queues.values())
            queues = {name: entry.snapshot() for name, entry in self._queues.items()}
            last_request_type = self._last_request_type
            last_admission = self._last_admission
        with self._resource_lock:
            gpu_cap, cpu_cap, kv_cap, activation_cap = self._resource_limits()
            resources = self._resources.snapshot(
                gpu_capacity_bytes=gpu_cap,
                cpu_capacity_bytes=cpu_cap,
                kv_capacity_bytes=kv_cap,
                activation_capacity_bytes=activation_cap,
            )
        with self._state_lock:
            active_collectors = len(self._collectors)
        mean_request_ms = 0.0 if total_calls == 0 else (total_time_ns / total_calls) / 1e6
        uptime_ns = max(time.perf_counter_ns() - self._started_ns, 1)
        return {
            "queued_requests": queued,
            "queue_peak": queue_peak,
            "requests_served": total_calls,
            "request_errors": total_errors,
            "mean_request_ms": mean_request_ms,
            "mean_queue_wait_ms": 0.0 if total_calls == 0 else (queue_wait_ns / total_calls) / 1e6,
            "scheduler_utilization": min(service_ns / uptime_ns, 1.0),
            "last_request_type": last_request_type,
            "active_collectors": active_collectors,
            "last_admission": last_admission,
            "queues": queues,
            "resources": resources,
            "capacity": self.capacity.snapshot(),
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
            with self._state_lock:
                return self._plans[plan]
        except KeyError as exc:
            raise KeyError(f"Unknown plan id {plan!r}.") from exc

    def _resolve_collector(self, collector: Collector | str) -> Collector:
        if isinstance(collector, Collector):
            return collector
        try:
            with self._state_lock:
                return self._collectors[collector]
        except KeyError as exc:
            raise KeyError(f"Unknown collector id {collector!r}.") from exc

    def _primary_device(self) -> torch.device:
        device = self._model.device
        if isinstance(device, tuple):
            return device[0]
        return device

    def _estimate_request(
        self,
        queue: str,
        plan: CompiledPlan,
        *,
        batch_size: int,
        prompt_tokens_per_request: int,
        decode_tokens_per_request: int,
    ) -> AdmissionEstimate:
        return estimate_admission(
            queue=queue,
            wrapped=self._model.wrapped,
            plan=plan,
            dtype=model_dtype(self._model.wrapped),
            batch_size=batch_size,
            prompt_tokens_per_request=prompt_tokens_per_request,
            decode_tokens_per_request=decode_tokens_per_request,
            bucket_multiple=self._scheduler.decode_bucket_multiple,
            max_kv_cache_bytes=self._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=self._scheduler.max_activation_capture_bytes,
        )

    def _estimate_collection(
        self,
        plan: CompiledPlan,
        *,
        batch: Mapping[str, Any],
        activation_budget_bytes: int | None = None,
    ) -> AdmissionEstimate:
        activation_cap = activation_budget_bytes or self._scheduler.max_activation_capture_bytes
        batch_size = _batch_size_from_mapping(batch)
        prompt_tokens_total = prompt_tokens_from_mapping(batch)
        # Per-request token count for KV reservation. With padded contiguous
        # batches, KV reserves max_seq_len per request; use the average as a
        # cheap proxy.
        per_request = prompt_tokens_total // max(batch_size, 1)
        return estimate_admission(
            queue="collect_batch",
            wrapped=self._model.wrapped,
            plan=plan,
            dtype=model_dtype(self._model.wrapped),
            batch_size=batch_size,
            prompt_tokens_per_request=per_request,
            decode_tokens_per_request=0,
            bucket_multiple=self._scheduler.decode_bucket_multiple,
            max_kv_cache_bytes=self._scheduler.max_kv_cache_bytes,
            max_activation_capture_bytes=activation_cap,
        )

    @contextmanager
    def _scheduled(
        self,
        op: str,
        *,
        estimate: AdmissionEstimate | None = None,
        batch_size: int = 1,
        batch_tokens: int = 0,
        cpu_bytes: int = 0,
    ) -> Iterator[None]:
        queued_at = time.perf_counter_ns()
        kv_bytes = 0
        activation_bytes = 0
        reserved_gpu_bytes = 0
        reserved_cpu_bytes = 0
        is_cuda = self._primary_device().type == "cuda"
        with self._metrics_lock:
            queue = self._queues[op]
            queue.enqueued += 1
            queue.current_depth += 1
            queue.peak_depth = max(queue.peak_depth, queue.current_depth)
        if estimate is not None:
            kv_bytes = estimate.kv_cache_bytes
            activation_bytes = estimate.activation_bytes
            requested_gpu_bytes = kv_bytes + activation_bytes if is_cuda else 0
            requested_cpu_bytes = cpu_bytes + (0 if is_cuda else kv_bytes + activation_bytes)
            with self._resource_lock:
                gpu_cap, cpu_cap, kv_cap, activation_cap = self._resource_limits()
                reason = estimate.reason if not estimate.admitted else None
                if reason is None:
                    reason = self._resources.try_reserve(
                        kv_bytes=kv_bytes,
                        activation_bytes=activation_bytes,
                        gpu_bytes=requested_gpu_bytes,
                        cpu_bytes=requested_cpu_bytes,
                        gpu_capacity_bytes=gpu_cap,
                        cpu_capacity_bytes=cpu_cap,
                        kv_capacity_bytes=kv_cap,
                        activation_capacity_bytes=activation_cap,
                    )
                    if reason is None:
                        reserved_gpu_bytes = requested_gpu_bytes
                        reserved_cpu_bytes = requested_cpu_bytes
                self._last_admission = {
                    **estimate.snapshot(),
                    **self._resources.snapshot(
                        gpu_capacity_bytes=gpu_cap,
                        cpu_capacity_bytes=cpu_cap,
                        kv_capacity_bytes=kv_cap,
                        activation_capacity_bytes=activation_cap,
                    ),
                    "requested_gpu_bytes": requested_gpu_bytes,
                    "requested_cpu_bytes": requested_cpu_bytes,
                    "reason": reason,
                }
            if reason is not None:
                with self._metrics_lock:
                    queue = self._queues[op]
                    queue.record_reject(reason)
                    queue.current_depth = max(queue.current_depth - 1, 0)
                raise MemoryError(f"{op} rejected by admission control: {reason}.")
        started_at = time.perf_counter_ns()
        with self._metrics_lock:
            queue = self._queues[op]
            queue.current_depth = max(queue.current_depth - 1, 0)
            queue.started += 1
            queue.total_queue_wait_ns += started_at - queued_at
        with self._track(op):
            try:
                yield
            finally:
                if kv_bytes or activation_bytes or reserved_gpu_bytes or reserved_cpu_bytes:
                    with self._resource_lock:
                        self._resources.release(
                            kv_bytes=kv_bytes,
                            activation_bytes=activation_bytes,
                            gpu_bytes=reserved_gpu_bytes,
                            cpu_bytes=reserved_cpu_bytes,
                        )
                with self._metrics_lock:
                    queue = self._queues[op]
                    queue.completed += 1
                    queue.total_service_ns += time.perf_counter_ns() - started_at

    def _estimate_call(
        self,
        plan: CompiledPlan,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AdmissionEstimate:
        batch_size, prompt_tokens_total = _batch_metrics_from_call(args, kwargs)
        per_request = prompt_tokens_total // max(batch_size, 1)
        return self._estimate_request(
            "call",
            plan,
            batch_size=batch_size,
            prompt_tokens_per_request=per_request,
            decode_tokens_per_request=0,
        )

    def _estimate_generate(
        self,
        plan: CompiledPlan,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> AdmissionEstimate:
        batch_size = int(input_ids.shape[0]) if input_ids.ndim >= 2 else 1
        # Per-request prompt tokens. With padded contiguous KV cache we'll
        # reserve max_seq_len per request, so use the longest prompt in the
        # batch (worst case) for the estimate.
        per_request_prompt = int(attention_mask.sum(dim=-1).max().item())
        return self._estimate_request(
            "generate",
            plan,
            batch_size=batch_size,
            prompt_tokens_per_request=per_request_prompt,
            decode_tokens_per_request=max_new_tokens,
        )

    def _auto_chunk_batch(
        self,
        plan: CompiledPlan,
        batch: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
            return None
        if int(input_ids.shape[0]) <= 1 or self.capacity.device_capacity_bytes <= 0:
            return None
        max_batch = self.capacity.max_batch_size(
            plan,
            seq_len=int(input_ids.shape[-1]),
            bucket_multiple=self._scheduler.decode_bucket_multiple,
        )
        if max_batch >= int(input_ids.shape[0]):
            return None
        from .memory import auto_chunk

        extra_tensors = {key: value for key, value in batch.items() if key != "input_ids"}
        return auto_chunk(input_ids, max_batch=max_batch, extra_tensors=extra_tensors)

    def _resource_limits(self) -> tuple[int | None, int | None, int | None, int | None]:
        gpu_capacity_bytes = (
            self.capacity.gpu_capacity_bytes if self.capacity.gpu_capacity_bytes > 0 else None
        )
        cpu_capacity_bytes = (
            self.capacity.cpu_capacity_bytes if self.capacity.cpu_capacity_bytes > 0 else None
        )
        return (
            gpu_capacity_bytes,
            cpu_capacity_bytes,
            self.capacity.kv_cache_bytes,
            self.capacity.activation_capture_bytes,
        )

    def _record_physical_batch(
        self,
        op: str,
        *,
        batch_size: int,
        batch_tokens: int,
        context_tokens: int | None = None,
    ) -> None:
        with self._metrics_lock:
            self._queues[op].record_physical_batch(
                batch_size=batch_size,
                batch_tokens=batch_tokens,
                context_tokens=context_tokens,
            )

    def _record_split(
        self,
        op: str,
        *,
        reason: str,
        original_items: int,
        produced_chunks: int,
    ) -> None:
        with self._metrics_lock:
            self._queues[op].record_split(
                reason=reason,
                original_items=original_items,
                produced_chunks=produced_chunks,
            )

    @contextmanager
    def _track(self, op: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        with self._metrics_lock:
            entry = self._stats[op]
            entry.inflight += 1
            entry.peak_inflight = max(entry.peak_inflight, entry.inflight)
            self._last_request_type = op
        if get_debug() >= 1:
            print(f"[ti] runtime: op={op}")
        try:
            yield
        except Exception:
            with self._metrics_lock:
                self._stats[op].errors += 1
            raise
        finally:
            elapsed_ns = time.perf_counter_ns() - started
            with self._metrics_lock:
                current = self._stats[op]
                current.calls += 1
                current.total_ns += elapsed_ns
                current.inflight = max(current.inflight - 1, 0)

    def _execute_plan(
        self,
        plan: CompiledPlan,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        grad: bool = False,
        stop_at_last_get: bool = False,
    ) -> Any:
        return self._model._execute_now(
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            get_proxies=list(plan.get_proxies),
            map_proxies=dict(plan.map_dict),
            grad=grad,
            stop_at_last_get=stop_at_last_get,
            always_output=True,
        )

    def _extract_activations(self, plan: CompiledPlan, result: Any) -> dict[str, Any]:
        if not isinstance(result, Output):
            return {}
        return {path: result[proxy] for path, proxy in zip(plan.get_paths, plan.get_proxies, strict=True)}

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
            raw_logits = extract_last_token_logits(result) if logits_slice else self._extract_logits(result)
            logits = to_cpu(raw_logits, enabled=logits_to_cpu)
        completed_forward = result.completed_forward if isinstance(result, Output) else True
        return PlanResult(
            activations=activations if plan.output.activations else {},
            logits=logits,
            completed_forward=completed_forward,
        )

    def _extract_logits(self, result: Any) -> torch.Tensor:
        model_output = result._model_output if isinstance(result, Output) else result
        logits = getattr(model_output, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits
        if isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), torch.Tensor):
            return cast(torch.Tensor, model_output["logits"])
        if isinstance(model_output, torch.Tensor):
            return model_output
        raise TypeError(f"Cannot extract logits from {type(model_output).__name__}.")

    def _normalize_requests(
        self,
        requests: Sequence[Any],
        *,
        add_generation_prompt: bool,
        pad_side: str,
    ) -> RequestBatch:
        return normalize_requests(
            requests,
            tokenizer=self._model.tokenizer,
            add_generation_prompt=add_generation_prompt,
            pad_side=pad_side,
            pad_token_id=self._pad_token_id(),
            owner="Model",
        )

    def _pad_token_id(self) -> int:
        tokenizer = self._model.tokenizer
        if tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if isinstance(pad_token_id, int):
                return pad_token_id
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                return eos_token_id
            if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
                return int(eos_token_id[0])
        config = getattr(self._model.wrapped, "config", None)
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
        generate_fn = getattr(self._model.wrapped, "generate", None)
        if not callable(generate_fn):
            raise AttributeError(
                f"Wrapped model {type(self._model.wrapped).__name__} does not define generate()."
            )
        device = self._primary_device()
        batch_input_ids = cast(torch.Tensor, move_tensors_to(input_ids, device))
        batch_attention = default_attention_mask(attention_mask, like=batch_input_ids, device=device)
        estimate = self._estimate_generate(
            plan,
            input_ids=batch_input_ids,
            attention_mask=batch_attention,
            max_new_tokens=max_new_tokens,
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
            estimate=estimate,
            batch_size=int(batch_input_ids.shape[0]),
            batch_tokens=int(batch_attention.sum().item()),
        ):
            with torch.inference_mode():
                output = cast(torch.Tensor, generate_fn(**generate_kwargs))
            self._record_physical_batch(
                "generate",
                batch_size=int(batch_input_ids.shape[0]),
                batch_tokens=int(batch_attention.sum().item()),
                context_tokens=int(batch_attention.sum().item()),
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
        capture: str,
        **kwargs: Any,
    ) -> PlanResult:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")
        if capture not in {"all", "generated"}:
            raise ValueError("generate(..., capture=...) must be 'all' or 'generated'.")
        source_device = input_ids.device
        device = self._primary_device()
        batch_input_ids = cast(torch.Tensor, move_tensors_to(input_ids, device))
        batch_attention = default_attention_mask(attention_mask, like=batch_input_ids, device=device)
        if max_new_tokens == 0:
            prompt_lengths = [int(value) for value in batch_attention.sum(dim=-1).tolist()]
            prompt_width = int(batch_input_ids.shape[-1])
            activations: dict[str, Any] = {}
            logits: torch.Tensor | None = None
            if plan.output.activations or plan.output.logits:
                estimate = self._estimate_generate(
                    plan,
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                    max_new_tokens=0,
                )
                prepared = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention,
                    "use_cache": False,
                    **cast(dict[str, Any], move_tensors_to(dict(kwargs), device)),
                }
                with self._scheduled(
                    "generate",
                    estimate=estimate,
                    batch_size=int(batch_input_ids.shape[0]),
                    batch_tokens=int(batch_attention.sum().item()),
                ):
                    with torch.inference_mode():
                        output = self._execute_plan(
                            plan,
                            kwargs=filter_supported_kwargs(self._model.wrapped, prepared),
                        )
                    self._record_physical_batch(
                        "generate",
                        batch_size=int(batch_input_ids.shape[0]),
                        batch_tokens=int(batch_input_ids.numel()),
                        context_tokens=int(batch_attention.sum().item()),
                    )
                if plan.output.activations:
                    activations = self._extract_activations(plan, output)
                if plan.output.logits:
                    logits = self._extract_logits(output)
            empty_generated = batch_input_ids.new_empty((int(batch_input_ids.shape[0]), 0))
            return PlanResult(
                activations=activations,
                logits=logits,
                token_ids=empty_generated.to(device=source_device),
                sequences=batch_input_ids.to(device=source_device),
                prompt_length=prompt_lengths[0] if len(prompt_lengths) == 1 else None,
                metadata={
                    "capture": capture,
                    "prompt_lengths": prompt_lengths,
                    "generated_lengths": [0 for _ in prompt_lengths],
                    "generated_length": 0 if len(prompt_lengths) == 1 else None,
                    "prompt_width": prompt_width,
                },
            )
        # Tier-1 fast path: only run admission estimation when at least one
        # budget is set. The estimate itself does an attention_mask.sum().item()
        # CUDA sync; with admission off (the default) we can skip it entirely.
        admission_active = (
            self._scheduler.max_kv_cache_bytes is not None
            or self._scheduler.max_activation_capture_bytes is not None
        )
        estimate = (
            self._estimate_generate(
                plan,
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                max_new_tokens=max_new_tokens,
            )
            if admission_active
            else None
        )
        prompt_attention = batch_attention.clone()
        extra_kwargs = cast(dict[str, Any], move_tensors_to(dict(kwargs), device))
        requested_use_cache = extra_kwargs.pop("use_cache", None)
        use_cache = callable(getattr(self._model.wrapped, "prepare_inputs_for_generation", None))
        if requested_use_cache is not None:
            use_cache = use_cache and bool(requested_use_cache)
        eos_ids = eos_token_ids(self._model.wrapped)
        # Build a device-resident eos tensor once so the per-step EOS check
        # can be a vectorised compare instead of B `.item()` syncs per step.
        eos_tensor = (
            torch.tensor(sorted(eos_ids), dtype=batch_input_ids.dtype, device=device)
            if eos_ids
            else None
        )
        requested_pad_token_id = extra_kwargs.pop("pad_token_id", None)
        pad_token_id = self._pad_token_id() if requested_pad_token_id is None else int(requested_pad_token_id)
        batch_size = int(batch_input_ids.shape[0])
        # Pre-allocate the attention mask for the full prompt + decode horizon
        # so the per-step grow is a slice instead of a torch.cat reallocation.
        prompt_len = int(batch_attention.shape[-1])
        total_len = prompt_len + max_new_tokens
        mask_buffer = batch_attention.new_empty((batch_size, total_len))
        mask_buffer[:, :prompt_len].copy_(batch_attention)
        if max_new_tokens > 0:
            mask_buffer[:, prompt_len:].fill_(1)
        # CPU-side cumulative context-token counter so we don't sync each
        # step just to record `batch_attention.sum().item()` for telemetry.
        prompt_token_total = int(prompt_attention.sum().item())
        context_tokens_cum = prompt_token_total
        generated_steps: list[torch.Tensor] = []
        generated_activations: dict[str, list[torch.Tensor]] = {}
        prompt_activations: dict[str, Any] = {}
        with self._scheduled(
            "generate",
            estimate=estimate,
            batch_size=batch_size,
            batch_tokens=prompt_token_total,
        ):
            with torch.inference_mode():
                prepared = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention,
                    "use_cache": use_cache,
                    **extra_kwargs,
                }
                output = self._execute_plan(
                    plan,
                    kwargs=filter_supported_kwargs(self._model.wrapped, prepared),
                )
                self._record_physical_batch(
                    "generate",
                    batch_size=batch_size,
                    batch_tokens=int(batch_input_ids.numel()),
                    context_tokens=context_tokens_cum,
                )
                if plan.output.activations:
                    prompt_activations = self._extract_activations(plan, output)
                cache = getattr(
                    output._model_output if isinstance(output, Output) else output,
                    "past_key_values",
                    None,
                )
                logits = extract_last_token_logits(output)
                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
                for step_idx in range(max_new_tokens):
                    next_token = _sample_next_token(
                        logits,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    if finished.any():
                        filler = torch.full_like(next_token, pad_token_id)
                        next_token = torch.where(finished.unsqueeze(-1), filler, next_token)
                    generated_steps.append(next_token)
                    if eos_tensor is not None:
                        # Vectorised EOS detection: no .item() syncs per request.
                        last_tok = next_token[:, 0] if next_token.ndim > 1 else next_token
                        is_eos = (last_tok.unsqueeze(-1) == eos_tensor).any(dim=-1)
                        finished = finished | is_eos
                    needs_trailing_forward = plan.output.activations
                    if step_idx == max_new_tokens - 1 and not needs_trailing_forward:
                        break
                    # When we're not extracting activations, we can break
                    # once every sequence has emitted EOS. That requires a
                    # bool(finished.all()) sync; avoid it on the activation-
                    # extraction hot path where we always run the full loop.
                    if not needs_trailing_forward and bool(finished.all()):
                        break
                    # Slice the pre-allocated mask buffer instead of cat.
                    # `.contiguous()` ensures the kernel doesn't fall back to a
                    # strided-mask path; the copy is small (B * cur_len bytes).
                    batch_attention = mask_buffer[:, : prompt_len + step_idx + 1].contiguous()
                    context_tokens_cum += batch_size
                    if use_cache:
                        prepared = _prepare_inputs_for_generation(
                            self._model.wrapped,
                            input_ids=next_token,
                            attention_mask=batch_attention,
                            cache=cache,
                            extra_kwargs=extra_kwargs,
                        )
                    else:
                        full_tokens = torch.cat([batch_input_ids, *generated_steps], dim=-1)
                        prepared = {
                            "input_ids": full_tokens,
                            "attention_mask": batch_attention,
                            **extra_kwargs,
                        }
                    output = self._execute_plan(
                        plan,
                        kwargs=filter_supported_kwargs(self._model.wrapped, prepared),
                    )
                    self._record_physical_batch(
                        "generate",
                        batch_size=batch_size,
                        batch_tokens=int(next_token.numel()) if use_cache else int(full_tokens.numel()),
                        context_tokens=context_tokens_cum,
                    )
                    if plan.output.activations:
                        for path, value in self._extract_activations(plan, output).items():
                            if isinstance(value, torch.Tensor):
                                generated_activations.setdefault(path, []).append(value[:, -1:].detach())
                    cache = getattr(
                        output._model_output if isinstance(output, Output) else output,
                        "past_key_values",
                        cache,
                    )
                    if step_idx == max_new_tokens - 1:
                        break
                    if not needs_trailing_forward and bool(finished.all()):
                        break
                    logits = extract_last_token_logits(output)
        generated = (
            torch.cat(generated_steps, dim=-1)
            if generated_steps
            else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )
        activations: dict[str, Any] = {}
        prompt_lengths = [int(value) for value in prompt_attention.sum(dim=-1).tolist()]
        generated_lengths = _generated_lengths(generated, eos_ids)
        if plan.output.activations:
            for path in sorted(set(prompt_activations).union(generated_activations)):
                prompt_value = prompt_activations.get(path)
                generated_value = _stack_generated_activation_slices(
                    generated_activations.get(path, []),
                    batch_size=batch_size,
                    device=device,
                )
                if capture == "generated":
                    activations[path] = generated_value
                elif isinstance(prompt_value, torch.Tensor) and isinstance(generated_value, torch.Tensor):
                    activations[path] = torch.cat([prompt_value, generated_value], dim=1)
                else:
                    activations[path] = prompt_value if prompt_value is not None else generated_value
        return PlanResult(
            activations=activations,
            token_ids=generated.to(device=source_device),
            sequences=torch.cat([batch_input_ids, generated], dim=-1).to(device=source_device),
            completed_forward=True,
            metadata={
                "capture": capture,
                "prompt_lengths": prompt_lengths,
                "generated_lengths": generated_lengths,
                "prompt_width": int(prompt_attention.shape[-1]),
                "left_padded": False,
            },
        )

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
        sequences: list[torch.Tensor | None]
        if isinstance(result.sequences, torch.Tensor) and result.sequences.shape[0] == batch_size:
            sequences = [result.sequences[idx : idx + 1] for idx in range(batch_size)]
        else:
            sequences = [result.sequences for _ in range(batch_size)]
        prompt_lengths = cast(list[int] | None, result.metadata.get("prompt_lengths"))
        generated_lengths = cast(list[int] | None, result.metadata.get("generated_lengths"))
        prompt_width = cast(int | None, result.metadata.get("prompt_width"))
        capture = cast(str | None, result.metadata.get("capture"))
        left_padded = bool(result.metadata.get("left_padded", False))
        activations: list[dict[str, Any]] = []
        for idx in range(batch_size):
            item: dict[str, Any] = {}
            for path, value in result.activations.items():
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == batch_size:
                    row_value = value[idx : idx + 1]
                    if (
                        capture is not None
                        and prompt_lengths is not None
                        and generated_lengths is not None
                        and prompt_width is not None
                        and row_value.ndim >= 2
                    ):
                        row_value = _trim_generate_activation(
                            row_value,
                            prompt_length=prompt_lengths[idx],
                            generated_length=generated_lengths[idx],
                            prompt_width=prompt_width,
                            capture=capture,
                            left_padded=left_padded,
                        )
                    item[path] = row_value
                else:
                    item[path] = value
            activations.append(item)
        return [
            PlanResult(
                activations=activations[idx],
                logits=logits[idx],
                token_ids=(
                    _trim_generated_token_ids(token_ids[idx], generated_length=generated_lengths[idx])
                    if token_ids[idx] is not None and generated_lengths is not None
                    else token_ids[idx]
                ),
                sequences=(
                    _trim_generate_sequence(
                        sequences[idx],
                        prompt_length=prompt_lengths[idx],
                        generated_length=generated_lengths[idx],
                        prompt_width=prompt_width,
                        left_padded=left_padded,
                    )
                    if sequences[idx] is not None and prompt_lengths is not None and generated_lengths is not None
                    else sequences[idx]
                ),
                prompt_length=(prompt_lengths[idx] if prompt_lengths is not None else None),
                completed_forward=result.completed_forward,
                metadata={
                    **dict(result.metadata),
                    "prompt_lengths": None,
                    "generated_lengths": None,
                    "generated_length": (generated_lengths[idx] if generated_lengths is not None else None),
                    "prompt_width": prompt_width,
                },
            )
            for idx in range(batch_size)
        ]

    def _generate_output_from_result(
        self,
        result: PlanResult,
        *,
        left_padded: bool = False,
    ) -> GenerateOutput:
        prompt_lengths = cast(list[int] | None, result.metadata.get("prompt_lengths"))
        generated_lengths = cast(list[int] | None, result.metadata.get("generated_lengths"))
        generated_length = cast(int | None, result.metadata.get("generated_length"))
        prompt_width = cast(int | None, result.metadata.get("prompt_width"))
        capture = cast(str | None, result.metadata.get("capture"))
        activations = dict(result.activations)
        if (
            prompt_lengths is not None
            and generated_lengths is not None
            and prompt_width is not None
            and capture is not None
            and isinstance(result.sequences, torch.Tensor)
            and result.sequences.shape[0] == 1
        ):
            activations = {
                path: _trim_generate_activation(
                    value,
                    prompt_length=prompt_lengths[0],
                    generated_length=generated_lengths[0],
                    prompt_width=prompt_width,
                    capture=capture,
                    left_padded=left_padded or bool(result.metadata.get("left_padded", False)),
                )
                if isinstance(value, torch.Tensor) and value.ndim >= 2
                else value
                for path, value in activations.items()
            }
        return generate_output_from_path_activations(
            cast(torch.Tensor, result.sequences),
            cast(torch.Tensor, result.token_ids),
            activations,
            prompt_length=(
                prompt_lengths[0]
                if prompt_lengths is not None and len(prompt_lengths) == 1
                else result.prompt_length if result.prompt_length is not None else prompt_lengths
            ),
            generated_length=(
                generated_lengths[0]
                if generated_lengths is not None and len(generated_lengths) == 1
                else generated_length if generated_length is not None else generated_lengths
            ),
        )


def _merge_plan_results(results: Sequence[PlanResult]) -> PlanResult:
    if not results:
        return PlanResult()
    first = results[0]
    paths = set().union(*(result.activations for result in results))
    activations = {
        path: _merge_plan_value([result.activations[path] for result in results if path in result.activations])
        for path in sorted(paths)
    }
    metadata = dict(first.metadata)
    if any(result.metadata != first.metadata for result in results[1:]):
        metadata = {}
    return PlanResult(
        activations=activations,
        logits=_merge_plan_value([result.logits for result in results]),
        token_ids=_merge_plan_value([result.token_ids for result in results]),
        sequences=_merge_plan_value([result.sequences for result in results]),
        completed_forward=all(result.completed_forward for result in results),
        metadata=metadata,
    )


def _merge_plan_value(values: Sequence[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if all(isinstance(value, torch.Tensor) for value in present):
        return torch.cat(cast(list[torch.Tensor], present), dim=0)
    return present[0]


def _stack_generated_activation_slices(
    values: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if values:
        return torch.cat(list(values), dim=1)
    return torch.empty((batch_size, 0), dtype=torch.float32, device=device)


def _generated_lengths(generated: torch.Tensor, eos_ids: set[int]) -> list[int]:
    if generated.ndim != 2:
        raise ValueError("Expected generated tokens with shape [batch, steps].")
    if not eos_ids:
        return [int(generated.shape[1])] * int(generated.shape[0])
    lengths: list[int] = []
    for row in generated:
        row_values = row.detach().cpu().tolist()
        length = len(row_values)
        for idx, token_id in enumerate(row_values):
            if int(token_id) in eos_ids:
                length = idx + 1
                break
        lengths.append(length)
    return lengths


def _trim_generated_token_ids(
    value: torch.Tensor | None,
    *,
    generated_length: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value[:, :generated_length]


def _trim_generate_sequence(
    value: torch.Tensor | None,
    *,
    prompt_length: int,
    generated_length: int,
    prompt_width: int | None,
    left_padded: bool,
) -> torch.Tensor | None:
    if value is None:
        return None
    if prompt_width is None:
        return value[:, : prompt_length + generated_length]
    prompt_slice = value[:, :prompt_width]
    generated_slice = value[:, prompt_width : prompt_width + generated_length]
    if left_padded:
        prompt_slice = prompt_slice[:, prompt_width - prompt_length :]
    else:
        prompt_slice = prompt_slice[:, :prompt_length]
    return torch.cat([prompt_slice, generated_slice], dim=1)


def _trim_generate_activation(
    value: torch.Tensor,
    *,
    prompt_length: int,
    generated_length: int,
    prompt_width: int,
    capture: str,
    left_padded: bool,
) -> torch.Tensor:
    if capture == "generated":
        return value[:, :generated_length]
    prompt_slice = value[:, :prompt_width]
    generated_slice = value[:, prompt_width : prompt_width + generated_length]
    if left_padded:
        prompt_slice = prompt_slice[:, prompt_width - prompt_length :]
    else:
        prompt_slice = prompt_slice[:, :prompt_length]
    return torch.cat([prompt_slice, generated_slice], dim=1)


def _prepare_inputs_for_generation(
    wrapped: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any | None,
    extra_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
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
    prepare_kwargs.pop("use_cache", None)
    prepare_kwargs.pop("pad_token_id", None)
    cache_position = _cache_position(cache, input_ids)
    if cache_position is not None and prepare_kwargs.get("cache_position") is None:
        prepare_kwargs["cache_position"] = cache_position
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
    if cache_position is not None and normalized.get("cache_position") is None:
        normalized["cache_position"] = cache_position
    return normalized


def _cache_position(cache: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    if cache is None:
        start = 0
    elif callable(getattr(cache, "get_seq_length", None)):
        start = int(cache.get_seq_length())
    else:
        return None
    return torch.arange(start, start + input_ids.shape[-1], device=input_ids.device)


def _concat_plan_results(results: Sequence[PlanResult]) -> PlanResult:
    if not results:
        return PlanResult()
    activations: dict[str, Any] = {}
    for path in results[0].activations:
        values = [result.activations[path] for result in results if path in result.activations]
        if values and all(isinstance(value, torch.Tensor) for value in values):
            activations[path] = torch.cat(cast(list[torch.Tensor], values), dim=0)
        elif values:
            activations[path] = values[-1]
    logits: torch.Tensor | None = None
    logit_values = [result.logits for result in results if isinstance(result.logits, torch.Tensor)]
    if logit_values:
        logits = torch.cat(cast(list[torch.Tensor], logit_values), dim=0)
    metadata: dict[str, Any] = {}
    for result in results:
        metadata.update(result.metadata)
    return PlanResult(
        activations=activations,
        logits=logits,
        completed_forward=all(result.completed_forward for result in results),
        metadata=metadata,
    )


def _batch_size_from_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    return 1


def _batch_metrics_from_call(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[int, int]:
    input_ids = kwargs.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim >= 2:
            return int(input_ids.shape[0]), int(input_ids.shape[0] * input_ids.shape[1])
        if input_ids.ndim == 1:
            return 1, int(input_ids.shape[0])
    if args and isinstance(args[0], torch.Tensor):
        tensor = cast(torch.Tensor, args[0])
        if tensor.ndim >= 2:
            return int(tensor.shape[0]), int(tensor.shape[0] * tensor.shape[1])
        if tensor.ndim == 1:
            return 1, int(tensor.shape[0])
    return 1, 0


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits
    if temperature > 0:
        scaled = scaled / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.shape[-1])
        values, indices = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sample = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, sample)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
