"""In-process inference server for HuggingFace-style CausalLM execution."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
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
class _RemoteGradHandle:
    id: str
    activations: dict[str, torch.Tensor]
    logits: torch.Tensor | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _output_from_plan_result(result: PlanResult) -> Output:
    return output_from_path_activations(
        result,
        result.activations,
        completed_forward=result.completed_forward,
    )


class _RuntimeCore:
    """Shared lowered runtime used by local and deployed mirin execution."""

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
        self._gpu_fraction = gpu_fraction
        self._cpu_fraction = cpu_fraction
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
        self._runtime_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._metrics_lock = threading.Lock()
        self._decode_engine = DecodeEngine(self)
        self._prefill_engine = PrefillEngine(self, self._decode_engine)
        self._budget: Any | None = None
        self._serve_shutdown = threading.Event()
        self._listen_socket: Any | None = None
        self._listen_path: str | None = None
        self._client_sockets: set[Any] = set()
        self._client_lock = threading.Lock()
        self._remote_value_count = 0
        self._remote_value_lock = threading.Lock()
        self._remote_grads: dict[str, _RemoteGradHandle] = {}
        self._remote_grad_count = 0
        self._remote_grad_lock = threading.Lock()

    @property
    def budget(self) -> Any:
        """Lazy-initialized memory budget for server scheduling heuristics."""
        if self._budget is None:
            from .memory import MemoryBudget

            self._budget = MemoryBudget(
                self._model.wrapped,
                self._primary_device(),
                gpu_fraction=self._gpu_fraction,
                cpu_fraction=self._cpu_fraction,
            )
        return self._budget

    def serve(self, sock_path: str = "/tmp/mirin.sock") -> None:
        """Start a persistent server on a Unix socket. Blocks forever."""
        from .remote import serve as _serve

        _serve(self, sock_path)

    def close(self) -> None:
        """Stop serving, close active clients, and drop runtime-owned state."""
        self._serve_shutdown.set()
        if self._listen_socket is not None:
            try:
                self._listen_socket.close()
            except OSError:
                pass
            self._listen_socket = None
        with self._client_lock:
            sockets = list(self._client_sockets)
            self._client_sockets.clear()
        for client in sockets:
            try:
                client.close()
            except OSError:
                pass
        with self._runtime_lock:
            with self._state_lock:
                self._collectors.clear()
                sessions = list(self._sessions.values())
                self._sessions.clear()
            for session in sessions:
                self._decode_engine.close_session(session)
        with self._remote_grad_lock:
            self._remote_grads.clear()
            self._remote_grad_count = 0

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
        with self._scheduled("call"):
            result = self._execute_plan(
                compiled,
                args=move_tensors_to(tuple(args), self._primary_device()),
                kwargs=cast(dict[str, Any], move_tensors_to(dict(kwargs), self._primary_device())),
            )
            return self._build_plan_result(
                compiled,
                result,
                logits_slice=False,
                activations_to_cpu=compiled.output.activations_to_cpu,
                logits_to_cpu=compiled.output.logits_to_cpu,
            )

    def call_grad(
        self,
        plan: CompiledPlan | str | None = None,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, PlanResult]:
        compiled = self._resolve_plan(plan)
        device = self._primary_device()
        grad_args = move_tensors_to(tuple(args), device)
        grad_kwargs = cast(dict[str, Any], move_tensors_to(dict(kwargs), device))
        with self._scheduled("call_grad"):
            result = self._execute_plan(
                compiled,
                args=grad_args,
                kwargs=grad_kwargs,
                grad=True,
            )
        plan_result = self._build_plan_result(
            compiled,
            result,
            logits_slice=False,
            activations_to_cpu=False,
            logits_to_cpu=False,
        )
        if not plan_result.activations and plan_result.logits is None:
            raise ValueError("Remote grad=True requires logits or captured activations.")
        if isinstance(plan_result.logits, torch.Tensor) and plan_result.logits.requires_grad:
            plan_result.logits.retain_grad()
        grad_id = uuid.uuid4().hex
        with self._remote_grad_lock:
            self._remote_grads[grad_id] = _RemoteGradHandle(
                id=grad_id,
                activations={
                    path: value
                    for path, value in plan_result.activations.items()
                    if isinstance(value, torch.Tensor)
                },
                logits=plan_result.logits,
                args=tuple(grad_args),
                kwargs=dict(grad_kwargs),
            )
            self._remote_grad_count += 1
        return grad_id, plan_result

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
        return self._split_plan_result(
            result,
            batch_size=len(normalized.rows),
        )

    def fetch_grad_value(self, grad_id: str, target: str | int) -> torch.Tensor:
        tensor = self._resolve_grad_target(grad_id, target)
        return to_cpu(tensor, enabled=True)

    def fetch_target_grad(self, grad_id: str, target: str | int) -> torch.Tensor | None:
        tensor = self._resolve_grad_target(grad_id, target)
        if not isinstance(tensor.grad, torch.Tensor):
            return None
        return to_cpu(tensor.grad, enabled=True)

    def fetch_input_grads(self, grad_id: str) -> dict[str, Any]:
        handle = self._resolve_grad_handle(grad_id)
        grads: dict[str, Any] = {}
        if handle.args:
            grads["args"] = tuple(self._grad_tree(value) for value in handle.args)
        for key, value in handle.kwargs.items():
            grads[key] = self._grad_tree(value)
        return grads

    def backward_grad(
        self,
        grad_id: str,
        target: str | int,
        gradient: torch.Tensor | None = None,
    ) -> None:
        tensor = self._resolve_grad_target(grad_id, target)
        grad_tensor = None
        if isinstance(gradient, torch.Tensor):
            grad_tensor = cast(torch.Tensor, move_tensors_to(gradient, tensor.device))
        if grad_tensor is None:
            if tensor.numel() != 1:
                raise ValueError("Gradient tensor is required for non-scalar remote backward().")
            tensor.backward()
            return
        tensor.backward(grad_tensor)

    def release_grad(self, grad_id: str) -> bool:
        with self._remote_grad_lock:
            handle = self._remote_grads.pop(grad_id, None)
            if handle is None:
                return False
            self._remote_grad_count = max(self._remote_grad_count - 1, 0)
            return True

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
            activations_to_cpu = compiled.output.activations_to_cpu
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
        with self._state_lock:
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
        **kwargs: Any,
    ) -> list[PlanResult]:
        state = self._resolve_collector(collector)
        normalized = self._normalize_requests(
            requests,
            add_generation_prompt=False,
            pad_side="right",
        )
        result = self.collect_batch(state, self._merge_batch_kwargs(normalized.batch, kwargs))
        return self._split_plan_result(
            result,
            batch_size=len(normalized.rows),
        )

    def close_collector(self, collector: Collector | str) -> None:
        state = self._resolve_collector(collector)
        with self._state_lock:
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
        if cache == "static" and not supports_static_cache_model(self._model.wrapped):
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
            and callable(getattr(self._model.wrapped, "prepare_inputs_for_generation", None)),
            max_total_tokens=max_total_tokens,
            max_new_tokens_hint=max_new_tokens_hint,
            cache=None,
        )
        with self._state_lock:
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
            raise ValueError("decode() expects at least one session.")
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
        with self._runtime_lock:
            state = self._resolve_session(session)
            self._decode_engine.close_session(state)
            with self._state_lock:
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
        capture: str = "all",
        **kwargs: Any,
    ) -> torch.Tensor | GenerateOutput:
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
        cache: str = "dynamic",
        max_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        capture: str = "all",
        **kwargs: Any,
    ) -> GenerateOutput:
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
        if not callable(getattr(self._model.wrapped, "prepare_inputs_for_generation", None)):
            lengths = {int(row["input_ids"].shape[-1]) for row in normalized.rows}
            if len(lengths) > 1:
                return merge_generate_outputs(
                    [
                        generate_output_from_value(
                            self.generate(
                                input_ids=row["input_ids"],
                                attention_mask=row["attention_mask"],
                                plan=compiled,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_k=top_k,
                                capture=capture,
                                **kwargs,
                            ),
                            prompt_length=int(row["input_ids"].shape[-1]),
                        )
                        for row in normalized.rows
                    ]
                )
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
            if compiled.output.activations:
                raise RuntimeError("Plain wrapped generation cannot return captured activations.")
        else:
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
                [
                    self._generate_output_from_result(
                        item,
                        left_padded=True,
                    )
                    for item in split
                ]
            )
        sequences = (
            output.sequences
            if isinstance(output, PlanResult)
            else cast(torch.Tensor, output)
        )
        assert isinstance(sequences, torch.Tensor)
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
            last_request_type = self._last_request_type
            last_admission = self._last_admission
            queues = {name: entry.snapshot() for name, entry in self._queues.items()}
        with self._state_lock:
            active_sessions = len(self._sessions)
            active_collectors = len(self._collectors)
        with self._client_lock:
            connected_clients = len(self._client_sockets)
        mean_request_ms = 0.0 if total_calls == 0 else (total_time_ns / total_calls) / 1e6
        uptime_ns = max(time.perf_counter_ns() - self._server_started_ns, 1)
        return {
            "connected_clients": connected_clients,
            "live_remote_values": self._remote_value_count,
            "live_remote_grads": self._remote_grad_count,
            "queued_requests": queued,
            "queue_peak": queue_peak,
            "requests_served": total_calls,
            "request_errors": total_errors,
            "mean_request_ms": mean_request_ms,
            "mean_queue_wait_ms": 0.0 if total_calls == 0 else (queue_wait_ns / total_calls) / 1e6,
            "scheduler_utilization": min(service_ns / uptime_ns, 1.0),
            "last_request_type": last_request_type,
            "active_sessions": active_sessions,
            "active_collectors": active_collectors,
            "last_admission": last_admission,
            "queues": queues,
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

    def _resolve_grad_handle(self, grad_id: str) -> _RemoteGradHandle:
        with self._remote_grad_lock:
            handle = self._remote_grads.get(grad_id)
        if handle is None:
            raise KeyError(f"Unknown grad handle {grad_id!r}.")
        return handle

    def _resolve_grad_target(self, grad_id: str, target: str | int) -> torch.Tensor:
        handle = self._resolve_grad_handle(grad_id)
        if target == "logits":
            if not isinstance(handle.logits, torch.Tensor):
                raise KeyError("Grad handle does not expose logits.")
            return handle.logits
        tensor = handle.activations.get(str(target))
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(f"Grad handle does not expose target {target!r}.")
        return tensor

    @staticmethod
    def _grad_tree(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if not value.requires_grad or not isinstance(value.grad, torch.Tensor):
                return None
            return to_cpu(value.grad, enabled=True)
        if isinstance(value, tuple):
            return tuple(_RuntimeCore._grad_tree(item) for item in value)
        if isinstance(value, list):
            return [_RuntimeCore._grad_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: _RuntimeCore._grad_tree(item) for key, item in value.items()}
        return None

    def _resolve_session(self, session: Session | str) -> Session:
        if isinstance(session, Session):
            return session
        try:
            with self._state_lock:
                return self._sessions[session]
        except KeyError as exc:
            raise KeyError(f"Unknown session id {session!r}.") from exc

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

    def _estimate_collection(
        self,
        plan: CompiledPlan,
        *,
        batch: Mapping[str, Any],
    ) -> AdmissionEstimate:
        return estimate_admission(
            queue="collect_batch",
            wrapped=self._model.wrapped,
            plan=plan,
            dtype=model_dtype(self._model.wrapped),
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
            wrapped=self._model.wrapped,
            plan=plan,
            dtype=model_dtype(self._model.wrapped),
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
        with self._metrics_lock:
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
        guard = self._guard_for_op(op)
        with guard if guard is not None else nullcontext():
            started_at = time.perf_counter_ns()
            with self._metrics_lock:
                queue = self._queues[op]
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
                    with self._metrics_lock:
                        queue = self._queues[op]
                        queue.completed += 1
                        queue.total_service_ns += time.perf_counter_ns() - started_at

    @contextmanager
    def _track(self, op: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        with self._metrics_lock:
            entry = self._stats[op]
            entry.inflight += 1
            entry.peak_inflight = max(entry.peak_inflight, entry.inflight)
            self._last_request_type = op
        if get_debug() >= 1:
            print(f"[ti] server: op={op}")
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

    def _guard_for_op(self, op: str) -> threading.Lock | None:
        if op == "decode":
            return self._runtime_lock
        return None

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

    def _prepare_inputs_for_generation(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Any | None,
        extra_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        wrapped = self._model.wrapped
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
    ) -> RequestBatch:
        return normalize_requests(
            requests,
            tokenizer=self._model.tokenizer,
            add_generation_prompt=add_generation_prompt,
            pad_side=pad_side,
            pad_token_id=self._pad_token_id(),
            owner="Server",
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
        capture: str,
        **kwargs: Any,
    ) -> PlanResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1.")
        if capture not in {"all", "generated"}:
            raise ValueError("generate(..., capture=...) must be 'all' or 'generated'.")
        source_device = input_ids.device
        device = self._primary_device()
        batch_input_ids = cast(torch.Tensor, move_tensors_to(input_ids, device))
        batch_attention = default_attention_mask(
            attention_mask,
            like=batch_input_ids,
            device=device,
        )
        prompt_attention = batch_attention.clone()
        extra_kwargs = cast(dict[str, Any], move_tensors_to(dict(kwargs), device))
        requested_use_cache = extra_kwargs.pop("use_cache", None)
        sampling = SamplingConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        use_cache = callable(getattr(self._model.wrapped, "prepare_inputs_for_generation", None))
        if requested_use_cache is not None:
            use_cache = use_cache and bool(requested_use_cache)
        eos_ids = eos_token_ids(self._model.wrapped)
        requested_pad_token_id = extra_kwargs.pop("pad_token_id", None)
        pad_token_id = self._pad_token_id() if requested_pad_token_id is None else int(
            requested_pad_token_id
        )
        batch_size = int(batch_input_ids.shape[0])
        generated_steps: list[torch.Tensor] = []
        generated_activations: dict[str, list[torch.Tensor]] = {}
        prompt_activations: dict[str, Any] = {}
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
                output = self._execute_plan(
                    plan,
                    kwargs=filter_supported_kwargs(self._model.wrapped, prepared),
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
                    next_token = sample_next_token(logits, sampling)
                    if finished.any():
                        filler = torch.full_like(next_token, pad_token_id)
                        next_token = torch.where(finished.unsqueeze(-1), filler, next_token)
                    generated_steps.append(next_token)
                    if eos_ids:
                        for idx in range(batch_size):
                            if not bool(finished[idx]) and contains_eos(
                                next_token[idx : idx + 1], eos_ids
                            ):
                                finished[idx] = True
                    needs_trailing_forward = plan.output.activations
                    if step_idx == max_new_tokens - 1 and not needs_trailing_forward:
                        break
                    if bool(finished.all()) and not needs_trailing_forward:
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
                    if plan.output.activations:
                        for path, value in self._extract_activations(plan, output).items():
                            if isinstance(value, torch.Tensor):
                                generated_activations.setdefault(path, []).append(
                                    value[:, -1:].detach()
                                )
                    cache = getattr(
                        output._model_output if isinstance(output, Output) else output,
                        "past_key_values",
                        cache,
                    )
                    if step_idx == max_new_tokens - 1 or bool(finished.all()):
                        break
                    logits = extract_last_token_logits(output)
        generated = (
            torch.cat(generated_steps, dim=-1)
            if generated_steps
            else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )
        activations: dict[str, Any] = {}
        if plan.output.activations:
            prompt_lengths = [int(value) for value in prompt_attention.sum(dim=-1).tolist()]
            generated_lengths = _generated_lengths(generated, eos_ids)
            for path in sorted(set(prompt_activations).union(generated_activations)):
                prompt_value = prompt_activations.get(path)
                generated_value = _stack_generated_activation_slices(
                    generated_activations.get(path, []),
                    batch_size=batch_size,
                    device=device,
                )
                if capture == "generated":
                    activations[path] = generated_value
                    continue
                if (
                    isinstance(prompt_value, torch.Tensor)
                    and isinstance(generated_value, torch.Tensor)
                ):
                    activations[path] = torch.cat([prompt_value, generated_value], dim=1)
                    continue
                activations[path] = prompt_value if prompt_value is not None else generated_value
        else:
            prompt_lengths = [int(value) for value in prompt_attention.sum(dim=-1).tolist()]
            generated_lengths = _generated_lengths(generated, eos_ids)
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
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 1
                    and value.shape[0] == batch_size
                ):
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
                    _trim_generated_token_ids(
                        token_ids[idx],
                        generated_length=generated_lengths[idx],
                    )
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
                    if sequences[idx] is not None
                    and prompt_lengths is not None
                    and generated_lengths is not None
                    else sequences[idx]
                ),
                prompt_length=(prompt_lengths[idx] if prompt_lengths is not None else None),
                completed_forward=result.completed_forward,
                metadata={
                    **dict(result.metadata),
                    "prompt_lengths": None,
                    "generated_lengths": None,
                    "generated_length": (
                        generated_lengths[idx] if generated_lengths is not None else None
                    ),
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
        path: _merge_plan_value(
            [result.activations[path] for result in results if path in result.activations]
        )
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
        session_id=first.session_id
        if all(result.session_id == first.session_id for result in results)
        else None,
        prompt_length=first.prompt_length
        if all(result.prompt_length == first.prompt_length for result in results)
        else None,
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
    """Return generated token counts, including EOS when it is emitted."""

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


def _trim_sequence_output(
    value: torch.Tensor | None,
    *,
    prompt_length: int,
    generated_length: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value[:, : prompt_length + generated_length]


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
        return _trim_sequence_output(
            value,
            prompt_length=prompt_length,
            generated_length=generated_length,
        )
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


def _cache_position(cache: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    if cache is None:
        start = 0
    elif callable(getattr(cache, "get_seq_length", None)):
        start = int(cache.get_seq_length())
    else:
        return None
    return torch.arange(start, start + input_ids.shape[-1], device=input_ids.device)


def _batch_size_from_mapping(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    return 1


class Server:
    """Deployment wrapper around the shared lowered mirin runtime."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "_runtime", _RuntimeCore(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_runtime":
            object.__setattr__(self, name, value)
            return
        setattr(self._runtime, name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self) + dir(self._runtime)))

    @property
    def runtime(self) -> _RuntimeCore:
        return self._runtime
