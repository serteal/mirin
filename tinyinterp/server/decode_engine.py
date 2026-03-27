"""Decode engine with persistent family state for session workloads."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from ..output import Output
from .cache import CacheAdapter, select_cache_adapter
from .results import PlanResult
from .runtime import (
    contains_eos,
    eos_token_ids,
    extract_last_token_logits,
    filter_supported_kwargs,
    split_activation_dict,
    split_batch_tensor,
    to_cpu,
    to_cpu_dict,
)
from .sessions import SamplingConfig, Session, sample_next_token

if TYPE_CHECKING:
    from .inference import Server


@dataclass(slots=True)
class DecodeFamily:
    """One persistent batched decode family."""

    key: tuple[str, ...]
    adapter: CacheAdapter
    plan_fingerprint: str
    cache_mode: str
    prompt_length: int
    decode_bucket_len: int
    sessions: list[Session] = field(default_factory=list)
    cache: Any | None = None
    attention_mask: torch.Tensor | None = None


class DecodeEngine:
    """Own session state after prefill and advance decode workloads."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self._families: dict[tuple[str, ...], DecodeFamily] = {}

    def prefill_key(self, session: Session) -> tuple[str, ...]:
        prompt_length = str(session.prompt_length or session.current_length)
        bucket = str(
            session.decode_bucket_len or session.max_total_tokens or session.current_length
        )
        return (
            session.plan.fingerprint,
            session.cache_mode,
            prompt_length,
            bucket,
        )

    def register_prefill_result(
        self,
        session: Session,
        *,
        raw_result: Any,
        plan_result: PlanResult,
        attention_mask: torch.Tensor,
        adapter: CacheAdapter | None = None,
        shared_cache: Any | None = None,
        shared_index: int | None = None,
    ) -> PlanResult:
        logits = extract_last_token_logits(raw_result)
        if shared_index is not None:
            session.last_logits = logits[shared_index : shared_index + 1]
        else:
            session.last_logits = logits
        session.finished = False
        session.pending_input_ids = None
        session.prompt_length = int(attention_mask.sum().item())
        session.current_length = session.prompt_length
        session.input_ids = None
        session.generated_cpu.clear()
        if session.use_hf_cache:
            active_adapter = adapter or select_cache_adapter(session.cache, session.cache_mode)
            if shared_cache is None:
                self._attach_single_session(session, active_adapter, attention_mask)
        else:
            session.attention_mask = self._make_attention_row(
                decode_bucket_len=session.decode_bucket_len or session.current_length,
                prompt_length=session.prompt_length,
            )
        return self.server._result_for_session_plan(session, plan_result)

    def register_prefilled_family(
        self,
        sessions: Sequence[Session],
        *,
        shared_cache: Any,
        adapter: CacheAdapter,
        attention_mask: torch.Tensor,
        raw_result: Any,
    ) -> None:
        if not sessions:
            return
        first = sessions[0]
        key = (
            first.plan.fingerprint,
            first.cache_mode,
            adapter.name,
            str(first.prompt_length),
            str(first.decode_bucket_len or first.current_length),
        )
        family = self._families.get(key)
        block = self._make_attention_block(
            batch_size=len(sessions),
            decode_bucket_len=first.decode_bucket_len or first.current_length,
            prompt_length=first.prompt_length,
            device=attention_mask.device,
        )
        if family is None:
            family = DecodeFamily(
                key=key,
                adapter=adapter,
                plan_fingerprint=first.plan.fingerprint,
                cache_mode=first.cache_mode,
                prompt_length=first.prompt_length,
                decode_bucket_len=first.decode_bucket_len or first.current_length,
                cache=shared_cache,
                attention_mask=block,
            )
            self._families[key] = family
        else:
            family.cache = adapter.append_cache(
                family.cache, shared_cache, self.server.model.wrapped
            )
            family.attention_mask = (
                block
                if family.attention_mask is None
                else torch.cat([family.attention_mask, block], dim=0)
            )
        logits = extract_last_token_logits(raw_result)
        start = len(family.sessions)
        for idx, session in enumerate(sessions):
            session.family_key = key
            session.slot_index = start + idx
            session.cache = None
            session.attention_mask = None
            session.last_logits = logits[idx : idx + 1]
            family.sessions.append(session)

    def close_session(self, session: Session) -> None:
        if session.family_key is None:
            return
        family = self._families.get(session.family_key)
        if family is None:
            return
        if session.slot_index is None or session.slot_index >= len(family.sessions):
            session.family_key = None
            session.slot_index = None
            return
        keep_indices = [idx for idx, item in enumerate(family.sessions) if item is not session]
        family.cache = family.adapter.compact_cache(
            family.cache, keep_indices, self.server.model.wrapped
        )
        if family.attention_mask is not None:
            family.attention_mask = (
                family.attention_mask[keep_indices].clone() if keep_indices else None
            )
        family.sessions = [family.sessions[idx] for idx in keep_indices]
        for idx, item in enumerate(family.sessions):
            item.slot_index = idx
        session.family_key = None
        session.slot_index = None
        if not family.sessions:
            self._families.pop(family.key, None)

    def decode(
        self,
        sessions: Sequence[Session],
        *,
        max_new_tokens: int,
        do_sample: bool | None,
        temperature: float | None,
        top_k: int | None,
    ) -> list[PlanResult]:
        resolved = list(sessions)
        if not resolved:
            return []
        outputs = [PlanResult(session_id=session.id) for session in resolved]
        token_chunks: list[list[torch.Tensor]] = [[] for _ in resolved]
        latest_activations: dict[str, dict[str, Any]] = {}
        eos_ids = eos_token_ids(self.server.model.wrapped)

        for _ in range(max_new_tokens):
            active = [session for session in resolved if not session.finished]
            if not active:
                break
            advanced = self._advance_pending_sessions(active)
            latest_activations.update(
                {sid: result.activations for sid, result in advanced.items() if result.activations}
            )
            for idx, session in enumerate(resolved):
                if session.finished:
                    continue
                if session.last_logits is None:
                    raise RuntimeError("decode() requires prefill() before sampling.")
                sampling = self._override_sampling(
                    session.sampling,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                )
                next_token = sample_next_token(session.last_logits, sampling)
                token_chunks[idx].append(next_token)
                session.generated_cpu.extend(
                    int(token) for token in next_token.view(-1).detach().cpu().tolist()
                )
                outputs[idx].logits = (
                    to_cpu(
                        session.last_logits,
                        enabled=session.plan.output.logits_to_cpu,
                    )
                    if session.plan.output.logits
                    else None
                )
                outputs[idx].activations = latest_activations.get(session.id, {})
                if contains_eos(next_token, eos_ids):
                    session.finished = True
                    session.pending_input_ids = None
                else:
                    session.pending_input_ids = next_token
            self._compact_finished_families()

        for idx, result in enumerate(outputs):
            if token_chunks[idx]:
                result.token_ids = torch.cat(token_chunks[idx], dim=-1)
            result.completed_forward = True
        return outputs

    def _attach_single_session(
        self,
        session: Session,
        adapter: CacheAdapter,
        attention_mask: torch.Tensor,
    ) -> None:
        if not adapter.supports_batched_decode() or session.cache is None:
            session.attention_mask = self._make_attention_row(
                decode_bucket_len=session.decode_bucket_len or session.current_length,
                prompt_length=session.prompt_length,
            )
            return
        key = (
            session.plan.fingerprint,
            session.cache_mode,
            adapter.name,
            str(session.prompt_length),
            str(session.decode_bucket_len or session.current_length),
        )
        family = self._families.get(key)
        if family is None:
            family = DecodeFamily(
                key=key,
                adapter=adapter,
                plan_fingerprint=session.plan.fingerprint,
                cache_mode=session.cache_mode,
                prompt_length=session.prompt_length,
                decode_bucket_len=session.decode_bucket_len or session.current_length,
                sessions=[],
                cache=None,
                attention_mask=None,
            )
            self._families[key] = family
        family.cache = adapter.append_cache(family.cache, session.cache, self.server.model.wrapped)
        session.cache = None
        session.family_key = key
        session.slot_index = len(family.sessions)
        family.sessions.append(session)
        row = self._make_attention_row(
            decode_bucket_len=family.decode_bucket_len,
            prompt_length=session.prompt_length,
            device=attention_mask.device,
        )
        family.attention_mask = (
            row
            if family.attention_mask is None
            else torch.cat(
                [family.attention_mask, row],
                dim=0,
            )
        )
        session.attention_mask = None

    def _advance_pending_sessions(self, sessions: Sequence[Session]) -> dict[str, PlanResult]:
        pending = [session for session in sessions if session.pending_input_ids is not None]
        if not pending:
            return {}
        grouped: dict[tuple[str, ...], list[Session]] = defaultdict(list)
        for session in pending:
            key = session.family_key or (
                session.plan.fingerprint,
                session.cache_mode,
                "fallback",
                str(session.prompt_length),
                str(session.decode_bucket_len or session.current_length),
            )
            grouped[key].append(session)

        outputs: dict[str, PlanResult] = {}
        for key, group in grouped.items():
            family = self._families.get(key)
            if family is not None and family.adapter.supports_batched_decode():
                active_family = [item for item in family.sessions if not item.finished]
                if len(group) == len(active_family) and all(
                    item.pending_input_ids is not None for item in active_family
                ):
                    outputs.update(self._advance_family(family))
                    continue
            for session in group:
                outputs.update(self._advance_single(session))
        return outputs

    def _advance_family(self, family: DecodeFamily) -> dict[str, PlanResult]:
        ordered = [session for session in family.sessions if not session.finished]
        first = ordered[0]
        pending = torch.cat(
            [cast(torch.Tensor, session.pending_input_ids) for session in ordered],
            dim=0,
        )
        next_length = first.current_length + pending.shape[-1]
        assert family.attention_mask is not None
        family.attention_mask[: len(ordered), first.current_length : next_length] = 1
        attention_view = family.attention_mask[: len(ordered), :next_length]
        prepared = self.server._prepare_inputs_for_generation(
            input_ids=pending,
            attention_mask=attention_view,
            cache=family.cache,
            extra_kwargs=first.extra_kwargs,
        )
        with torch.inference_mode():
            result = self.server.model(
                get=list(first.plan.get_proxies),
                map=first.plan.map_dict,
                **filter_supported_kwargs(self.server.model.wrapped, prepared),
            )
        if isinstance(result, Output):
            model_output = result._model_output
        else:
            model_output = result
        family.cache = getattr(model_output, "past_key_values", family.cache)
        split_logits = split_batch_tensor(extract_last_token_logits(result), len(ordered))
        split_acts = split_activation_dict(
            self.server._extract_activations(first.plan, result),
            len(ordered),
        )
        outputs: dict[str, PlanResult] = {}
        for idx, session in enumerate(ordered):
            session.current_length = next_length
            session.last_logits = split_logits[idx]
            session.pending_input_ids = None
            outputs[session.id] = PlanResult(
                session_id=session.id,
                logits=to_cpu(
                    split_logits[idx],
                    enabled=session.plan.output.logits and session.plan.output.logits_to_cpu,
                )
                if session.plan.output.logits
                else None,
                activations=to_cpu_dict(
                    split_acts[idx],
                    enabled=session.plan.output.activations_to_cpu,
                )
                if session.plan.output.activations
                else {},
            )
        return outputs

    def _advance_single(self, session: Session) -> dict[str, PlanResult]:
        pending = cast(torch.Tensor, session.pending_input_ids)
        next_length = session.current_length + pending.shape[-1]
        if session.use_hf_cache:
            if session.attention_mask is None:
                session.attention_mask = self._make_attention_row(
                    decode_bucket_len=session.decode_bucket_len or next_length,
                    prompt_length=session.current_length,
                    device=pending.device,
                )
            session.attention_mask[:, session.current_length : next_length] = 1
            attention_view = session.attention_mask[:, :next_length]
            prepared = self.server._prepare_inputs_for_generation(
                input_ids=pending,
                attention_mask=attention_view,
                cache=session.cache,
                extra_kwargs=session.extra_kwargs,
            )
        else:
            pending_cpu = pending.view(-1).detach().cpu().tolist()
            prefix = (
                session.generated_cpu[: -len(pending_cpu)] if pending_cpu else session.generated_cpu
            )
            full_tokens = torch.tensor(
                [session.history_cpu + prefix + pending_cpu],
                device=pending.device,
                dtype=torch.long,
            )
            if session.attention_mask is None:
                session.attention_mask = self._make_attention_row(
                    decode_bucket_len=session.decode_bucket_len or full_tokens.shape[-1],
                    prompt_length=session.current_length,
                    device=pending.device,
                )
            session.attention_mask[:, session.current_length : next_length] = 1
            attention_view = session.attention_mask[:, :next_length]
            prepared = {
                "input_ids": full_tokens,
                "attention_mask": attention_view,
                **session.extra_kwargs,
            }
        with torch.inference_mode():
            result = self.server.model(
                get=list(session.plan.get_proxies),
                map=session.plan.map_dict,
                **filter_supported_kwargs(self.server.model.wrapped, prepared),
            )
        if isinstance(result, Output):
            model_output = result._model_output
        else:
            model_output = result
        session.cache = getattr(model_output, "past_key_values", session.cache)
        session.current_length = next_length
        session.last_logits = extract_last_token_logits(result)
        session.pending_input_ids = None
        plan_result = self.server._build_plan_result(
            session.plan,
            result,
            logits_slice=True,
            activations_to_cpu=session.plan.output.activations_to_cpu,
            logits_to_cpu=False,
        )
        return {session.id: self.server._result_for_session_plan(session, plan_result)}

    def _compact_finished_families(self) -> None:
        for key, family in list(self._families.items()):
            keep_indices = [
                idx for idx, session in enumerate(family.sessions) if not session.finished
            ]
            if len(keep_indices) == len(family.sessions):
                continue
            family.cache = family.adapter.compact_cache(
                family.cache,
                keep_indices,
                self.server.model.wrapped,
            )
            family.attention_mask = (
                family.attention_mask[keep_indices].clone()
                if family.attention_mask is not None and keep_indices
                else None
            )
            family.sessions = [family.sessions[idx] for idx in keep_indices]
            for idx, session in enumerate(family.sessions):
                session.slot_index = idx
            if not family.sessions:
                self._families.pop(key, None)

    def _make_attention_row(
        self,
        *,
        decode_bucket_len: int,
        prompt_length: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        row = torch.zeros(
            (1, max(decode_bucket_len, prompt_length)),
            dtype=torch.long,
            device=device or self.server._primary_device(),
        )
        row[:, :prompt_length] = 1
        return row

    def _make_attention_block(
        self,
        *,
        batch_size: int,
        decode_bucket_len: int,
        prompt_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        block = torch.zeros(
            (batch_size, max(decode_bucket_len, prompt_length)), dtype=torch.long, device=device
        )
        block[:, :prompt_length] = 1
        return block

    def _override_sampling(
        self,
        base: SamplingConfig,
        *,
        do_sample: bool | None,
        temperature: float | None,
        top_k: int | None,
    ) -> SamplingConfig:
        return SamplingConfig(
            do_sample=base.do_sample if do_sample is None else do_sample,
            temperature=base.temperature if temperature is None else temperature,
            top_k=base.top_k if top_k is None else top_k,
        )
