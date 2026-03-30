"""Unit tests for decode-engine lifecycle edge cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

import tinyinterp as ti
from tinyinterp.server.cache import FallbackCacheAdapter
from tinyinterp.server.decode_engine import DecodeFamily

from .helpers import FakeLlamaModel


@dataclass(slots=True)
class _BrokenBatchAdapter:
    name: str = "broken_batch"

    def supports_cache(self, _cache: Any) -> bool:
        return True

    def supports_batched_decode(self) -> bool:
        return True

    def append_cache(self, _existing: Any, _new_cache: Any, _wrapped: torch.nn.Module) -> Any:
        raise RuntimeError("boom")

    def compact_cache(
        self,
        cache: Any,
        _keep_indices: list[int],
        _wrapped: torch.nn.Module,
    ) -> Any:
        return cache


def test_close_session_ignores_stale_slot_ownership() -> None:
    server = ti.Server(FakeLlamaModel())
    engine = server.runtime._decode_engine
    session_a = server.open_session(cache="none")
    session_b = server.open_session(cache="none")
    replacement = server.open_session(cache="none")
    key = ("fp", "none", "fallback", "4", "4")
    family = DecodeFamily(
        key=key,
        adapter=FallbackCacheAdapter(),
        plan_fingerprint="fp",
        cache_mode="none",
        current_length=4,
        decode_bucket_len=4,
        sessions=[session_a, replacement],
        cache=None,
        attention_mask=torch.ones(2, 4, dtype=torch.long),
    )
    engine._families[key] = family
    session_a.family_key = key
    session_a.slot_index = 0
    session_b.family_key = key
    session_b.slot_index = 1
    replacement.family_key = key
    replacement.slot_index = 1

    engine.close_session(session_b)

    assert family.sessions == [session_a, replacement]
    assert session_b.family_key is None
    assert session_b.slot_index is None
    assert replacement.family_key == key
    assert replacement.slot_index == 1


def test_attach_single_session_rolls_back_new_family_on_append_failure() -> None:
    server = ti.Server(FakeLlamaModel())
    engine = server.runtime._decode_engine
    session = server.open_session()
    session.prompt_length = 4
    session.current_length = 4
    session.decode_bucket_len = 4
    session.cache = object()

    with pytest.raises(RuntimeError, match="boom"):
        engine._attach_single_session(
            session,
            _BrokenBatchAdapter(),
            torch.ones(1, 4, dtype=torch.long),
        )

    assert engine._families == {}
    assert session.family_key is None
    assert session.slot_index is None
    assert session.cache is not None


def test_advance_family_requires_attention_mask() -> None:
    server = ti.Server(FakeLlamaModel())
    engine = server.runtime._decode_engine
    session = server.open_session(cache="none")
    session.current_length = 4
    session.pending_input_ids = torch.tensor([[5]], dtype=torch.long)
    family = DecodeFamily(
        key=("fp", "none", "fallback", "4", "4"),
        adapter=FallbackCacheAdapter(),
        plan_fingerprint=session.plan.fingerprint,
        cache_mode=session.cache_mode,
        current_length=4,
        decode_bucket_len=4,
        sessions=[session],
        cache=None,
        attention_mask=None,
    )

    with pytest.raises(RuntimeError, match="attention mask"):
        engine._advance_family(family)


def test_advance_family_validates_logits_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    server = ti.Server(FakeLlamaModel())
    engine = server.runtime._decode_engine
    sessions = [server.open_session(cache="none"), server.open_session(cache="none")]
    for session in sessions:
        session.current_length = 4
        session.pending_input_ids = torch.tensor([[5]], dtype=torch.long)
    family = DecodeFamily(
        key=("fp", "none", "fallback", "4", "4"),
        adapter=FallbackCacheAdapter(),
        plan_fingerprint=sessions[0].plan.fingerprint,
        cache_mode=sessions[0].cache_mode,
        current_length=4,
        decode_bucket_len=4,
        sessions=sessions,
        cache=None,
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    monkeypatch.setattr(
        server.runtime,
        "_execute_plan",
        lambda _plan, *, kwargs: {"logits": torch.zeros(1, 7)},
    )

    with pytest.raises(ValueError, match="logits batch"):
        engine._advance_family(family)
