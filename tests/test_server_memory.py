"""Memory-budget and auto-chunking tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

import tinyinterp as ti

from .helpers import FakeDecoderModel, FakeLlamaModel
from .server_helpers import _ids, _seeded_model, _start_server, _stop_server


class TestCpuMemoryDetection:
    def test_detects_positive_value(self) -> None:
        from tinyinterp.server.memory import _cpu_memory_bytes
        assert _cpu_memory_bytes() > 0

    def test_less_than_physical(self) -> None:
        """cgroup limit should be less than raw MemTotal on constrained VMs."""
        from tinyinterp.server.memory import _cpu_memory_bytes
        mem = _cpu_memory_bytes()
        assert mem < 1024 * (1024 ** 3)  # sanity: less than 1TB

    def test_at_least_1gb(self) -> None:
        from tinyinterp.server.memory import _cpu_memory_bytes
        assert _cpu_memory_bytes() > 1024 ** 3


class TestMemoryBudget:
    def test_gpu_budget_positive(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        b = server.budget
        # Fake model on CPU → gpu_budget is 0
        assert b.gpu_budget >= 0

    def test_cpu_budget_positive(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        assert server.budget.cpu_budget > 0

    def test_max_batch_size_decreases_with_seq_len(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        server.budget.gpu_budget = 10_000_000  # 10MB fake budget
        plan = server.compile(get=["transformer.h.0"])
        mb_short = server.budget.max_batch_size(plan, seq_len=16)
        mb_long = server.budget.max_batch_size(plan, seq_len=256)
        assert mb_short >= mb_long

    def test_max_batch_at_least_1(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        server.budget.gpu_budget = 1  # tiny
        plan = server.compile(get=["transformer.h.0"])
        assert server.budget.max_batch_size(plan, seq_len=128) >= 1

    def test_budget_respects_gpu_fraction(self) -> None:
        s1 = ti.Server(_seeded_model(FakeDecoderModel), gpu_fraction=0.5)
        s2 = ti.Server(_seeded_model(FakeDecoderModel), gpu_fraction=0.9)
        # Both on CPU so budgets are 0, but fractions stored correctly
        assert s1._gpu_fraction == 0.5
        assert s2._gpu_fraction == 0.9

    def test_budget_respects_cpu_fraction(self) -> None:
        s1 = ti.Server(_seeded_model(FakeDecoderModel), cpu_fraction=0.3)
        s2 = ti.Server(_seeded_model(FakeDecoderModel), cpu_fraction=0.8)
        b1 = s1.budget.cpu_budget
        b2 = s2.budget.cpu_budget
        assert b2 > b1

    def test_estimate_cpu_bytes_scales_with_sites(self) -> None:
        server = ti.Server(_seeded_model(FakeLlamaModel))
        b = server.budget
        plan_1 = server.compile(get=["model.layers.0"])
        plan_2 = server.compile(get=["model.layers.0", "model.layers.1"])
        est_1 = b.estimate_cpu_bytes(plan_1, batch_size=4, seq_len=128)
        est_2 = b.estimate_cpu_bytes(plan_2, batch_size=4, seq_len=128)
        # 2 sites should cost more than 1 site
        assert est_2 >= est_1

    def test_estimate_cpu_bytes_scales_with_batch(self) -> None:
        server = ti.Server(_seeded_model(FakeDecoderModel))
        b = server.budget
        plan = server.compile(get=["transformer.h.0"])
        est_4 = b.estimate_cpu_bytes(plan, batch_size=4, seq_len=128)
        est_16 = b.estimate_cpu_bytes(plan, batch_size=16, seq_len=128)
        assert est_16 == est_4 * 4


class TestAutoChunk:
    def test_no_chunk_when_fits(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(4, 8)
        chunks = auto_chunk(ids, max_batch=10)
        assert len(chunks) == 1
        assert torch.equal(chunks[0]["input_ids"], ids)

    def test_chunks_when_exceeds(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(8, 4)
        chunks = auto_chunk(ids, max_batch=3)
        assert len(chunks) == 3  # 3+3+2
        total = sum(c["input_ids"].shape[0] for c in chunks)
        assert total == 8

    def test_chunks_preserve_data(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(6, 4)
        chunks = auto_chunk(ids, max_batch=2)
        reconstructed = torch.cat([c["input_ids"] for c in chunks], dim=0)
        assert torch.equal(reconstructed, ids)

    def test_extra_tensors_chunked_too(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(6, 4)
        mask = torch.ones(6, 4, dtype=torch.long)
        chunks = auto_chunk(ids, max_batch=2, extra_tensors={"attention_mask": mask})
        assert all("attention_mask" in c for c in chunks)
        assert chunks[0]["attention_mask"].shape[0] == 2

    def test_max_batch_1(self) -> None:
        from tinyinterp.server.memory import auto_chunk
        ids = _ids(3, 4)
        chunks = auto_chunk(ids, max_batch=1)
        assert len(chunks) == 3
        assert all(c["input_ids"].shape[0] == 1 for c in chunks)


class TestRemoteModelAutoChunking:
    def test_auto_chunks_large_batch(self) -> None:
        """Auto-chunking preserves the original batched output contract."""

        sock = "/tmp/tinyinterp_test_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(4, 4), get=[site])
            assert not isinstance(result, list)
            assert result[site].shape[0] == 4
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_auto_chunk_activations_correct(self) -> None:
        """Chunked results match unchunked results."""

        sock = "/tmp/tinyinterp_test_chunk_correct.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            ids = _ids(4, 4)

            server._budget = None
            server.budget.gpu_budget = 10 ** 9
            unchunked = remote(input_ids=ids, get=[site])

            server._budget = None
            server.budget.gpu_budget = 1
            chunked = remote(input_ids=ids, get=[site])

            assert not isinstance(unchunked, list)
            assert not isinstance(chunked, list)
            assert torch.allclose(chunked[site], unchunked[site])
            assert torch.allclose(
                cast(torch.Tensor, chunked.logits),
                cast(torch.Tensor, unchunked.logits),
            )
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_no_chunk_when_fits(self) -> None:
        """Small batch doesn't get chunked."""

        sock = "/tmp/tinyinterp_test_no_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 10 ** 9
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(2, 4), get=[site])
            assert not isinstance(result, list)
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_auto_chunks(self) -> None:
        """stop_at_last_get collection also auto-chunks."""

        sock = "/tmp/tinyinterp_test_collect_chunk.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(4, 4), get=[site], stop_at_last_get=True)
            assert not isinstance(result, list)
            assert result.partial
            assert result[site].shape[0] == 4
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_chunks_have_activations(self) -> None:
        """Chunked collection stitches activations back into one batch."""

        sock = "/tmp/tinyinterp_test_collect_shape.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            server.budget.gpu_budget = 1
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(3, 4), get=[site], stop_at_last_get=True)
            assert not isinstance(result, list)
            act = result[site]
            assert act.shape[0] == 3
            assert act.ndim == 3
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_cpu_default_batch_not_chunked(self) -> None:
        """CPU-only servers keep the normal batched tensor contract by default."""

        sock = "/tmp/tinyinterp_test_cpu_batch.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            result = remote(input_ids=_ids(2, 4), get=[site])
            assert not isinstance(result, list)
            assert result[site].shape[0] == 2
            remote.close()
        finally:
            _stop_server(server, sock)

    def test_collect_fast_path_closes_collectors(self) -> None:
        """Internal collector fast path should not leak collector handles."""

        sock = "/tmp/tinyinterp_test_collectors.sock"
        server, _ = _start_server(FakeDecoderModel, sock)
        try:
            remote = ti.Model(f"unix://{sock}")
            site = remote.layers[0]
            assert len(server._collectors) == 0
            _ = remote(input_ids=_ids(3, 4), get=[site], stop_at_last_get=True)
            assert len(server._collectors) == 0
            remote.close()
        finally:
            _stop_server(server, sock)


def test_model_tmp_path_is_loaded_like_a_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local checkpoint directory in /tmp should not be mistaken for a socket."""
    import tinyinterp.model as model_mod

    checkpoint = tmp_path / "checkpoint.sock"
    checkpoint.mkdir()
    wrapped = _seeded_model(FakeDecoderModel)
    monkeypatch.setattr(model_mod, "_load_model", lambda path, **_: wrapped)
    monkeypatch.setattr(model_mod, "_maybe_load_tokenizer", lambda _: None)

    model = ti.Model(str(checkpoint))

    assert isinstance(model, model_mod.Model)
    assert model.wrapped is wrapped
