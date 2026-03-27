"""Smoke test for the benchmark harness."""

from __future__ import annotations

from benchmarks.phase3 import BenchmarkConfig, run_phase3_benchmarks
from benchmarks.phase4_server import ServerBenchmarkConfig, run_phase4_server_benchmarks


def test_phase3_benchmark_harness_smoke() -> None:
    report = run_phase3_benchmarks(
        BenchmarkConfig(
            device="cpu",
            dtype="float32",
            layers=2,
            width=32,
            n_heads=4,
            vocab_size=64,
            seq_len=16,
            batch_size=2,
            micro_warmup=0,
            micro_trials=1,
            throughput_warmup=0,
            throughput_runs=1,
            sweep_width=2,
        )
    )

    assert report["environment"]["device"] == "cpu"
    assert all(check["ok"] for check in report["correctness"].values())

    cases = {case["name"]: case for case in report["cases"]}
    assert cases["raw_forward"]["median_ms"] > 0.0
    assert "get_one_stop_at_last" in cases
    assert cases["batch_fused"]["user_calls"] == 2
    assert cases["batch_fused"]["forward_passes"] == 1


def test_phase4_server_benchmark_harness_smoke() -> None:
    report = run_phase4_server_benchmarks(
        ServerBenchmarkConfig(
            model_name=None,
            device="cpu",
            dtype="float32",
            seq_len=16,
            dataset_batch_size=2,
            dataset_batches=2,
            generate_batch_size=2,
            max_new_tokens=2,
            warmup=0,
            trials=1,
            layers=2,
            width=32,
            n_heads=4,
            vocab_size=64,
        )
    )

    assert report["environment"]["device"] == "cpu"
    assert all(check["ok"] for check in report["correctness"].values())
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["hf_hook_loop"]["median_ms"] > 0.0
    assert "server_collector" in cases
    assert "server_generate_multi_session" in cases
