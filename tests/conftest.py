"""Pytest helpers for optional backend-specific test groups."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tinyinterp as ti


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="run CUDA-marked tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_cuda = config.getoption("--run-cuda") or os.environ.get("RUN_CUDA_TESTS") == "1"
    if run_cuda:
        return
    skip_cuda = pytest.mark.skip(reason="need --run-cuda or RUN_CUDA_TESTS=1 to run CUDA tests")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def _reset_counters_and_cleanup_sockets() -> None:
    ti.Counters.reset()
    _cleanup_test_sockets()
    yield
    ti.Counters.reset()
    _cleanup_test_sockets()


def _cleanup_test_sockets() -> None:
    for path in Path("/tmp").glob("tinyinterp_test_*.sock"):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue
