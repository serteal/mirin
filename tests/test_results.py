"""Unit tests for result objects."""

from __future__ import annotations

from mirin.server.results import PlanResult


def test_plan_result_defaults_are_empty_and_mutable() -> None:
    result = PlanResult()
    assert result.activations == {}
    assert result.metadata == {}
    result.activations["site"] = 1
    result.metadata["key"] = "value"
    assert result.activations["site"] == 1
    assert result.metadata["key"] == "value"
