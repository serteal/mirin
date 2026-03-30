"""Unified real-model benchmark matrix runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compare_libraries import CompareConfig, format_compare_report, run_compare_benchmarks
from .registry import ModelSpec, resolve_models
from .remote_compare import (
    RemoteCompareConfig,
    format_remote_compare_report,
    run_remote_compare_benchmarks,
)


@dataclass(slots=True)
class MatrixConfig:
    """Low-cost real-model benchmark matrix config."""

    model_names: list[str] | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    local_batch_size: int = 1
    local_seq_len: int = 64
    local_warmup: int = 0
    local_trials: int = 1
    remote_batch_size: int = 1
    remote_seq_len: int = 64
    remote_max_new_tokens: int = 4
    remote_warmup: int = 0
    remote_trials: int = 1
    json_output: str | None = None


def run_matrix(config: MatrixConfig) -> dict[str, Any]:
    """Run the real-model local/remote benchmark matrix."""

    models = resolve_models(config.model_names)
    reports = [_run_one_model(spec, config) for spec in models]
    output = {
        "config": asdict(config),
        "models": [spec.as_dict() for spec in models],
        "reports": reports,
    }
    if config.json_output is not None:
        path = Path(config.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def format_matrix_report(report: dict[str, Any]) -> str:
    """Format the matrix report for CLI output."""

    lines = ["Real Model Benchmark Matrix"]
    for entry in report["reports"]:
        spec = entry["model"]
        lines.append("")
        lines.append(f"{spec['model_name']}  family={spec['family']}  size={spec['size_label']}")
        lines.append(f"  local_compare: {_format_status(entry['local_compare'])}")
        lines.append(f"  remote_compare: {_format_status(entry['remote_compare'])}")

        local_report = entry["local_compare"].get("report")
        if local_report is not None:
            lines.append(_indent_block(format_compare_report(local_report), prefix="    "))
        remote_compare_report = entry["remote_compare"].get("report")
        if remote_compare_report is not None:
            lines.append(
                _indent_block(
                    format_remote_compare_report(remote_compare_report),
                    prefix="    ",
                )
            )
    return "\n".join(lines)


def _run_one_model(spec: ModelSpec, config: MatrixConfig) -> dict[str, Any]:
    local_compare = _run_local_compare(spec, config)
    remote_compare = _run_remote_compare(spec, config)
    return {
        "model": spec.as_dict(),
        "local_compare": local_compare,
        "remote_compare": remote_compare,
    }


def _run_local_compare(spec: ModelSpec, config: MatrixConfig) -> dict[str, Any]:
    try:
        report = run_compare_benchmarks(
            CompareConfig(
                model_name=spec.model_name,
                model_family=spec.family,
                device=config.device,
                dtype=config.dtype,
                batch_size=config.local_batch_size,
                seq_len=config.local_seq_len,
                warmup=config.local_warmup,
                trials=config.local_trials,
                hf_block_path=None,
            )
        )
    except Exception as exc:
        return _status_from_exception(exc)
    return _status_from_report(report)


def _run_remote_compare(spec: ModelSpec, config: MatrixConfig) -> dict[str, Any]:
    try:
        report = run_remote_compare_benchmarks(
            RemoteCompareConfig(
                model_name=spec.model_name,
                model_family=spec.family,
                device=config.device,
                dtype=config.dtype,
                batch_size=config.remote_batch_size,
                seq_len=config.remote_seq_len,
                max_new_tokens=config.remote_max_new_tokens,
                warmup=config.remote_warmup,
                trials=config.remote_trials,
            )
        )
    except Exception as exc:
        return _status_from_exception(exc)
    return _status_from_report(report)


def _status_from_report(report: dict[str, Any]) -> dict[str, Any]:
    if _has_correctness_failure(report):
        return {"status": "correctness_failed", "report": report}
    if _has_performance_failure(report):
        return {"status": "performance_failed", "report": report}
    return {"status": "ok", "report": report}


def _has_correctness_failure(report: dict[str, Any]) -> bool:
    for check in report.get("correctness", {}).values():
        if check.get("skipped"):
            continue
        if not bool(check.get("ok", False)):
            return True
    return False


def _has_performance_failure(report: dict[str, Any]) -> bool:
    for check in report.get("performance", {}).values():
        if not bool(check.get("ok", False)):
            return True
    return False


def _status_from_exception(exc: Exception) -> dict[str, Any]:
    reason = f"{type(exc).__name__}: {exc}"
    lowered = reason.lower()
    if any(
        token in lowered
        for token in (
            "gated repo",
            "access to model",
            "not recognize this architecture",
            "does not recognize this architecture",
            "unsupported",
            "requires",
            "not found",
        )
    ):
        return {"status": "skipped", "reason": reason}
    return {"status": "failed", "reason": reason}


def _format_status(entry: dict[str, Any]) -> str:
    status = entry["status"]
    if status == "ok":
        return "ok"
    reason = entry.get("reason")
    if reason:
        return f"{status} ({reason})"
    return status


def _indent_block(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
