# Benchmark Harness

`tinyinterp` follows the plan's tinygrad-style rule: if we claim a speedup, we benchmark it.

This harness benchmarks real user-visible workloads on real Hugging Face checkpoints.

The canonical entrypoint is the unified matrix runner. It benchmarks:

- local interpretability comparisons:
  - raw Hugging Face
  - `tinyinterp` local
  - TransformerLens when supported and installed
  - `nnterp` when installed
- local vs remote API comparisons:
  - raw Hugging Face baselines
  - `tinyinterp` local
  - `tinyinterp` remote `ti.Model("unix://...")`

The model registry is intentionally small and representative:

- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3.5-4B`
- `google/gemma-2-2b-it`
- `google/gemma-3-4b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

If a library is not installed, a checkpoint is gated, or a family is not supported by that
library in this matrix, the run records a skip and continues. That is deliberate: the output
should show the real support matrix, not pretend all libraries cover all models.

The repo keeps correctness tests under `tests/` and benchmark execution under `benchmarks/`.
That split matches the tinygrad-style rule: correctness stays deterministic, while performance
evidence stays in explicit benchmark commands on real workloads.

The harness does three things on every run:

1. Validates correctness before timing.
2. Warms up each path and synchronizes around timing.
3. Reports median, p90, standard deviation, environment metadata, and `ti.Counters` summaries.

## Run

Install the benchmark dependencies first:

```bash
uv sync --extra transformers --group bench
```

Run the full real-model matrix:

```bash
uv run python -m benchmarks.run_matrix --device cuda --dtype bfloat16
```

Save structured results:

```bash
uv run python -m benchmarks.run_matrix \
  --device cuda \
  --dtype bfloat16 \
  --json-output /tmp/tinyinterp-matrix.json
```

Run one model only:

```bash
uv run python -m benchmarks.run_matrix \
  --device cuda \
  --dtype bfloat16 \
  --model-name google/gemma-2-2b-it
```

## Slice Commands

The matrix runner is the main entrypoint. The commands below are for deeper slice debugging.

### Model API

Benchmarks the local `ti.Model(...)` API against raw wrapped forwards:

- raw wrapped model
- `ti.Model(...)` with no `get=` / `map=`
- one and several `get=` sites
- one and several `map=` sites
- eager sweep vs `ti.batch()`

```bash
uv run python -m benchmarks.run_model_api \
  --model-name meta-llama/Llama-3.1-8B-Instruct
```

### Cross-Library Local Compare

Measures the same semantic site across raw Hugging Face, `tinyinterp`, TransformerLens, and
`nnterp`, with support-aware skips.

```bash
uv run python -m benchmarks.run_compare_libraries \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --json-output /tmp/tinyinterp-compare-libraries.json
```

### Local vs Remote API Compare

Measures the same user-visible collect/generate workload across local `tinyinterp` and remote
`tinyinterp`. Both paths now go through the same lowered runtime; the only intended difference is
that the remote path adds transport and serving overhead on top of the shared execution core.
This slice reports whether deployment costs stay close to the local baseline on the covered cases.

```bash
uv run python -m benchmarks.run_remote_compare \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --model-family llama3.1
```

### Runtime Internals

For lower-level runtime diagnostics, there is also a separate runtime-internals slice. This is not
part of the public product surface. It exists to justify or reject runtime complexity and to
measure the lowered execution core directly, separate from the local-vs-remote API contract.

```bash
uv run python -m benchmarks.run_runtime_internals \
  --model-name meta-llama/Llama-3.1-8B-Instruct
```

### Compare Larger Dataset Workloads

For a larger dataset-style one-site capture workload:

- `tinyinterp` manual loop
- TransformerLens capture loop
- `nnterp` capture loop

```bash
uv run python -m benchmarks.run_compare_streaming_libraries
```

Save structured results:

```bash
uv run python -m benchmarks.run_compare_streaming_libraries \
  --json-output /tmp/tinyinterp-compare-streaming.json
```

## CUDA E2E Test

```bash
uv run python -m pytest -q tests/test_cuda.py --run-cuda
```
