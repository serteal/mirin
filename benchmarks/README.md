# Benchmark Harness

`tinyinterp` follows the plan's tinygrad-style rule: if we claim a speedup, we benchmark it.

This harness benchmarks the current Phase 3 matrix against the current implementation:

- raw wrapped model
- `ti.Model(...)` with no `get=` / `map=`
- one and several `get=` sites
- one and several `map=` sites
- eager sweep vs `ti.batch()`

The harness does three things on every run:

1. Validates correctness before timing.
2. Warms up each path and synchronizes around timing.
3. Reports median, p90, standard deviation, environment metadata, and `ti.Counters` summaries.

## Run

Install the benchmark dependencies first:

```bash
uv sync --extra transformers --group bench
```

By default the CLI benchmarks these checkpoints:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-4b-it`
- `Qwen/Qwen3.5-4B`

If a checkpoint is gated and the current HuggingFace token cannot access it, the suite records the load
failure and continues with the remaining models.

```bash
uv run python -m benchmarks.run_phase3
```

Save structured results:

```bash
uv run python -m benchmarks.run_phase3 --json-output benchmarks/results/phase3.json
```

Run only one real model:

```bash
uv run python -m benchmarks.run_phase3 \
  --model-name meta-llama/Llama-3.1-8B-Instruct
```

Smaller synthetic smoke run:

```bash
uv run python -m benchmarks.run_phase3 \
  --synthetic \
  --device cpu \
  --dtype float32 \
  --layers 2 \
  --width 64 \
  --n-heads 4 \
  --seq-len 32 \
  --batch-size 2 \
  --micro-warmup 0 \
  --micro-trials 2 \
  --throughput-warmup 0 \
  --throughput-runs 1 \
  --sweep-width 2
```

## Compare Libraries

The separate comparison harness measures the same semantic GPT-2 site across:

- raw HuggingFace GPT-2
- `tinyinterp`
- TransformerLens
- `nnterp`

It benchmarks:

- plain forward
- one activation capture
- one activation zeroing intervention

The correctness baseline is a manual HuggingFace forward hook on GPT-2 block 0 output, so the
comparison is about equivalent user-visible work rather than different hook sites.

```bash
uv run python -m benchmarks.run_compare_libraries
```

Save structured results:

```bash
uv run python -m benchmarks.run_compare_libraries \
  --json-output benchmarks/results/compare-libraries-llama31-8b.json
```

## Compare Larger Dataset Workloads

For a much larger dataset-style one-site capture workload, benchmark:

- `tinyinterp` manual loop
- TransformerLens capture loop
- `nnterp` capture loop

Default workload:

- model: `meta-llama/Llama-3.1-8B-Instruct`
- source batches: `16`
- chunk batch size: `8`
- sequence length: `512`
- dataset batches: `16`

```bash
uv run python -m benchmarks.run_compare_streaming_libraries
```

Save structured results:

```bash
uv run python -m benchmarks.run_compare_streaming_libraries \
  --json-output benchmarks/results/compare-streaming-libraries-llama31-8b.json
```

## Phase 4 Server Benchmarks

The server harness benchmarks:

- plain HuggingFace manual-hook collection loop
- local tinyinterp capture loop
- `ti.Server` collector mode
- plain HuggingFace `generate()` for one and many prompts
- `ti.Server` single-request generation
- `ti.Server` multi-session decode

The default real-model matrix is:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-4b-it`
- `Qwen/Qwen3.5-4B`

Run the full suite:

```bash
uv run python -m benchmarks.run_phase4_server
```

Save structured results:

```bash
uv run python -m benchmarks.run_phase4_server \
  --json-output benchmarks/results/phase4-server.json
```

Run the synthetic smoke workload:

```bash
uv run python -m benchmarks.run_phase4_server \
  --synthetic \
  --device cpu \
  --dtype float32 \
  --seq-len 32 \
  --dataset-batch-size 2 \
  --dataset-batches 2 \
  --generate-batch-size 2 \
  --max-new-tokens 2 \
  --warmup 0 \
  --trials 1
```
