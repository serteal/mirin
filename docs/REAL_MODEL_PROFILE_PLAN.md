# Real-Model Profiling Plan

This plan is for profiling the current local-only `mirin.Model(...)` stack on real models such as `Qwen/Qwen3-4B`, validating correctness, and measuring end-to-end behavior of the current `probelab` integration.

The goal is not to produce one headline number. The goal is to answer:

1. Where is time actually spent in the current `mirin.Model` paths?
2. Are the current local collection paths correct on real workloads?
3. Which parts are slower than they should be?
4. How does the current `feat/mirin-backend` integration behave end to end on the same machine, same model, same dataset shape, and same output semantics?


## Principles

- Measure real user workloads, not only microbenchmarks.
- Keep correctness checks next to performance checks.
- Compare only semantically equivalent paths.
- Prefer one profiling harness with explicit workload labels over many ad hoc scripts.
- Do not add public API surface for benchmarking.


## Current Surfaces To Profile

### `mirin`

Current local-only surfaces that matter:

- `model(...)`
- `model.generate(...)`
- `model.collect(one_batch, ...)`
- `model.collect(request_list, ...)`
- `model.collect(dataset_iterable, ...)`
- `model.collect(..., out="cpu")`
- `model.collect(..., out=PATH)`
- `model.collect(..., process=fn)`

Relevant implementation points:

- public API in `mirin/model.py`
- executor batching/export behavior in `mirin/executors.py`
- shared runtime in `mirin/runtime/core.py`
- local collect path in `mirin/runtime/prefill.py`

### `probelab`

Current relevant surfaces on `feat/mirin-backend`:

- `stream_activations(...)`
- `collect_activations(...)`
- pooled collection
- flat token streaming
- downstream scoring/training-style workloads

Relevant implementation points:

- `probelab/processing/activations.py`
- `perf_checks/profile_score_benchmark.py`


## Workload Matrix

We should profile four classes of real workloads.

### 1. Raw `mirin.Model` API

Purpose:

- isolate `mirin` itself from `probelab`
- find overhead in request normalization, batching, hook capture, and export paths

Cases:

- plain forward:
  - `model(batch)`
- single-site capture:
  - `model(batch, get=[site])`
- multi-site capture:
  - `model(batch, get=[site1, site2, ...])`
- `stop_at_last_get=True`
- `model.collect(batch, out="gpu")`
- `model.collect(batch, out="cpu")`
- `model.collect(dataset, out="cpu")`
- `model.collect(dataset, out=PATH)`
- `model.collect(dataset, process=fn, out="gpu")`

### 2. `probelab` activation collection

Purpose:

- measure the actual current integration path
- understand batching, flattening, detection-mask handling, and pooling costs

Cases:

- `stream_activations(...)`
- `collect_activations(..., pool=None)`
- `collect_activations(..., pool="mean")`
- token-level collection with flat output
- a representative scoring workload that uses the collected activations directly

### 3. Large dataset export

Purpose:

- stress OOM safety and export path throughput
- identify CPU copy and mmap/write bottlenecks

Cases:

- `mirin.Model.collect(dataset, out=PATH)`
- `probelab.stream_activations(...)` + probelab-side memmap writer
- if an older branch has a materially different export path, note it in a one-off investigation but do not keep it as a standing baseline

### 4. Online per-batch processing

Purpose:

- model the “collect and immediately reduce/train” workflow
- see whether `process=fn` moves the bottleneck back to GPU or CPU

Cases:

- `mirin.Model.collect(dataset, process=pool_fn, out="gpu")`
- `mirin.Model.collect(dataset, process=flatten_fn, out="gpu")`
- probelab streaming + pooling
- probelab streaming + simple probe-training step


## Model Matrix

Primary target:

- `Qwen/Qwen3-4B`

Optional secondary target if auth is available:

- `google/gemma-3-4b-it`

Keep one small toy lane for fast regression:

- `toy-llama`

For real models, use:

- `dtype=torch.bfloat16`
- `device=cuda`
- tokenizer loaded from the same model id


## Dataset Shapes

Use both synthetic and real datasets.

### Synthetic

Purpose:

- stable, reproducible length distributions
- easy sweep over size and sequence length

Profiles:

- short prompts: 32-96 tokens
- mixed prompts: 64-256 tokens
- long prompts: 256-1024 tokens

Dataset sizes:

- small: 64 rows
- medium: 512 rows
- large: 4096 rows

### Real

Use at least one `probelab` dataset that exercises detection masks and assistant-token pooling.

Good candidates from current usage:

- `wildguard_mix`
- one smaller dataset used by examples/tests

For real-data profiling, record:

- sample count
- total tokens
- sequence length quantiles
- detection-mask density


## Measurements To Collect

### End-to-end

- elapsed wall time
- samples/s
- tokens/s
- rows exported/s

### Runtime stats

From `model.stats()` after each run:

- queue totals for `collect_batch`, `call`, and `generate`
- total physical batches
- total tokens
- total sessions
- split counts
- reject counts
- peak inflight

### Memory

- peak GPU allocated
- peak GPU reserved
- peak runtime-accounted GPU bytes
- peak runtime-accounted CPU bytes
- ending reserved bytes
- mmap bytes written

### Correctness

- max absolute difference against a reference path
- output shape checks
- export row/file counts
- pooled equivalence where semantics match


## Profiling Breakdown

We need layered profiling, not just one timer.

### Layer 1: structured wall-clock timers

Add explicit timing around:

- request normalization
- batch materialization
- runtime collect call
- result splitting
- CPU transfer
- mmap write
- `process=fn`
- probelab flattening
- probelab pooling

This should be the default profiling mode because it is cheap and easy to compare across runs.

### Layer 2: PyTorch profiler

Use `torch.profiler` on a focused subset of cases:

- `model.collect(batch, out="gpu")`
- `model.collect(batch, out="cpu")`
- `model.collect(dataset, out=PATH)`
- `probelab.collect_activations(..., pool="mean")`

Capture:

- CPU op time
- CUDA kernel time
- memory timeline
- copies and synchronizations

This is for finding kernel-level bottlenecks after the structured timers identify the bad path.

### Layer 3: allocator and transport evidence

For each long run:

- `torch.cuda.reset_peak_memory_stats()`
- record `max_memory_allocated`
- record `max_memory_reserved`
- capture runtime `model.stats()`

For mmap export:

- bytes written
- files written
- write time share


## Correctness Baselines

Every benchmark lane needs a baseline.

### For raw `mirin`

- baseline: direct wrapped HF model on the same batched inputs
- compare logits and selected activations

### For `probelab`

- baseline: the current branch against stable local references on the same inputs
- compare:
  - activation shapes
  - pooled outputs
  - flat token counts
  - score outputs if the downstream workload is included

Where semantics differ between codepaths, mark the lane unsupported instead of pretending they are comparable.


## Concrete Execution Order

### Phase 1. Environment setup

1. Ensure `transformers` extras are installed in both repos.
2. Ensure `accelerate` is installed if the benchmark path uses `device_map="auto"`.
3. Record environment once:
   - GPU
   - driver
   - torch version
   - transformers version
   - commit hashes of `mirin` and `probelab`

### Phase 2. Raw `mirin` profiling

1. Reuse and extend `mirin/benchmarks/model_api.py`.
2. Add explicit collect-path cases:
   - `collect_gpu`
   - `collect_cpu`
   - `collect_process`
   - `collect_export`
3. Run first on `toy-llama`, then `Qwen/Qwen3-4B`.
4. Save JSON reports.

### Phase 3. Current `probelab` branch profiling

1. Extend or replace `probelab/perf_checks/profile_score_benchmark.py`.
2. Add structured timing around:
   - tokenization
   - batch selection
   - `model.collect`
   - flattening
   - pooling
3. Run:
   - `stream_activations`
   - `collect_activations(pool=None)`
   - `collect_activations(pool="mean")`
4. Save JSON reports.

### Phase 4. Deep profiler pass

For the slowest 2-3 lanes only:

1. run `torch.profiler`
2. inspect CPU copies, padding overhead, flattening cost, and mmap write share
3. identify the real bottleneck category:
   - model compute
   - hook capture
   - output assembly
   - CPU transfer
   - pooling
   - flattening
   - Python overhead
   - mmap writes

### Phase 5. Stress and OOM validation on real model

Using `Qwen/Qwen3-4B`:

1. large CPU collection
2. large mmap export
3. `process=fn` per-batch reduction
4. mixed-length large dataset

For each:

- verify no unexpected OOM
- verify ending reserved bytes are zero
- verify row/file counts


## Deliverables

The profiling work should produce:

1. One reusable benchmark harness for raw `mirin.Model`.
2. One reusable benchmark harness for the current `probelab` branch.
3. JSON reports checked into a `benchmarks/results/` or `perf_checks/results/` folder, not hand-written notes.
4. A short findings report with:
   - bottlenecks by workload
   - correctness issues found
   - recommended next code changes in priority order


## Expected Bottleneck Areas

These are the likely hotspots to confirm or disprove:

- request normalization for request-list collection
- repeated batch splitting/merging in `mirin/executors.py`
- activation materialization to CPU
- mmap write frequency causing too many small files/chunks
- probelab-side flattening and pooling dominating wall time
- variable-length generation fallback reducing batching efficiency


## Out Of Scope

- reintroducing any server/remote path
- multi-GPU serving
- distributed training
- productizing benchmark commands into a public CLI
