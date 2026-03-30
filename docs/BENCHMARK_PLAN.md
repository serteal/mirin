# Benchmark Plan

_This file is the execution plan for what still needs to be benchmarked and compared. [`BENCHMARKING.md`](./BENCHMARKING.md) stays the philosophy document._

## 0. Goal

Measure the public contract we actually want to support:

- one `ti.Model(...)` API
- local execution
- remote execution backed by `ti.Server(...)`
- one shared lowered runtime under both local and remote `ti.Model(...)`
- explicit runtime-internals benchmarks only where they justify implementation complexity
- optional comparisons against other libraries and serving stacks when they are actually
  installed and support the checkpoint family being benchmarked

The plan is not to benchmark every internal helper. The plan is to benchmark the user-visible workloads that justify the runtime complexity.

## 1. Rules

Every benchmark group must:

1. verify correctness before timing
2. compare the same semantic workload across implementations
3. report environment exactly
4. include median and spread, not one lucky run
5. keep the baseline simple and honest

If a path is not benchmarked, it should not be described as a speedup.

## 2. Hardware Matrix

We need at least these environments:

1. CPU shakedown run
   Purpose: catch obvious correctness and transport regressions before GPU benchmarking.

2. Single-GPU A100 class machine
   Purpose: main performance evidence for local, server, remote, and cross-library comparisons.

3. Multi-GPU machine
   Purpose: only for the multi-GPU plan once the implementation exists and is benchmarkable.

The canonical recorded environment fields should include:

- model name
- dtype
- device
- GPU name
- CUDA version
- PyTorch version
- transformers version
- optional library versions: TransformerLens, nnterp, vLLM

## 3. Model Matrix

The benchmark set should cover:

1. `Qwen/Qwen3-1.7B`
   Purpose: small Qwen 3 text model.

2. `Qwen/Qwen3.5-4B`
   Purpose: newer Qwen family checkpoint and long-context support probe.

3. `google/gemma-2-2b-it`
   Purpose: Gemma 2 dense-attention baseline.

4. `google/gemma-3-4b-it`
   Purpose: Gemma 3 text-only benchmark target with a different module tree.

5. `meta-llama/Llama-3.1-8B-Instruct`
   Purpose: larger dense-attention serving and local interp anchor.

Real-model runs are the evidence. If a benchmark does not exercise a realistic model and workload,
it should not be used to justify complexity or performance claims.

The matrix must record support-aware skips instead of failing wholesale. Missing packages, gated
repos, and unsupported checkpoint families are part of the real benchmark picture.

## 4. Local vs Remote Matrix

These are the core same-API comparisons for `ti.Model(...)`. The question here is not whether the
internal runtime is fast in isolation. The question here is whether the deployed remote path stays
close to the local path once transport and serving overhead are included.

### 4.1 Forward

Compare:

- raw wrapped model
- `ti.Model(local)` with no `get=` / `map=`
- `ti.Model(remote)` with no `get=` / `map=`

Measure:

- median latency
- p90 latency
- max memory
- output parity

### 4.2 Activation Capture

Compare:

- raw HF manual hook loop
- `ti.Model(local, get=[site])`
- `ti.Model(local, get=[site], stop_at_last_get=True)`
- `ti.Model(remote, get=[site])`
- `ti.Model(remote).collect(...)`

Measure:

- latency for one batch
- examples / second
- tokens / second
- activation parity
- host/device transfer cost

### 4.3 Intervention / `map=`

Compare:

- local `map=`
- remote `map=`

Measure:

- latency
- logits parity vs local
- any transfer overhead from remote execution

### 4.4 Generation

Compare:

- raw HF `generate()`
- `ti.Model(local).generate(...)`
- `ti.Model(remote).generate(...)`

Workloads:

- single prompt
- fixed-size prompt batch
- variable-length prompt batch

Measure:

- prompt latency
- decode throughput
- generated-token parity
- session-management overhead

### 4.5 Gradients

Compare:

- `ti.Model(local, grad=True)`
- `ti.Model(remote, grad=True)` for the supported tensor-input subset

Measure:

- forward latency
- backward latency
- activation parity
- gradient parity
- handle fetch overhead

This should stay scoped to the supported serving/interp subset. Do not overclaim arbitrary-model parity.

## 5. Runtime Internals Matrix

These justify the extra runtime complexity directly. They are implementation diagnostics, not
public API paths. This is where direct collector, scheduler, session, and decode benchmarks belong.

### 5.1 Stateless Throughput

Compare:

- sequential runtime call path
- concurrent stateless runtime call path
- raw HF loop

Measure:

- requests / second
- tokens / second
- scheduler utilization
- queue wait

### 5.2 Collector Throughput

Compare:

- raw HF manual-hook dataset loop
- local `ti.Model.collect(...)`
- remote `ti.Model.collect(...)`
- runtime collector fast path

Measure:

- examples / second
- tokens / second
- CPU memory
- GPU memory
- transfer bytes if available

### 5.3 Decode Throughput

Compare:

- raw HF batched `generate()`
- runtime generate fast path
- explicit multi-session `prefill/decode`
- remote `model.generate(...)`

Workloads:

- equal prompt lengths
- mixed prompt lengths
- subset decode after shared prefill

Measure:

- prompt throughput
- decode tokens / second
- end-to-end latency
- cache memory

### 5.4 Session Runtime Stress

Stress:

- many open sessions
- many collectors
- repeated value-handle fetch/release
- repeated grad-handle fetch/backward/release

Measure:

- resource counts from runtime stats
- leaked handles after completion
- cleanup behavior after disconnect

## 6. Remote Transport Matrix

Remote is now below the API boundary, but it still needs its own measurements.

Compare:

- local `ti.Model(...)`
- remote `ti.Model(...)`
- runtime primitive directly where relevant

Measure:

- plan compilation overhead
- request round-trip overhead
- lazy value fetch cost
- lazy grad fetch cost
- prompt-list call overhead
- large activation fetch overhead

The important number is not only absolute latency. It is the delta relative to local execution for the same user call.

## 7. Cross-Library Matrix

We need two kinds of comparison.

### 7.1 Local Interpretability Libraries

Compare:

- raw HuggingFace
- `tinyinterp` local
- TransformerLens
- nnterp

Workloads:

- plain forward
- one activation capture
- capture-only fast path
- one zeroing intervention
- dataset-scale one-site capture loop

This is mostly already present in `benchmarks/compare_libraries.py` and `benchmarks/compare_streaming_libraries.py`, but it needs to stay aligned with the rebooted public API.

### 7.2 Local vs Remote `tinyinterp`

Compare:

- raw HuggingFace `generate()`
- local `ti.Model(...)`
- remote `ti.Model(...)`

Workloads:

- single prompt latency
- fixed-size batch generation
- mixed-length batch generation
- multi-session-backed remote generation
- remote client call overhead

Metrics:

- prompt latency
- decode tokens / second
- batch throughput
- GPU memory use
- end-to-end delta vs local

This is the comparison that justifies whether the remote path is worth using for user-visible
workloads.

## 8. Failure-Mode Matrix

These are not optional. They are part of the benchmark plan because weird failure modes are part of runtime quality.

Benchmark or stress-test:

- remote disconnect during `CALL_MANY`
- remote disconnect during value fetch
- remote disconnect during grad backward
- session close during decode
- subset decode after family prefill
- chunked prefill with activation capture
- queue saturation and admission rejection
- large activation capture near memory limits
- repeated connect/close cycles

Measure:

- correctness
- cleanup behavior
- leaked runtime state
- time to recover for the next request

## 9. Reporting Format

Each saved report should include:

- benchmark config
- environment
- correctness checks
- timed cases
- counters or queue stats when relevant
- notes on unsupported paths or skips

The output should be structured JSON first, human-readable CLI formatting second.

## 10. Run Cadence

Every change set should run:

1. CPU smoke benchmarks
2. CUDA smoke benchmarks
3. affected correctness tests

Every performance-sensitive change should run:

1. Model API on CUDA
2. Runtime internals on CUDA
3. cross-library local compare on CUDA
4. any affected failure-mode stress cases

Before claiming a major speedup, run the real-model matrix on the main GPU box and save results under `benchmarks/results/`.

## 11. Immediate Gaps

The highest-priority missing benchmark work is:

1. permanent CUDA-marked coverage for end-to-end remote paths and the main matrix
2. local vs remote latency deltas for the same `ti.Model(...)` call
3. remote `grad=True` cost vs local `grad=True`
4. runtime stateless concurrency throughput under load, not just one correctness case
5. remote latency deltas on a wider model/context matrix

## 12. Definition Of Done

The benchmark story is in good shape when:

- every public performance claim maps to a benchmark case
- local vs remote parity is benchmarked for the supported API subset
- runtime complexity is justified by measured wins on the runtime-internals matrix
- cross-library comparisons cover both local interp and local-vs-remote workloads
- failure-mode stress results are recorded, not guessed
