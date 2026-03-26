# Benchmarking Philosophy

`tinyinterp` should follow the same basic rule stated in the plan: if we claim a speedup, we benchmark it. The benchmark is not a marketing artifact. It is the evidence that a feature improved the real workload without making the system harder to understand or maintain.

## Principles

1. Correctness before speed.
   Every benchmarked path must first have a correctness check against the unfused eager path or the raw wrapped model.

2. Measure user-visible work.
   We care about whole forward passes, intervention sweeps, and dataset streaming throughput. Microbenchmarks are useful only when they explain a user-visible result.

3. Compare against the right baseline.
   The main baselines are:
   - raw wrapped model
   - `tinyinterp` with no `get=` / `map=`
   - `tinyinterp` eager `get=` / `map=`
   - `tinyinterp` inside `ti.batch()`
   - `tinyinterp` `model.stream(...)`

4. Report environment exactly.
   Every result should include:
   - model name and attention implementation
   - sequence length and batch size
   - dtype
   - GPU model
   - PyTorch version
   - CUDA / Metal backend details

5. Avoid cherry-picking.
   Use fixed seeds, fixed prompts or batches, fixed warmup counts, and fixed repetition counts. Report median and spread, not just the best run.

## Phase 3 Benchmark Matrix

When we move to a GPU machine, Phase 3 should benchmark these cases:

1. Inactive hook overhead.
   Compare raw model vs `ti.Model(model)` with no `get=` / `map=`.

2. Activation capture overhead.
   Compare no hooks vs one `get=` vs several `get=` sites.

3. Activation patching overhead.
   Compare no hooks vs one `map=` vs several `map=` sites.

4. Batch sweep fusion.
   Compare a sweep run eagerly vs the same sweep inside `ti.batch()`.
   Report:
   - user calls
   - actual forward passes
   - total wall time
   - speedup factor

5. Streaming throughput.
   Compare a manual Python loop over dataset batches vs `model.stream(...)`.
   Report:
   - examples / second
   - tokens / second when relevant
   - host memory usage
   - GPU memory usage if available

## Method

1. Warm up each path.
   Run several untimed iterations first so kernels, caches, and allocations stabilize.

2. Synchronize around timing.
   On GPU, synchronize before starting and after finishing each timed region.

3. Use repeated trials.
   Prefer at least 20 timed trials for microbenchmarks and at least 5 full runs for larger throughput tests.

4. Record median, p90, and standard deviation.
   Median is the main comparison number. p90 helps expose unstable paths.

5. Capture counters.
   Include `ti.Counters.summary()` alongside wall-clock timing so we can explain why a path got faster or slower.

## What Counts As A Win

A change is a real win when:

- it preserves correctness
- it improves a benchmark that matters to researchers
- the result reproduces across runs
- the complexity added is proportional to the speedup

Small or noisy wins are not enough to justify a large abstraction. If the benchmark does not show a clear gain, the simpler implementation stays.
