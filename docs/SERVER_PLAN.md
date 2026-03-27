# tinyinterp Server Plan

_The core stays tiny. The server is allowed to be complicated only where benchmarks prove it buys real throughput._

---

## 0. Server Philosophy (non-negotiable)

This document replaces the idea that server mode is just a remote `ti.Model` mirror.

**a)** `ti.Model` remains the tiny, architecture-agnostic core. Server complexity lives in `tinyinterp.server`, not in the core wrapper.

**b)** The server optimizes real bottlenecks: KV cache ownership, batching, scheduler policy, memory traffic, and transport overhead. Not API cleverness.

**c)** Every claimed speedup is benchmarked against plain HuggingFace baselines:

- `model(**inputs)` for full-sequence forward
- `model.generate(...)` for single-request generation
- naive Python loops for dataset activation collection

**d)** We do not reimplement kernels on day 1. We first squeeze the maximum out of HuggingFace + PyTorch + modern attention backends by owning the execution loop correctly.

**e)** We do not promise vLLM/SGLang parity unless we implement equivalent primitives. Their wins come from scheduler + cache-manager + memory-system design, not just a nicer API.

**f)** Complexity is quarantined. The hot path may specialize to HuggingFace CausalLM serving. The public API should still be small and obvious.

**g)** The server has two jobs only:

1. Fast inference with optional interventions
2. Fast activation collection at scale

If a feature does not help one of those, it does not belong in the server.

---

## 1. One-Sentence Summary

The tinyinterp server is a specialized inference engine for HuggingFace CausalLMs that can read and modify internal activations while owning KV cache, prefill/decode scheduling, continuous batching, adaptive collection batching, and narrow GPU-aware transport.

---

## 2. Scope

### 2.1 Must support

- Interactive generation with persistent per-session state
- Full-sequence forward passes with `get=` and `map=`
- Batched dataset activation collection over large corpora
- Continuous batching across active decode sessions
- Adaptive batching for both collection and serving
- Correct GPU residency for weights, caches, and outputs
- HuggingFace model loading, tokenizer integration, quantization, and attention backends

### 2.2 Explicit non-goals for v1

- Replacing tinyinterp core semantics with server-only abstractions
- Supporting every PyTorch model family on the high-performance path
- Promising custom paged-attention kernels before benchmarks show HuggingFace backends are the blocker
- Shipping a generic "remote Python object RPC" and calling it a serving engine

### 2.3 Honest target

v1 should materially beat normal HuggingFace eager forward and `generate()` on:

- Multi-request decode throughput
- Long-running interactive sessions
- Full-dataset activation extraction
- Batched capture-only workloads

v1 does **not** need to beat vLLM or SGLang everywhere. It needs to move from "debug RPC" to "real inference engine."

---

## 3. The Split: Core vs Server

### 3.1 Core stays generic

The existing local API remains the foundation:

- `ti.Model`
- `get=`
- `map=`
- `stop_at_last_get=True`
- `ti.batch()` for local opt-in batching

This layer remains architecture-agnostic and small.

### 3.2 Server is explicitly specialized

The performance server is allowed to say:

- this is a HuggingFace CausalLM
- this request is prefill
- this request is decode
- this request owns a cache-bearing session
- these requests are batch-compatible

That specialization is not a betrayal of the tinygrad philosophy. It is complexity quarantine.

### 3.3 Two server faces

The server package should expose two distinct faces:

1. `DebugServer`
   Thin remote mirror of `ti.Model` for tests, notebooks, and remote control

2. `InferenceServer`
   High-performance engine for CausalLM serving and dataset collection

The existing ZMQ mirror fits `DebugServer`. It should not be treated as the final serving architecture.

---

## 4. Execution Modes

The engine has three execution modes. They share hook machinery, but not scheduler assumptions.

### 4.1 Collection mode

Use case: offline activation extraction over a dataset.

Properties:

- `use_cache=False` by default
- large full-sequence batches
- bucketed by sequence length / token count
- optional `stop_at_last_get=True` when logits are not needed
- outputs typically written to CPU pinned memory, memory-mapped files, or server-side reducers

This mode is throughput-first and should ignore generation-specific complexity.

### 4.2 Session mode

Use case: interactive generation.

Properties:

- server-owned session id
- persistent KV cache
- explicit `prefill` then repeated `decode`
- continuous batching across sessions
- chunked prefill for long prompts
- small narrow outputs per step: token ids, logits slices, requested activations

This is where vLLM/SGLang-style ideas matter most.

### 4.3 Hybrid inspect mode

Use case: generation while also reading selected internals.

Properties:

- same session semantics as generation
- `get=` allowed on prefill and decode
- `map=` allowed if the plan is fixed for the session
- return only explicitly requested sites

This mode is the tinyinterp differentiator. It must not destroy the serving engine's cache and scheduler model.

---

## 5. Public API Shape

The API should be explicit about state and scheduling.

```python
server = ti.InferenceServer(
    "meta-llama/Llama-3.2-1B",
    device="cuda",
    attn_backend="flash_attention_2",
)

plan = server.compile(
    get=["model.layers.8"],
    map=None,
    output={"logits": False, "activations": True},
)

session = server.open_session(plan=plan, cache="dynamic")
server.prefill(session, input_ids=prompt_ids)
step = server.decode([session], max_new_tokens=1)

collector = server.open_collector(
    plan=plan,
    use_cache=False,
    stop_at_last_get=True,
)
for batch_out in collector.run(dataset):
    ...
```

### 5.1 Core request types

- `compile(...) -> plan_id`
- `open_session(plan_id, cache=..., limits=...) -> session_id`
- `prefill(session_id, input_ids, attention_mask=None, ...)`
- `decode(session_ids, max_new_tokens=1, ...)`
- `close_session(session_id)`
- `open_collector(plan_id, ...) -> collector_id`
- `collect_batch(collector_id, batch)`
- `close_collector(collector_id)`

### 5.2 What the API does not pretend

It does **not** pretend that a high-performance server is just:

```python
remote_model.generate(...)
```

That hides the exact control we need to own.

---

## 6. Plan Compilation

The server should not resolve paths and rebuild hook specs on every call.

### 6.1 Compile once

`compile(get=..., map=..., output=...)` turns user-facing paths into:

- module ids
- hook flags
- map op descriptors
- activation shape metadata if known
- a stable plan fingerprint

### 6.2 Plan fingerprint

Every executable request is keyed by:

- model id
- dtype / quantization mode
- attention backend
- execution mode (`collect`, `prefill`, `decode`)
- cache mode
- compiled interpretability plan fingerprint
- output policy fingerprint

This fingerprint is the unit of batch compatibility.

### 6.3 Why this matters

The current debug server resolves proxy paths at runtime and serializes Python call shapes every request. That is acceptable for debugging. It is wrong for the hot path.

---

## 7. KV Cache Design

The server owns KV cache. The client never sends `past_key_values` back and forth.

### 7.1 Session-owned cache

Each open session owns:

- active sequence length
- cache object
- model/config fingerprint
- compiled plan fingerprint
- sampling state
- output preferences

### 7.2 Cache modes

We support three cache modes, in order:

1. `dynamic`
   HuggingFace `DynamicCache`; easiest correct baseline

2. `static`
   HuggingFace `StaticCache`; good for `torch.compile` and low-latency stable-shape decode

3. `paged`
   Server-managed block allocator for vLLM-style cache paging if benchmarks show HuggingFace cache objects are the blocker

`dynamic` ships first. `static` is the first optimization tier. `paged` is earned, not assumed.

### 7.3 Cache validity and interventions

Cache reuse is only correct if the prior tokens were generated under the same effective computation.

Safe rule for v1:

- `get=` is cache-safe
- a session's `map=` plan is fixed when the session opens
- changing `map=` means opening a new session or re-prefilling

This rule is conservative and correct.

### 7.4 Prefix reuse

Prefix caching is useful, but it is phase-2 work.

The path is:

1. session-local cache reuse
2. identical-prefix reuse inside the same session family
3. global prefix cache / radix-tree style reuse if benchmarks justify the added complexity

---

## 8. Scheduler

The scheduler is the heart of the server.

### 8.1 Separate queues

Maintain separate queues for:

- decode
- prefill
- collection

Decode has the highest priority because it is latency-sensitive and cheap per step. Collection is throughput-sensitive and can backfill idle capacity.

### 8.2 Continuous batching

Decode requests from multiple sessions are merged into microbatches continuously.

The scheduler should batch by:

- compatible plan fingerprint
- compatible backend/dtype
- compatible cache mode
- current decode shape constraints

This is not `ti.batch()`. This is always-on server-side scheduling.

### 8.3 Chunked prefill

Long prompts should not monopolize the GPU.

Prefill is chunked by token budget so the scheduler can:

- interleave decode work
- stay within KV memory budgets
- avoid latency spikes from giant single-request prefills

### 8.4 Adaptive batching

Batch size is not a fixed integer. It is determined by:

- total token budget
- estimated activation bytes
- free KV pages / cache capacity
- output payload size
- current queue pressure

The scheduler should maximize useful GPU occupancy, not maximize "batch count."

### 8.5 Admission control

Before scheduling, estimate:

- prompt tokens
- projected decode tokens
- KV cache bytes
- activation capture bytes

Reject or downgrade requests early instead of OOMing the process.

### 8.6 Scheduling principle

Prefer a boring scheduler with correct accounting over a clever scheduler with hidden edge cases.

---

## 9. GPU and Memory Model

### 9.1 Residency rules

- weights live on GPU
- KV cache lives on GPU unless explicitly offloaded
- requested activations stay on GPU until the output policy says otherwise
- only final outbound payloads move to CPU

### 9.2 Output policies

Every request declares an output policy:

- `tokens_only`
- `logits_slice`
- `activations`
- `activations_to_cpu`
- `activations_to_mmap`
- `reduce_on_server`

This avoids accidental giant payloads.

### 9.3 Transfer strategy

For large activation collection, use:

- pinned CPU staging buffers
- optional double-buffered transfer
- optional background writer threads

This is an internal optimization. It should exist only if benchmarks show a win. We do **not** reintroduce a public `model.stream()` API.

### 9.4 Static shapes where helpful

For stable decode workloads:

- pad to a small shape set
- use `StaticCache`
- optionally use `torch.compile`

This is a server-level optimization mode, not a core API change.

### 9.5 Correctness over heroics

No hidden CPU copies.
No silent `.cpu()` of large tensors in the hot path.
No implicit serialization of model outputs that the caller did not ask for.

---

## 10. Transport and Control Plane

The control plane and data plane should be different.

### 10.1 Control plane

Used for:

- model tree inspection
- path discovery
- plan compilation
- stats and debugging

The existing path-based RPC model is acceptable here.

### 10.2 Data plane

Used for:

- batched inputs
- narrow outputs
- collector writes

This path must not use Python object pickle as its default transport.

### 10.3 Data-plane requirements

The hot path should support one of:

- binary tensor frames + small metadata envelope
- shared-memory handles for large CPU-side payloads
- direct in-process engine calls when the client lives in the same process

### 10.4 Narrow by default

For generation, the default outbound payload is:

- new token ids
- optional sampled logits slice
- optional requested activation tensors

Not:

- full model output object
- full logits tensor for the whole vocabulary unless requested
- arbitrary Python return values

---

## 11. Activation Collection

This is a first-class server workload, not an afterthought.

### 11.1 Collection defaults

For collector mode:

- `use_cache=False`
- `grad=False`
- batch by token budget
- bucket by sequence length
- use `stop_at_last_get=True` when logits are not needed

### 11.2 Collector output targets

Support:

- in-memory tensors for small jobs
- memory-mapped arrays for large jobs
- chunked on-disk formats
- server-side reducers

### 11.3 Reducers

Many interpretability jobs do not need raw full activations. They need:

- mean over token positions
- selected token slices
- top-k channels
- pooled head outputs

If a reducer can run on server-side GPU or CPU and cut payload size by 10x, it should.

### 11.4 Adaptive collector batching

Collector batching should account for:

- prompt token count
- requested sites
- activation dtype
- expected output bytes

The biggest safe batch for `get=[layer_8]` is not the biggest safe batch for `get=[all_layers]`.

### 11.5 Internal overlap

If benchmarked beneficial, the collector may overlap:

- next batch H2D
- current batch compute
- previous batch D2H / disk write

This overlap lives inside the collector engine, not in the public user API.

---

## 12. Interventions

Interventions are the unique requirement that normal serving engines do not have.

### 12.1 Server-supported map ops

v1 server supports only built-in map ops with serializable semantics:

- `zero`
- `add`
- `scale`
- `replace`

Custom Python lambdas are for local mode, not the performance server.

### 12.2 Batch compatibility and interventions

Requests can batch together only when their intervention plans are compatible.

The simplest correct rule:

- same `get=` plan
- same `map=` op types
- same mapped sites
- same output policy

We can relax this later only if benchmarks justify the extra scheduler complexity.

### 12.3 Cache and interventions

If an intervention changes the effective hidden-state computation for past tokens, the existing cache is no longer valid.

Therefore:

- fixed-plan sessions are the default
- "change the intervention mid-chat" means branch the session or re-prefill

### 12.4 No magic semantics

The server should never pretend that arbitrary internal edits are free or cache-preserving.

---

## 13. HuggingFace Integration

The server still wraps HuggingFace. That is the right move.

### 13.1 What we inherit from HuggingFace

- model loading
- tokenizer integration
- config inspection
- quantization integrations
- attention implementations
- cache classes
- architecture coverage

### 13.2 What we do **not** delegate

- execution scheduling
- session lifecycle
- cache ownership
- cross-request batching
- collector pipelines

### 13.3 Hot-path rule

Do not call HuggingFace `generate()` on the hot path.

Use explicit `forward(...)` with:

- `use_cache=True` for serving
- explicit cache objects
- explicit prefill/decode separation
- explicit scheduler-owned batching

`generate()` can remain as a convenience wrapper above the engine, not as the engine itself.

### 13.4 Backends

Support backend selection at load time:

- eager
- SDPA
- Flash Attention 2
- quantized variants when compatible

Benchmark them. Pick defaults by evidence.

---

## 14. Benchmarking

The benchmark suite decides what complexity is worth keeping.

### 14.1 Baselines

- plain HuggingFace forward loop
- plain HuggingFace `generate()`
- current debug server
- local tinyinterp `ti.Model`

### 14.2 Server benchmarks

- single-request prefill latency
- single-request decode latency
- multi-session decode throughput
- mixed prefill/decode load
- dataset activation collection throughput
- activation collection bytes/sec to CPU / disk
- overhead of `get=` alone
- overhead of `map=` alone
- overhead of `get+map`

### 14.3 Required metrics

- tokens/sec
- requests/sec
- p50 / p95 / p99 latency
- GPU memory used by weights
- GPU memory used by KV cache
- activation bytes transferred
- queue wait time
- scheduler utilization
- prefix cache hit rate if enabled

### 14.4 Acceptance gates

We keep an optimization only if it clearly improves one of:

- latency
- throughput
- memory efficiency
- implementation simplicity at equal performance

No benchmark win, no feature.

---

## 15. File Structure

One possible layout:

```text
tinyinterp/server/
+-- debug_server.py    # current remote mirror, kept simple
+-- inference.py       # user-facing high-performance server
+-- plans.py           # compile get/map/output into stable plans
+-- sessions.py        # session state and lifecycle
+-- cache.py           # dynamic/static/paged cache backends
+-- scheduler.py       # prefill/decode/collector scheduler
+-- collector.py       # dataset activation engine
+-- transport.py       # hot-path data plane
+-- control.py         # model tree inspection and stats
`-- metrics.py         # counters and benchmark helpers
```

Rule: core tinyinterp files should not become polluted with scheduler logic.

---

## 16. Roadmap

### Phase 0: Rename what exists

- keep current remote mirror as `DebugServer`
- stop calling it the performance server

### Phase 1: Compiled plans + collector

- compile path-based `get=` / `map=` once
- add batched collector mode
- support `stop_at_last_get=True` collector fast path
- add output policies and memory-mapped sinks

### Phase 2: Session engine

- explicit `open_session`
- server-owned `DynamicCache`
- `prefill` and `decode` APIs
- narrow decode outputs

### Phase 3: Continuous batching

- decode queue
- microbatch scheduler
- chunked prefill
- queue metrics and admission control

### Phase 4: Static decode optimization

- `StaticCache`
- shape bucketing
- optional `torch.compile`
- pinned-memory transfer path

### Phase 5: Advanced cache management

- paged KV backend if justified
- prefix cache / shared-prefix reuse if justified
- more flexible batch-compatibility rules if justified

### Phase 6: Polish

- benchmark suite
- docs
- migration guide from debug server to inference server

---

## 17. The Explain-It-In-One-Paragraph Test

tinyinterp's performance server is not a generic remote wrapper around `ti.Model`; it is a specialized inference engine for HuggingFace causal language models that keeps model weights and KV cache on the GPU, splits prefill from decode, continuously batches compatible decode requests, adaptively batches activation-collection jobs by token and memory budget, and lets users attach fixed interpretability plans that read or modify selected internal activations. The tiny core API stays small and generic; the server earns extra complexity only where benchmarks show clear wins over plain HuggingFace forward loops and `generate()`.
