# tinyinterp Reboot Plan

_`docs/PLAN.md` stays the philosophy document. This file is the execution roadmap for the next major revamp._

---

## 0. Non-Negotiables

This plan assumes we can break compatibility and delete anything that gets in the way.

The bar from [PLAN.md](./PLAN.md) still applies:

- one obvious public API
- low line count as a proxy for low complexity
- no dead abstractions
- benchmark every claimed speedup
- keep architecture knowledge out of the core unless it buys real user value

If a feature is fast but makes the system harder to explain, it needs stronger benchmark evidence.

---

## 1. The Goal

Build the version of tinyinterp that is actually worth maintaining:

- `ti.Model(...)` is the only model API users learn
- local and remote use the same high-level calls
- `ti.Server(...)` is runtime infrastructure, not a second model facade
- remote execution lives below the API boundary
- fast paths are honest, narrow, and benchmarked
- failure modes are explicit and tested

The library should feel small from the outside and sharp on the inside.

---

## 2. Hard Decisions

### 2.1 What we keep

- `ti.Model` as the core user object
- module proxies and path-based navigation
- `get=`, `map=`, `generate()`, `collect()`
- the generic local eager path for arbitrary `nn.Module` exploration
- the serving runtime for optimized HuggingFace CausalLM workloads

### 2.2 What we delete

- any public notion that there are different "kinds" of model APIs
- semantic drift between local output, server output, and remote output
- remote transport based on pickled Python requests on the hot path
- endpoint heuristics that guess user intent from filesystem-looking strings
- legacy docs that describe APIs or capabilities that do not exist

### 2.3 What we narrow

The optimized runtime does not need to support every PyTorch model equally.

The split should be:

- generic eager mode: broad model support, minimal assumptions, notebook-first
- optimized runtime mode: explicit support for the serving subset we can benchmark and test well

That is consistent with tinygrad's philosophy. The public object stays simple; specialization is quarantined underneath it.

---

## 3. End State

### 3.1 Public API

```python
import tinyinterp as ti

local = ti.Model(hf_model)
remote = ti.Model("unix:///tmp/tinyinterp.sock")
server = ti.Server(hf_model, tokenizer=tokenizer)
server.serve("/tmp/tinyinterp.sock")
```

Public user methods:

- `model(...)`
- `model.generate(...)`
- `model.collect(...)`
- `grad=True` where the backend actually supports it

Public runtime methods:

- `Server.serve(...)`
- `Server.stats()`
- `Server.close()`

Everything else is internal unless we can defend it as a stable user concept.

### 3.2 Internal Architecture

The code should collapse toward five concepts:

1. `Model`
   Frontend API, module navigation, request normalization, user-facing semantics.

2. `Plan`
   Compiled description of `get=`, `map=`, output policy, and execution mode.

3. `Executor`
   Runs a plan. `LocalExecutor` and `RemoteExecutor` share the same frontend contract.

4. `Runtime`
   Server-side engine for scheduling, cache ownership, batching, collection, and stats.

5. `Refs`
   Lightweight handles for remote-owned values, sessions, buffers, and gradients.

The public API should not expose separate local/server/remote wrapper classes.

---

## 4. Core Design Rules

### 4.1 One output contract

There should be one `Output` type in `tinyinterp/output.py`.

It may contain:

- materialized local tensors
- lazy remote value refs
- partial results for capture-only execution

But the access pattern must be the same:

```python
out = model(..., get=[site])
act = out[site]
logits = out.logits
```

### 4.2 One request normalization path

Prompt parsing, token request normalization, chat-message handling, and site validation should live in one place.

Local, in-process server, and remote should not each reinterpret requests differently.

### 4.3 One planning path

`get=` and `map=` should compile once into a stable internal plan with:

- stable site ids
- output policy
- execution mode
- plan fingerprint

The same plan representation should feed both the local executor and the runtime.

### 4.4 Remote is an execution target, not a second API

Remote should stop being "send a model call over a socket."

Remote should become:

- connect
- fetch model metadata and site table
- compile plan
- upload inputs
- run plan
- fetch requested values or tokens
- manage session and grad handles

That is the tinygrad move: complexity belongs below the user object.

---

## 5. Repo Restructure

This is the preferred direction, not a mandate for exact filenames.

### 5.1 `tinyinterp/model.py`

Keep only:

- `Model`
- proxies
- shared request normalization
- frontend validation

Move out:

- backend-specific transport logic
- backend-specific output wrappers
- endpoint guessing

### 5.2 `tinyinterp/output.py`

Make this the single user-visible result shape.

It should know how to:

- expose logits
- expose activation lookup by proxy
- represent partial execution
- lazily fetch remote values when needed

### 5.3 `tinyinterp/server/inference.py`

Turn this into runtime orchestration only:

- runtime lifecycle
- compilation cache
- scheduling
- collector/session ownership
- stats

It should not define a second public model facade.

### 5.4 `tinyinterp/server/remote.py`

Split this into:

- transport/protocol
- remote executor

Delete:

- model-RPC semantics
- pickled hot-path metadata
- duplicate output wrappers

### 5.5 New internal modules

The revamp likely wants internal modules for:

- `executors.py`
- `requests.py`
- `refs.py`
- `protocol.py`

Only add them if they reduce complexity. The point is fewer conceptual seams, not more files.

---

## 6. Phased Execution Plan

## Phase 0: Delete And Simplify

Goal: remove abstractions that make the current design harder to reason about.

Tasks:

- remove public claims of multiple model APIs
- delete duplicate output wrappers in favor of one `Output`
- delete endpoint auto-detection heuristics
- trim stale docs
- mark the optimized runtime as a scoped serving backend, not a generic Python RPC layer

Exit criteria:

- new contributor can explain the public API in under two minutes
- docs do not contradict the code

## Phase 1: Define The Shared Contract

Goal: freeze the frontend semantics before reworking execution.

Tasks:

- define the exact contract for `model(...)`
- define the exact contract for `model.generate(...)`
- define the exact contract for `model.collect(...)`
- define what `grad=True` means on each backend
- define partial-output semantics for capture-only mode
- define the parity target for local vs remote

Exit criteria:

- one contract doc
- one set of contract tests
- unsupported combinations fail loudly and consistently

## Phase 2: Introduce Executors

Goal: make the backend split internal.

Tasks:

- add `LocalExecutor`
- add `RemoteExecutor`
- route `Model` through an executor interface
- move backend selection under construction time only
- make `Output` backend-agnostic

Exit criteria:

- the frontend codepath is shared
- local and remote differ only after planning/request normalization

## Remote Runtime RPC

Goal: move remote below the API boundary.

Tasks:

- replace `CALL`/`CALL_MANY` model-RPC with plan-oriented commands
- add versioned binary protocol
- use stable site ids instead of repeated string paths
- upload tensors via explicit buffers or shared memory
- return value handles instead of always materializing tensors immediately
- keep session state, cache, and collector state on the server

Exit criteria:

- remote collect/generate use the same `Model` API as local
- the transport protocol is about execution resources, not Python calls

## Remote Grad Semantics

Goal: make `grad=True` honest.

Tasks:

- local eager keeps direct autograd
- remote grad becomes handle-based server-owned tape
- add backward RPCs and gradient/value fetch
- support a minimal first cut:
  raw tensor inputs, `get=`, scalar backward, input grads
- add capability flags so unsupported backends fail early

Exit criteria:

- remote `grad=True` is either correct and tested or clearly unsupported
- no backend silently ignores gradient requests

## Phase 5: Upgrade The Runtime

Goal: make the optimized backend actually competitive on the workloads we care about.

Tasks:

- remove the global request lock from the hot path
- tighten compilation and scheduling boundaries
- support continuous decode batching cleanly
- improve chunked prefill correctness and semantics
- reduce cache copy churn
- make collector output residency explicit
- expose runtime stats that explain performance, not just report it

Exit criteria:

- throughput gains are real on the benchmark matrix
- stats explain queueing, memory, and batching behavior

## Phase 6: Failure Modes First

Goal: make the system survive the ugly cases.

Tasks:

- client disconnect during prefill
- client disconnect during decode
- stale session handle
- stale value handle
- stale grad handle
- oversized activation capture
- queue saturation
- cancellation
- partial batch failure
- timeout behavior
- OOM recovery behavior
- remote/server version mismatch

Exit criteria:

- every handled failure mode has a test
- every unhandled failure mode raises a precise error

## Phase 7: Benchmark Gates

Goal: stop merging complexity without evidence.

Tasks:

- define the benchmark matrix
- define correctness baselines
- define required environment reporting
- add perf smoke in CI
- add manual full benchmark runs for real hardware

Exit criteria:

- every speed claim in docs points to a benchmark
- perf-sensitive PRs add or update benchmark evidence

---

## 7. Benchmark Matrix

The benchmark suite needs to split into correctness, latency, throughput, and stress.

### 7.1 Public API parity

- local `model(...)` vs remote `model(...)`
- local `model.collect(...)` vs remote `model.collect(...)`
- local `model.generate(...)` vs remote `model.generate(...)`
- local `grad=True` vs remote `grad=True` for supported cases

Report:

- output equality or tolerance
- activation equality or tolerance
- token equality with fixed seeds

### 7.2 Overhead

- raw model vs `ti.Model(local)` inactive overhead
- local executor vs remote executor overhead with no `get=`
- plan compilation cold vs warm
- value fetch cold vs warm

### 7.3 Collection throughput

- Python loop baseline
- local `collect(...)`
- remote `collect(...)`
- runtime collector internal path
- capture-only vs full-forward
- CPU output vs GPU-resident output

### 7.4 Generation throughput

- HuggingFace `generate(...)`
- local `model.generate(...)`
- remote `model.generate(...)`
- runtime session decode path
- single-session latency
- multi-session throughput

### 7.5 Stress

- mixed prompt lengths
- long prompts with chunked prefill
- many concurrent sessions
- large activation captures
- repeated connect/disconnect
- remote reconnect after failure

---

## 8. Correctness Matrix

Before optimizing anything, we need a repo-wide correctness table for:

- logits parity
- activation parity
- map semantics
- `stop_at_last_get=True`
- chunked prefill
- batched decode
- subset decode within shared cache families
- session lifecycle
- collector lifecycle
- grad lifecycle

The benchmark harness should run correctness checks before timing the optimized path.

---

## 9. Monitoring And Introspection

`Server.stats()` should report enough to debug utilization without opening the code.

Minimum useful counters:

- queued requests
- active sessions
- active collectors
- prefill tokens per second
- decode tokens per second
- mean and p95 queue wait
- GPU memory used
- CPU pinned memory used
- cache bytes
- activation bytes emitted
- failed requests
- cancelled requests

These metrics are runtime concerns only. They should not leak into the `Model` API.

---

## 10. Documentation Plan

Docs should collapse to a small honest set:

- `PLAN.md`: philosophy and core architecture
- `REBOOT_PLAN.md`: this roadmap
- `SERVER_API.md`: truthful public runtime API
- `BENCHMARKING.md`: methodology and benchmark matrix

Everything else should either be updated, merged, or deleted.

No document should promise an API, optimization, or transport we do not ship.

---

## 11. Definition Of Done

The reboot is complete when all of the following are true:

- users learn one model API
- local and remote share one frontend contract
- the server is runtime infrastructure, not a second model wrapper
- remote transport is below the API boundary
- docs are truthful
- every major speed claim is benchmarked
- the main failure modes are tested
- the codebase is smaller in concepts even if some runtime internals get sharper

If we end up with more public surfaces, more wrapper types, and more docs explaining exceptions, the reboot failed even if some benchmarks improved.
