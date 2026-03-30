# tinyinterp Debugging & Profiling Tools

_Adapted from tinygrad's approach: environment variables, progressive verbosity, and zero-cost-when-off instrumentation._

---

## Part 1: What tinygrad has and what each tool does

### 1.1 The DEBUG levels (the core debugging tool)

tinygrad uses a single `DEBUG` environment variable with 7 levels. Each level adds more information on top of the previous one. Everything is controlled via `DEBUG=N` environment variables or `Context(DEBUG=N)` scoped overrides.

| Level      | What it shows                                                                | Purpose                                                      |
| ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `DEBUG>=1` | Lists devices being used                                                     | "Is my GPU even being used?"                                 |
| `DEBUG>=2` | Performance metrics per kernel: timing, memory usage, GFLOPS, GB/s bandwidth | "Where is time being spent?"                                 |
| `DEBUG>=3` | Buffers used per kernel (shape, dtype, strides) and optimization decisions   | "What data is moving and how is it being optimized?"         |
| `DEBUG>=4` | Generated kernel source code (CUDA/Metal/OpenCL)                             | "What code is actually running on the GPU?"                  |
| `DEBUG>=5` | UOp AST (intermediate representation of the computation graph)               | "What does the computation graph look like before lowering?" |
| `DEBUG>=6` | Linearized UOps — the operation sequence after scheduling                    | "What is the final execution order?"                         |
| `DEBUG>=7` | Generated assembly code for target hardware                                  | "What does the hardware actually execute?"                   |

**The key design:** each level is strictly additive. Level 2 shows everything level 1 shows plus more. You never need to set multiple flags. And the output goes to stderr, so it doesn't interfere with program output.

**Example output at DEBUG=2:**

```
*** METAL 4 E_25_4 arg 1 mem 0.22 GB tm 3.28us/ 0.00ms ( 0.00 GFLOPS 0.1|0.1 GB/s)
```

This tells you: backend (METAL), kernel number (4), kernel name (E_25_4), buffer count (1), total memory (0.22 GB), kernel time (3.28μs), cumulative time (0.00ms), compute throughput (0.00 GFLOPS), memory bandwidth (0.1 GB/s).

### 1.2 Context() — scoped configuration

tinygrad's `Context` is a context manager that temporarily sets environment variables within a scope:

```python
from tinygrad import Context, GlobalCounters

# Set DEBUG=2 only for this block
GlobalCounters.reset()
with Context(DEBUG=2):
    output = model(input)

# DEBUG is back to whatever it was before
```

This can also be used as a decorator:

```python
@Context(DEBUG=4)
def my_function():
    ...
```

**Critical insight:** Context variables don't just control DEBUG. They control BEAM search, JIT behavior, backend selection, and more. It's a single unified mechanism for runtime configuration.

### 1.3 GlobalCounters — aggregate performance metrics

`GlobalCounters` tracks cumulative stats across operations:

```python
from tinygrad import GlobalCounters
GlobalCounters.reset()
# ... do work ...
print(f"Kernels: {GlobalCounters.kernel_count}")
print(f"Memory: {GlobalCounters.mem_used / 1e9:.2f} GB")
print(f"Time: {GlobalCounters.time_sum_s:.3f}s")
print(f"GFLOPS: {GlobalCounters.global_ops / GlobalCounters.time_sum_s / 1e9:.1f}")
```

This lets you measure "how much work did this block of code do?" without caring about individual kernel details.

### 1.4 GRAPH=1 — computation graph visualization

```bash
GRAPH=1 python3 my_script.py
```

Outputs the computation graph as an SVG file (`/tmp/net.svg`). Shows the full DAG of operations — what depends on what, where fusion happened, what the data flow looks like.

### 1.5 VIZ=1 — interactive computation graph

```bash
VIZ=1 python3 my_script.py
```

Launches an interactive web-based visualization of the UOp graph. You can click on nodes, see their inputs/outputs, trace data flow. This is the "seriously, try VIZ=1" tool that geohot recommends — it shows the full computation graph at every level of the pipeline.

### 1.6 BEAM=N — kernel optimization search

```python
with Context(BEAM=2):
    output = model(input)
```

Does a BEAM search over possible kernel implementations, testing many scheduling/tiling options to find the fastest one for your hardware. Results are cached. This is a performance optimization tool, not a debugging tool, but it uses the same `Context` mechanism.

### 1.7 Process Replay — CI regression detection

Process replay is tinygrad's CI tool that compares the generated kernels of a PR against master. If your PR is a refactor, the generated kernels should be identical. If they differ, the CI flags it. This prevents subtle performance or correctness regressions.

It's not a runtime tool — it's a CI check that ensures the compiler produces the same output for the same input.

---

## Part 2: What tinyinterp needs — the translation

Our domain is different from tinygrad's. tinygrad profiles GPU kernels. We profile interp operations (hook activation, forward passes, activation capture, intervention application). But the DESIGN PRINCIPLES are identical:

1. **Single environment variable, progressive levels**
2. **Zero cost when off** — no branch in the hot path unless DEBUG is set
3. **Scoped overrides** via context manager
4. **Aggregate counters** for benchmarking
5. **Visualization** of what happens during a forward pass

### The mapping

| tinygrad tool      | tinyinterp equivalent | What it measures                                                       |
| ------------------ | --------------------- | ---------------------------------------------------------------------- |
| `DEBUG=1`          | `DEBUG=1`          | Which sites are hooked, what get/map is active                         |
| `DEBUG=2`          | `DEBUG=2`          | Per-call timing: hook overhead, forward pass time, memory              |
| `DEBUG=3`          | `DEBUG=3`          | Batch planner decisions, prefix sharing, buffer allocation             |
| `DEBUG=4`          | `DEBUG=4`          | Full hook trace: every hook fire, every capture, every map application |
| `DEBUG=5`          | `DEBUG=5`          | Model architecture discovery details                                   |
| `GlobalCounters`   | `ti.Counters`         | Aggregate stats: n_calls, total_time, total_activations_captured       |
| `Context(DEBUG=N)` | `ti.context(debug=N)` | Scoped override                                                        |
| `GRAPH=1`          | `GRAPH=1`          | SVG of the intervention graph (what was modified, what was captured)   |
| `VIZ=1`            | `VIZ=1`               | Reserved for future interactive visualization; not needed for v1       |
| `BEAM=N`           | —                     | Not applicable (we don't generate kernels)                             |
| Process Replay     | Numerical diff CI     | Compare activations against a known-good run                           |

---

## Part 3: Detailed design of each tool

### 3.1 DEBUG levels

#### DEBUG=1 — "What is happening?"

Shows: which model was loaded, what architecture was detected, which sites are available, and for each call, what get/map was requested.

```
$ DEBUG=1 python3 my_script.py

[ti] Model: LlamaForCausalLM (16 layers, 32 heads, d=2048)
[ti] Discovered 176 sites (16 layers × 11 sites/layer)
[ti] Device: cuda:0, dtype: bfloat16, params: 1.24B
[ti] call: get=[L5.resid] map={} input_shape=(1, 7)
[ti] call: get=[] map={L5.resid: replace} input_shape=(1, 7)
[ti] call: get=[L8.resid] map={L5.resid: zero} input_shape=(1, 7)
```

**When to use:** "Is tinyinterp even doing what I think it's doing?" First thing to try when something seems wrong.

**Implementation:** print statements gated by `if _debug >= 1:` at key points in `model.py` and `adapter.py`.

#### DEBUG=2 — "Where is time going?"

Shows: wall-clock timing breakdown for each call.

```
$ DEBUG=2 python3 my_script.py

[ti] call: get=[L5.resid] map={}
[ti]   resolve_sites:   0.002ms
[ti]   activate_hooks:  0.008ms  (1 get, 0 map)
[ti]   forward_pass:   12.453ms  (16 layers, input: [1, 7])
[ti]   collect:         0.003ms  (1 activation, 32.0 KB)
[ti]   deactivate:      0.001ms
[ti]   TOTAL:          12.467ms  (overhead: 0.014ms = 0.11%)
```

**When to use:** "Is tinyinterp adding overhead?" Compare `TOTAL` with a raw `model(input)` call. The overhead should be <1%.

**Implementation:** `time.perf_counter_ns()` around each phase in `Model.__call__`. The timing logic is ~15 lines.

#### DEBUG=3 — "What is the planner doing?"

Shows: decisions made by `ti.batch()`.

```
$ DEBUG=3 python3 my_script.py

[ti] batch: accumulated 32 calls
[ti]   group 1: 32 calls, same input, maps at L5.attn.head[0..31]
[ti]   strategy: batch_dim_fusion (32 → 1 forward pass)
[ti]   prefix_sharing: layers 0-4 computed once (5/16 = 31% saved)
[ti]   batch_input: [32, 7] (expanded from [1, 7], zero-copy)
[ti]   estimated speedup: 18.7x over sequential
[ti] batch: executing 1 forward pass
[ti]   forward_pass: 34.2ms (batch_size=32)
[ti]   per-element effective: 1.07ms (vs 12.5ms sequential)
```

**When to use:** "Is `ti.batch()` actually helping?" Check that the planner is finding the optimization opportunities you expect.

**Implementation:** print statements in `batch.py`'s planning and execution phases.

#### DEBUG=4 — "What is every hook doing?"

Shows: every individual hook fire, every activation capture, every map application.

```
$ DEBUG=4 python3 my_script.py

[ti] hook[L0.resid] (id=0): SKIP (flags: get=False, map=None)
[ti] hook[L0.attn] (id=1): SKIP
...
[ti] hook[L5.resid] (id=55): GET → buffer[3] shape=[1, 7, 2048] dtype=bf16 (32.0 KB)
[ti] hook[L5.attn] (id=56): MAP → zero() applied, output shape=[1, 7, 2048]
[ti] hook[L5.attn.pattern] (id=57): SKIP
...
[ti] hook[L8.resid] (id=88): GET → buffer[7] shape=[1, 7, 2048] dtype=bf16 (32.0 KB)
```

**When to use:** "Is the hook on the right module?" "Is the map function being applied where I think?" Low-level debugging.

**Implementation:** conditional print inside the hook function itself. This is the ONE place where debug output adds measurable overhead to the hot path (one branch check per hook per layer). With `DEBUG < 4`, the branch is never taken.

#### DEBUG=5 — "How was the model discovered?"

Shows: the full architecture discovery process.

```
$ DEBUG=5 python3 my_script.py

[ti] discovery: walking module tree of LlamaForCausalLM
[ti]   found ModuleList at model.model.layers (16 children)
[ti]   found ModuleList at model.model.layers.0.self_attn.heads (32 children)  ← smaller, skip
[ti]   selected: model.model.layers (16 layers)
[ti] layer[0] components:
[ti]   attn: self_attn (LlamaAttention)
[ti]     q_proj: q_proj (Linear, 2048→2048)
[ti]     k_proj: k_proj (Linear, 2048→256)  ← GQA: n_kv_heads=8
[ti]     v_proj: v_proj (Linear, 2048→256)
[ti]     o_proj: o_proj (Linear, 2048→2048)
[ti]   mlp: mlp (LlamaMLP)
[ti]   ln1: input_layernorm (LlamaRMSNorm)
[ti]   ln2: post_attention_layernorm (LlamaRMSNorm)
[ti] config: n_layers=16, n_heads=32, n_kv_heads=8, d_model=2048, d_head=64, d_mlp=8192
[ti] registered 176 sites (16 layers × 11 sites)
```

**When to use:** "Why can't tinyinterp find my model's layers?" "Why is it seeing the wrong number of heads?" Architecture debugging for new/exotic models.

**Implementation:** conditional prints in `adapter.py`'s discovery functions.

### 3.2 ti.Counters — aggregate performance metrics

Like tinygrad's `GlobalCounters`, but for interp operations.

```python
import tinyinterp as ti

ti.Counters.reset()

# ... do a bunch of interp work ...
for batch in dataloader:
    output = model(**batch, get=[model.layers[8].resid])

print(ti.Counters.summary())
```

Output:

```
tinyinterp counters:
  calls:                 312
  forward_passes:        312  (312 eager, 0 batched)
  total_time:            3.891s
  forward_time:          3.872s  (99.5%)
  hook_overhead:         0.014s  (0.4%)
  activations_captured:  312
  activations_bytes:     9.97 MB
  buffer_pool_hits:      312 / 312  (100.0%)
  buffer_pool_misses:    0
```

**What it tracks:**

| Counter                | What it measures                                                        |
| ---------------------- | ----------------------------------------------------------------------- |
| `calls`                | Number of `model()` calls                                               |
| `forward_passes`       | Number of actual forward passes (less than calls if batching is active) |
| `forward_time`         | Cumulative time in the actual model forward pass                        |
| `hook_overhead`        | Cumulative time in hook activation/deactivation/capture                 |
| `activations_captured` | Number of activation tensors captured                                   |
| `activations_bytes`    | Total bytes of captured activations                                     |
| `buffer_pool_hits`     | Times a pre-allocated buffer was reused                                 |
| `buffer_pool_misses`   | Times a new buffer had to be allocated                                  |
| `maps_applied`         | Number of map functions applied                                         |
| `batch_groups`         | Number of batched groups (inside `ti.batch()`)                          |
| `batch_fusions`        | Number of forward passes saved by batching                              |
| `prefix_layers_saved`  | Number of layer computations saved by prefix sharing                    |

**Implementation:**

```python
# tinyinterp/counters.py (~40 lines)
import threading

class _Counters:
    __slots__ = ("calls", "forward_passes", "forward_time_ns", "hook_overhead_ns",
                 "activations_captured", "activations_bytes",
                 "buffer_pool_hits", "buffer_pool_misses",
                 "maps_applied", "batch_groups", "batch_fusions", "prefix_layers_saved",
                 "_lock")

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            for attr in self.__slots__:
                if attr != "_lock":
                    setattr(self, attr, 0)

    def summary(self) -> str:
        # ... format the counters into a readable string
        ...

Counters = _Counters()
```

The counters are incremented inside `Model.__call__` with simple `Counters.calls += 1` statements. Cost: one integer increment per call. Negligible.

### 3.3 ti.context() — scoped configuration

```python
import tinyinterp as ti

# Scoped debug level
with ti.context(debug=2):
    output = model("Hello", get=[model.layers[5].resid])
    # ^^^ this call prints timing info

output = model("Hello", get=[model.layers[5].resid])
# ^^^ this call is silent
```

Also works as a decorator:

```python
@ti.context(debug=2)
def my_experiment():
    ...
```

**Implementation:**

```python
# tinyinterp/context.py (~30 lines)
from contextvars import ContextVar

_debug_level: ContextVar[int] = ContextVar("debug", default=0)

@contextmanager
def context(*, debug: int | None = None):
    tokens = []
    if debug is not None:
        tokens.append(_debug_level.set(debug))
    try:
        yield
    finally:
        for token in tokens:
            token.var.reset(token)

def _get_debug() -> int:
    """Get current debug level. Check env var on first call, then use context var."""
    level = _debug_level.get(None)
    if level is not None:
        return level
    import os
    return int(os.environ.get("DEBUG", "0"))
```

**Key design:** `_get_debug()` checks the context variable first (for scoped overrides), then falls back to the environment variable. This is exactly how tinygrad's `Context` works.

### 3.4 GRAPH=1 — intervention visualization

When set, after each `model()` call, output an SVG showing:

- The model's layer structure (vertical axis)
- Which sites had `get=` (highlighted in blue)
- Which sites had `map=` (highlighted in red)
- Data flow between sites (arrows)

```bash
GRAPH=1 python3 my_script.py
# outputs /tmp/graph.svg
```

For a causal tracing experiment, this would show:

- Embedding corrupted (red) at the top
- One layer restored (red) in the middle
- Residual stream captured (blue) at several layers

This is the interp equivalent of tinygrad's computation graph visualization: it shows what the library DID, not what the model IS.

**Implementation:** ~80 lines of SVG generation in a separate `debug.py` file. Only imported when `GRAPH=1` is set.

```python
# tinyinterp/debug.py

def render_intervention_graph(model, get_sites, map_sites, output_path="/tmp/graph.svg"):
    """Render a simple SVG showing which sites were read/modified."""
    n_layers = model.config.n_layers
    svg_lines = [f'<svg viewBox="0 0 600 {40 * n_layers + 80}" xmlns="...">']

    for i in range(n_layers):
        y = 40 + i * 40
        # Draw layer box
        color = "#fff"
        if any(s.name.startswith(f"L{i}.") for s in map_sites):
            color = "#fdd"  # red tint for mapped layers
        svg_lines.append(f'<rect x="100" y="{y}" width="400" height="30" fill="{color}" .../>')
        svg_lines.append(f'<text x="110" y="{y+20}">Layer {i}</text>')

        # Mark get sites
        for s in get_sites:
            if s.name.startswith(f"L{i}."):
                svg_lines.append(f'<circle cx="520" cy="{y+15}" r="8" fill="#4af" .../>')

        # Mark map sites
        for s in map_sites:
            if s.name.startswith(f"L{i}."):
                svg_lines.append(f'<circle cx="80" cy="{y+15}" r="8" fill="#f44" .../>')

    svg_lines.append("</svg>")
    Path(output_path).write_text("\n".join(svg_lines))
```

### 3.5 Numerical Diff CI (process replay equivalent)

tinygrad's process replay ensures that generated kernels don't change between PRs. Our equivalent ensures that activations don't change.

**The test:** for a set of reference inputs, run each interp operation and compare the output activations against a saved reference. If they differ beyond floating-point tolerance, the CI fails.

```python
# tests/test_numerical_diff.py

REFERENCE_FILE = "tests/references/gpt2_activations.pt"

def test_get_numerical_stability():
    model = ti.Model("gpt2")
    tokens = torch.tensor([[50256, 198, 15496, 995]])  # fixed input

    output = model(input_ids=tokens, get=[model.layers[5].resid])
    act = output[model.layers[5].resid]

    reference = torch.load(REFERENCE_FILE)["L5.resid"]
    assert torch.allclose(act, reference, atol=1e-5), \
        f"Numerical divergence: max diff = {(act - reference).abs().max()}"

def test_map_numerical_stability():
    model = ti.Model("gpt2")
    tokens = torch.tensor([[50256, 198, 15496, 995]])

    output = model(input_ids=tokens, map={model.layers[5].resid: ti.zero()})
    logits = output.logits

    reference = torch.load(REFERENCE_FILE)["zeroed_L5_logits"]
    assert torch.allclose(logits, reference, atol=1e-5)
```

**When references are regenerated:** only when we intentionally change something (new architecture discovery logic, different hook placement). The regeneration is an explicit CI step, not automatic.

---

## Part 4: Where these tools live in the codebase

### File structure additions

```
tinyinterp/
├── ... (existing files) ...
├── debug.py            # DEBUG output, GRAPH SVG generation    [~80 lines]
├── counters.py         # ti.Counters aggregate metrics                [~40 lines]
├── context.py          # ti.context() scoped configuration           [~30 lines]
└── sz.py               # line count enforcer                         [~10 lines]
```

**Total addition: ~150 lines.** This fits within the 2,500 line budget with room to spare.

### Where debug output is emitted

| File         | What it prints                                           | At which DEBUG level |
| ------------ | -------------------------------------------------------- | -------------------- |
| `model.py`   | Call summary (get/map/input shape)                       | ≥1                   |
| `model.py`   | Per-phase timing breakdown                               | ≥2                   |
| `batch.py`   | Planner decisions, batch grouping, prefix sharing        | ≥3                   |
| `hooks.py`   | Individual hook fire/skip/capture/map                    | ≥4                   |
| `adapter.py` | Module tree walk, component discovery, config extraction | ≥5                   |
| `debug.py`   | SVG graph generation (when GRAPH=1)                   | N/A (separate flag)  |

### How it integrates with the hot path

The critical constraint: **debug checks must not slow down production code.**

```python
# In model.py — the hot path
def __call__(self, *args, get=None, map=None, grad=False, **kwargs):
    debug = _get_debug()  # one function call, returns cached int

    if debug >= 1:
        _log_call_start(get, map, args, kwargs)

    if debug >= 2:
        t0 = time.perf_counter_ns()

    self._activate_hooks(get, map, grad)

    if debug >= 2:
        t1 = time.perf_counter_ns()

    # === THE FORWARD PASS (99% of wall time) ===
    model_output = self._model(*args, **kwargs)

    if debug >= 2:
        t2 = time.perf_counter_ns()

    activations = self._collect_and_deactivate()

    if debug >= 2:
        t3 = time.perf_counter_ns()
        _log_timing(t0, t1, t2, t3, get, activations)

    Counters.calls += 1  # always on, ~0 cost

    return Output(model_output, activations)
```

When `DEBUG=0` (the default), the cost is:

- One `_get_debug()` call (~50ns, reads a context variable)
- One `Counters.calls += 1` (~10ns, integer increment)
- Total: ~60ns overhead. Unmeasurable against a 10ms+ forward pass.

When `DEBUG>=2`, the cost adds:

- Three `time.perf_counter_ns()` calls (~100ns each)
- One print statement (~1μs)
- Total: ~1.3μs. Still unmeasurable.

When `DEBUG>=4` (hook-level tracing), the cost adds:

- One print per hook per layer (~200μs for a 32-layer model)
- This IS measurable but acceptable for debugging — you only turn this on when actively debugging.

### How counters integrate

Counters are updated in the hot path but with minimal cost:

```python
# In model.py
Counters.calls += 1
Counters.forward_time_ns += (t2 - t1)
if get:
    Counters.activations_captured += len(get)
    Counters.activations_bytes += sum(a.nbytes for a in activations.values())

# In hooks.py
if buffer_from_pool:
    Counters.buffer_pool_hits += 1
else:
    Counters.buffer_pool_misses += 1

# In batch.py
Counters.batch_fusions += (n_calls - n_forward_passes)
```

Each counter update is a single integer increment. No locking needed for the common single-threaded case (we use a simple `+=`). The thread lock in `_Counters` is only grabbed during `.reset()` and `.summary()`.

---

## Part 5: Usage examples for researchers

### "Is tinyinterp adding overhead to my forward pass?"

```python
import tinyinterp as ti
import time

model = ti.Model("gpt2")
tokens = tokenizer("Hello world", return_tensors="pt")

# Baseline: raw model
t0 = time.perf_counter()
for _ in range(100):
    _ = model.wrapped(**tokens)
baseline = (time.perf_counter() - t0) / 100

# With tinyinterp get
t0 = time.perf_counter()
for _ in range(100):
    _ = model(**tokens, get=[model.layers[5].resid])
with_get = (time.perf_counter() - t0) / 100

print(f"Baseline: {baseline*1000:.2f}ms")
print(f"With get: {with_get*1000:.2f}ms")
print(f"Overhead: {(with_get - baseline)*1000:.3f}ms ({(with_get/baseline - 1)*100:.2f}%)")
```

Or more simply, with DEBUG=2:

```python
with ti.context(debug=2):
    output = model("Hello", get=[model.layers[5].resid])
# Prints:
# [ti]   forward_pass: 4.231ms
# [ti]   TOTAL: 4.245ms (overhead: 0.014ms = 0.33%)
```

### "Is ti.batch() actually helping my sweep?"

```python
ti.Counters.reset()

# Without batching
for head in model.layers[5].attn.heads:
    out = model("Hello", map={head: ti.zero()})

print(f"Sequential: {ti.Counters.forward_passes} forward passes, "
      f"{ti.Counters.forward_time_ns / 1e6:.1f}ms")

ti.Counters.reset()

# With batching
with ti.batch():
    for head in model.layers[5].attn.heads:
        out = model("Hello", map={head: ti.zero()})

print(f"Batched: {ti.Counters.forward_passes} forward passes, "
      f"{ti.Counters.forward_time_ns / 1e6:.1f}ms")
print(f"Saved {ti.Counters.batch_fusions} forward passes via batching")
```

### "Why can't tinyinterp find my model's attention heads?"

```bash
DEBUG=5 python3 -c "
import tinyinterp as ti
model = ti.Model('my-exotic-model')
print(model.sites)
"
```

This prints the entire discovery process — every module it inspected, every pattern it tried, and why it classified each component as attn/mlp/norm/unknown.

### "Show me visually what my causal tracing experiment did"

```bash
GRAPH=1 python3 my_causal_tracing.py
open /tmp/graph.svg
```

The SVG shows each layer as a row, with blue dots for captured activations and red dots for interventions. At a glance you can see: embedding corrupted at the top, layer 7 restored in the middle, residual stream captured at every layer.

---

## Part 6: Impact on the main plan

### Line budget impact

| New file      | Lines    | Purpose                                        |
| ------------- | -------- | ---------------------------------------------- |
| `debug.py`    | ~80      | Debug output formatting + SVG graph generation |
| `counters.py` | ~40      | Aggregate metrics                              |
| `context.py`  | ~30      | Scoped configuration                           |
| **Total**     | **~150** |                                                |

Updated total: 2,070 (existing) + 150 (debug/profiling) = **2,220 lines**. Still under the 2,500 hard limit.

### Changes to existing files

- `model.py`: Add ~20 lines of `if debug >= N:` blocks in `__call__` and `generate`
- `hooks.py`: Add ~10 lines of `if debug >= 4:` in hook functions
- `adapter.py`: Add ~15 lines of `if debug >= 5:` in discovery functions
- `batch.py`: Add ~10 lines of `if debug >= 3:` in planner
- `__init__.py`: Export `Counters`, `context` (~3 lines)

**Total additions to existing files: ~58 lines.** This keeps each file within its line budget.

### Performance impact

| Scenario               | Overhead                                                      |
| ---------------------- | ------------------------------------------------------------- |
| `DEBUG=0` (default) | ~60ns per call (one context var read + one counter increment) |
| `DEBUG=1`           | ~1μs per call (one print statement)                           |
| `DEBUG=2`           | ~2μs per call (three perf_counter reads + one print)          |
| `DEBUG=3`           | ~5μs per call (planner logging, only inside `ti.batch()`)     |
| `DEBUG=4`           | ~200μs per call (one print per hook per layer)                |
| `DEBUG=5`           | ~0 at runtime (only fires during `ti.Model()` construction)   |
| `GRAPH=1`           | ~5ms per call (SVG generation to disk)                        |
| `ti.Counters`          | ~100ns per call (integer increments, always on)               |

At `DEBUG=0`, the total overhead is unmeasurable. At `DEBUG=4`, it's noticeable but acceptable for debugging sessions. This matches tinygrad's philosophy: debug output is always available, costs nothing when off, and progressively more detailed as you increase the level.

### Roadmap impact

Debug tools should be built IN PARALLEL with the features they observe, not after. Specifically:

| Milestone      | Debug tooling added                                                     |
| -------------- | ----------------------------------------------------------------------- |
| Phase 0        | `DEBUG=1,2` (call logging + timing) + `ti.Counters` + `ti.context()` |
| Phase 1        | `DEBUG=4` (hook trace) + `GRAPH=1` (intervention SVG)             |
| Phase 2        | `DEBUG=5` (architecture discovery trace)                             |
| Batching       | `DEBUG=3` (batch planner trace) + numerical diff CI                  |
| Server runtime | Server-mode stats (connected clients, queued requests, GPU utilization) |

Each milestone adds the debug tools for the features it introduces. By the time we ship v1.0, all debug levels are complete and tested.
