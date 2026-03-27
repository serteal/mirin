# tinyinterp: The Definitive Plan

_Hook any module. Read any activation. Modify any forward pass. That's all._

---

## 0. The tinygrad Philosophy (non-negotiable)

**a)** Low line count is a guiding light, but the true goal is reducing complexity and increasing readability. No code golf.

**b)** In general with no other constraints, your API should match what the user already knows. For tinygrad that's NumPy/PyTorch. For us that's the model's own module tree + HuggingFace conventions.

**c)** There's nothing special about a "Module" class in tinygrad, it's just a normal class. `nn.state.get_parameters` walks any class and finds tensors. Our equivalent: `ti.Model` wraps any `nn.Module` and makes all sub-modules hookable.

**d)** Anything you claim is a "speedup" must be benchmarked.

**e)** If your PR looks "complex", is a big diff, or adds lots of lines, it won't be reviewed or merged.

**f)** Dead code is the enemy. Less for new people to read and be confused by.

**g)** CI enforces line count. mypy strict. pre-commit hooks for linting.

---

## 1. The Core Insight

tinygrad's `get_parameters` doesn't know what a convolution weight is. It walks any Python object, finds `Tensor` attributes, returns them. It's ~10 lines.

Our `ti.Model` doesn't know what attention is. It wraps any `nn.Module`, installs hooks on every sub-module, and lets you read or modify any of them. The library is architecture-agnostic. A transformer, a Mamba model, a vision encoder, a diffusion U-Net — all work identically.

Architecture-specific knowledge (what's attention? where's the MLP? what's the unembedding matrix?) lives in **rename packs** — plain dicts that map real module names to canonical aliases. These are optional, composable, and separate from the core.

---

## 2. The Architecture: Three Layers

```
Layer 3: Rename packs (optional, ~20 lines each)
         ti.renames.llm, ti.renames.vision, user-defined dicts
         Provide canonical names: "layers", "self_attn", "lm_head"

Layer 2: Utilities (~30 lines)
         ti.find(module, pattern) — class-name search
         model.layers — shortcut (biggest ModuleList)

Layer 1: Core (~800 lines)
         ti.Model — proxy any nn.Module, hook everything
         model(input, get=, map=) — the ONE interp operation
         model.generate() — generation with interventions
         ti.replace/add/zero/scale/noise — named maps
         ti.batch() — batching hint
```

Each layer adds a little architecture knowledge. Layer 1 adds none. Layer 2 adds "there's a repeated block stack." Layer 3 adds "GPT-2 calls it `h`, LLaMA calls it `layers`."

---

## 3. Layer 1: The Core

### 3.1 ti.Model — the proxy

```python
import tinyinterp as ti

model = ti.Model(hf_model)
```

`ti.Model` does three things:

1. Stores a reference to the model (does NOT copy, reimplement, or modify it)
2. Installs a permanent no-op hook on every `nn.Module` in the tree
3. Proxies attribute access so every sub-module is navigable and hookable

```python
# Navigate using the model's OWN attribute paths
model.model.layers[5].self_attn          # LLaMA
model.transformer.h[5].attn             # GPT-2
model.backbone.layers[5].mixer          # Mamba
model.vit.encoder.layer[3]              # ViT
model.model.layers[5].block_sparse_moe  # Mixtral MoE

# Any module you can navigate to is hookable
output = model(input, get=[model.model.layers[5].self_attn])
```

### 3.2 model() — the ONE interp operation

```python
# Normal forward pass (pure passthrough, zero overhead)
logits = model(input_ids=ids, attention_mask=mask)

# Read activations
output = model(input, get=[model.model.layers[5]])
resid = output[model.model.layers[5]]        # the layer's output tensor

# Capture only and stop after the last requested site.
capture = model(input, get=[model.model.layers[5]], stop_at_last_get=True)
resid = capture[model.model.layers[5]]

# Modify activations
output = model(input, map={model.model.layers[5]: ti.zero()})

# Both at once
output = model(input,
    get=[model.model.layers[8]],
    map={model.model.layers[5]: ti.add(steer_vec)},
)

# Gradient tracking
output = model(input, get=[model.model.layers[5]], grad=True)
output[model.model.layers[5]].backward(...)
```

`stop_at_last_get=True` is an explicit capture-only optimization for normal `model(...)` calls. It
does not apply to `model.generate()`, it currently does not combine with `map=`, `grad=True`, or
`ti.batch()`, and the returned `Output` is partial: activation lookup works, but there is no final
`output.logits` because the forward did not finish.

**Input forwarding is total.** `*args` and `**kwargs` pass untouched to the real model. Any input format the model accepts works: strings (if the model handles them), token dicts, images, audio, `past_key_values`, labels. We intercept only `get=`, `map=`, and `grad=`.

**Every call is self-contained.** Hooks are activated before the forward pass and deactivated after (in a `finally` block). No state leaks. No `reset_hooks()`. If the forward pass throws an exception, hooks are still cleaned up.

### 3.3 The proxy implementation

```python
class _ModuleProxy:
    """Wraps any nn.Module. Usable as a site in get=/map=."""
    __slots__ = ("_module", "_path", "_hooks", "_renames")

    def __getattr__(self, name):
        # 1. Check renames: is `name` a canonical alias?
        real_name = self._resolve(name)

        # 2. Get the real child module
        child = getattr(self._module, real_name)
        if isinstance(child, nn.Module):
            return _ModuleProxy(child, f"{self._path}.{real_name}", self._hooks, self._renames)
        return child  # non-module attributes pass through

    def __getitem__(self, idx):
        child = self._module[idx]
        return _ModuleProxy(child, f"{self._path}.{idx}", self._hooks, self._renames)

    def _resolve(self, name):
        """Reverse-lookup: if name is a rename target, find the source."""
        for src, tgt in self._renames.items():
            if tgt == name and hasattr(self._module, src):
                return src
        return name

    # Hashable by module identity — works as dict key in get=/map=
    def __hash__(self): return id(self._module)
    def __eq__(self, other): return isinstance(other, _ModuleProxy) and self._module is other._module

    # Tab completion in IPython/Jupyter
    def __dir__(self):
        real = [n for n, _ in self._module.named_children()]
        aliases = [tgt for src, tgt in self._renames.items() if hasattr(self._module, src)]
        return sorted(set(real + aliases))

    def __repr__(self): return f"Site({self._path})"

    @property
    def weight(self): return self._module.weight
    @property
    def bias(self): return getattr(self._module, "bias", None)
```

~40 lines. Both real names and aliases work. Tab completion shows both.

### 3.4 The hook system

```python
class HookState:
    """Permanent no-op hooks on all modules. Activated per-call via flag arrays."""

    def __init__(self):
        self._id_map = {}          # id(module) → int
        self._get_flags = []       # [bool]
        self._map_fns = []         # [Callable | None]
        self._buffers = []         # [Tensor | None]
        self._n = 0

    def register(self, module: nn.Module):
        sid = self._n; self._n += 1
        self._id_map[id(module)] = sid
        self._get_flags.append(False)
        self._map_fns.append(None)
        self._buffers.append(None)

        def hook(mod, inp, out, _sid=sid):
            if self._get_flags[_sid]:
                self._buffers[_sid] = _extract(out)
            if self._map_fns[_sid] is not None:
                return _replace(out, self._map_fns[_sid](_extract(out)))

        module.register_forward_hook(hook)

    def activate(self, get_proxies, map_dict):
        if get_proxies:
            for p in get_proxies:
                self._get_flags[self._id_map[id(p._module)]] = True
        if map_dict:
            for p, fn in map_dict.items():
                self._map_fns[self._id_map[id(p._module)]] = fn

    def collect_and_deactivate(self):
        result = {}
        for i in range(self._n):
            if self._get_flags[i]:
                result[i] = self._buffers[i]
                self._buffers[i] = None
                self._get_flags[i] = False
            self._map_fns[i] = None
        return result

def _extract(output):
    """Get the main tensor from any module output format."""
    if isinstance(output, torch.Tensor): return output
    if isinstance(output, tuple): return output[0]
    if hasattr(output, "last_hidden_state"): return output.last_hidden_state
    raise TypeError(f"Cannot extract tensor from {type(output)}")

def _replace(output, new):
    """Put a modified tensor back into the original output format."""
    if isinstance(output, torch.Tensor): return new
    if isinstance(output, tuple): return (new,) + output[1:]
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = new; return output
    return new
```

~60 lines. Every module gets a permanent hook that checks two flags. Inactive hooks cost ~1μs per module (one branch check). Active hooks capture or modify the activation.

### 3.5 Named maps

```python
# tinyinterp/maps.py (~30 lines)
def replace(value):  return lambda _: value
def add(delta):      return lambda x: x + delta
def scale(factor):   return lambda x: x * factor
def zero():          return lambda _: 0
def noise(std):      return lambda x: x + torch.randn_like(x) * std

# Head-level utilities
def slice_head(tensor, head, n_heads):
    d = tensor.shape[-1] // n_heads
    return tensor[..., head*d:(head+1)*d]

def map_head(head, fn, n_heads):
    def transform(x):
        d = x.shape[-1] // n_heads
        x = x.clone()
        x[..., head*d:(head+1)*d] = fn(x[..., head*d:(head+1)*d])
        return x
    return transform
```

### 3.6 The Output object

```python
class Output:
    def __init__(self, model_output, activations, id_to_proxy):
        self._out = model_output
        self._acts = activations        # {int: Tensor}
        self._id_to_proxy = id_to_proxy # {id(module): int}

    def __getattr__(self, name):
        return getattr(self._out, name)   # delegates .logits, .loss, etc.

    def __getitem__(self, proxy):
        sid = self._id_to_proxy[id(proxy._module)]
        return self._acts[sid]
```

~15 lines. Wraps the model output + captured activations. `output.logits` works like normal HuggingFace. `output[some_module]` returns the captured activation.

---

## 4. Layer 2: Utilities

### 4.1 model.layers — the one structural shortcut

```python
@property
def layers(self):
    """Shortcut to the biggest ModuleList in the model."""
    if self._layers_proxy is None:
        best = None
        for name, mod in self._model.named_modules():
            if isinstance(mod, nn.ModuleList):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        if best is None:
            raise AttributeError("No ModuleList found. Navigate directly.")
        self._layers_path, self._layers_mod = best
        self._layers_proxy = _ModuleListProxy(self._layers_mod, self._hooks, self._renames)
    return self._layers_proxy
```

~15 lines. Finds the layer stack on any architecture. The ONLY structural discovery in the library.

### 4.2 ti.find — class-name search

```python
def find(proxy, pattern: str):
    """Find a child module whose class name contains pattern (case-insensitive)."""
    for name, child in proxy._module.named_children():
        if pattern.lower() in type(child).__name__.lower():
            return getattr(proxy, name)
    return None

def find_all(proxy, pattern: str) -> list:
    """Find ALL children matching pattern."""
    return [getattr(proxy, n) for n, c in proxy._module.named_children()
            if pattern.lower() in type(c).__name__.lower()]

def children(proxy) -> list[tuple[str, str]]:
    """List (name, class_name) for all children. For exploration."""
    return [(n, type(c).__name__) for n, c in proxy._module.named_children()]
```

~15 lines. `ti.find(model.layers[5], "attn")` works on any model that has an attention-like module. When it fails, you navigate directly.

---

## 5. Layer 3: Rename Packs

### 5.1 How renames work

Renames are a flat `dict[str, str]` mapping `{real_name: canonical_name}`. Passed at load time. Both old and new names work after renaming.

```python
# Without renames: real paths only
model = ti.Model(hf_model)
model.transformer.h[5].attn          # GPT-2's real path

# With renames: canonical names become aliases
model = ti.Model(hf_model, rename={"h": "layers", "transformer": "model", "attn": "self_attn"})
model.model.layers[5].self_attn      # alias works
model.transformer.h[5].attn          # real path ALSO still works
```

The `_resolve` method in `_ModuleProxy` handles the lookup: when you access `.layers`, it checks if any rename maps `something → "layers"`, finds `"h" → "layers"`, checks if the real module has an attribute called `h`, and returns it. ~5 lines of code.

### 5.2 Pre-built rename packs

```python
# tinyinterp/renames.py

llm = {
    # Model container
    "transformer": "model", "gpt_neox": "model", "decoder": "model",
    "language_model": "model",
    # Layer stack
    "h": "layers", "blocks": "layers",
    # Attention
    "attn": "self_attn", "self_attention": "self_attn",
    "attention": "self_attn", "norm_attn_norm": "self_attn",
    # MLP
    "block_sparse_moe": "mlp", "ffn": "mlp",
    # Final norm
    "ln_f": "ln_final", "norm_f": "ln_final", "final_layer_norm": "ln_final",
    "norm": "ln_final",
    # LM head
    "embed_out": "lm_head",
    # Embeddings
    "wte": "embed_tokens", "embed_in": "embed_tokens",
    "word_embeddings": "embed_tokens",
}

vision = {
    "patch_embed": "embed", "patch_embedding": "embed",
    "blocks": "layers",
    "head": "classifier", "heads": "classifier",
    "norm": "ln_final",
}
```

~30 lines total. Each pack is a plain dict. Users can compose them:

```python
model = ti.Model(hf_model, rename={**ti.renames.llm, "custom_thing": "my_alias"})
```

### 5.3 Why this is better than nnterp

nnterp's approach is 760 lines: a `RenameConfig` dataclass (12 fields), a `get_rename_dict` function (30 lines), global name lists (40+ entries), an `AttnProbFunction` ABC, shape validation (160 lines), renaming error classes, and accessor properties (60 lines).

Our approach is the same idea but stripped to its essence:

- **A dict replaces `RenameConfig` + `get_rename_dict` + all the global name lists.** One data structure instead of six.
- **`_ModuleProxy._resolve` replaces NNsight's rename machinery.** 5 lines instead of NNsight's module rename system.
- **No shape validation at load time.** If a rename doesn't apply (the real name doesn't exist on the module), it silently does nothing. No errors, no `RenamingError`, no `check_io`. The researcher sees the problem when they try to access a module that doesn't exist, and the error message lists what DOES exist.
- **No `AttnProbFunction`.** Attention probability access is out of scope for the core. If the model supports `output_attentions=True`, researchers use it directly.

### 5.4 How cross-model code works

```python
# Portable code: use renames + model.layers
model = ti.Model(hf_model, rename=ti.renames.llm)

for layer in model.layers:
    output = model(input, get=[layer])
    print(output[layer].shape)

# Semi-portable: use ti.find for sub-components
attn = ti.find(model.layers[5], "attn")
if attn:
    output = model(input, get=[attn])

# Model-specific: use real paths
output = model(input, get=[model.model.layers[5].self_attn.q_proj])
```

Three levels of portability, all available simultaneously. The researcher chooses.

---

## 6. Input Handling: Total Transparency

Our `__call__` does NOT tokenize, move devices, create masks, or convert formats. The model handles its own inputs.

```python
def __call__(self, *args, get=None, map=None, grad=False, **kwargs):
    self._hooks.activate(
        [p for p in get] if get else None,
        {p: fn for p, fn in map.items()} if map else None,
    )
    try:
        with torch.no_grad() if not grad else nullcontext():
            model_output = self._model(*args, **kwargs)
    finally:
        acts = self._hooks.collect_and_deactivate()

    if not get and not map:
        return model_output  # pure passthrough
    return Output(model_output, acts, self._hooks._id_map)
```

This means:

```python
# Token dict (probelab's format)
output = model(**batch_dict, get=[model.layers[12]])

# Raw token IDs
output = model(input_ids=ids, attention_mask=mask, get=[...])

# String (if model handles it)
output = model("Hello world", get=[...])

# Vision model
output = model(pixel_values=images, get=[model.vit.encoder.layer[3]])

# With any HF kwarg
output = model(input_ids=ids, use_cache=False, output_attentions=True, get=[...])
```

---

## 7. Generation

```python
def generate(self, *args, get=None, map=None, stream=False, **kwargs):
    """HuggingFace generate() with get=/map= applied at every step."""
    if map:
        # Install persistent maps for the duration of generation
        for proxy, fn in map.items():
            sid = self._hooks._id_map[id(proxy._module)]
            self._hooks._map_fns[sid] = fn
    try:
        tokens = self._model.generate(*args, **kwargs)
    finally:
        # Clear maps
        for i in range(self._hooks._n):
            self._hooks._map_fns[i] = None
    return tokens
```

```python
# Steering during generation
tokens = model.generate("The Eiffel Tower is in",
    max_new_tokens=20,
    map={model.layers[10]: ti.add(french_direction)},
)
```

---

## 8. Performance

### 8.1 Permanent hooks with flag arrays

Hooks installed ONCE at `ti.Model()` time. Each `model()` call flips booleans, doesn't register/remove hooks. Cost of inactive hooks: ~1μs per module per forward pass (one branch check).

### 8.2 ti.batch() for sweeps

```python
with ti.batch():
    for head_idx in range(n_heads):
        head = model.model.layers[5].self_attn
        out = model(input, map={head: ti.map_head(head_idx, ti.zero(), n_heads)})
        results[head_idx] = metric(out.logits)
```

Inside `ti.batch()`, calls are accumulated and batched into minimal forward passes. Without it, the loop runs eagerly. This is tinygrad's `TinyJit` pattern: laziness is opt-in.

### 8.3 Dataset collection stays a normal loop

```python
for batch, idx in _iter_batches(tokens, batch_size):
    output = model(**batch, use_cache=False, get=[model.layers[8]])
    acts = output[model.layers[8]]
```

`model.stream()` was prototyped and benchmarked, but it did not show a consistent improvement
over the plain Python loop on real workloads. Following the tinygrad philosophy, the simpler
version stays and the helper is removed.

### 8.4 Server mode

```python
server = ti.Server("meta-llama/Llama-3.2-1B", device="cuda")
plan = server.compile(
    get=["model.layers.8"],
    output={"logits": False, "activations": True},
)

collector = server.open_collector(plan=plan, stop_at_last_get=True)
for batch_out in collector.run(dataset):
    acts = batch_out.activations["model.layers.8"]

session = server.open_session(plan=plan, cache="dynamic")
server.prefill(session, input_ids=prompt_ids, attention_mask=prompt_mask)
step = server.decode([session], max_new_tokens=1)[0]
token = step.token_ids
```

---

## 9. Debugging & Profiling

Controlled by `DEBUG` / `GRAPH` env vars or `ti.context(debug=N, graph=...)`:

| Level | Shows                                            |
| ----- | ------------------------------------------------ |
| 1     | Call summary (get/map/input shape)               |
| 2     | Per-phase timing (hook overhead vs forward time) |
| 3     | Batch planner decisions                          |
| 4     | Individual hook fire/skip/capture                |
| 5     | Layer stack discovery details                    |

`ti.Counters` tracks aggregate stats (calls, forward time, activations captured, buffer pool hits). Always on, ~100ns overhead.

---

## 10. Integration: probelab as the test case

probelab currently has `models/architectures.py` (115 lines) + `models/hooks.py` (171 lines) = 286 lines of model handling. With tinyinterp:

```python
# probelab/backends/tinyinterp.py (~30 lines)

class TinyinterpBackend:
    name = "tinyinterp"

    def stream_raw(self, model_obj, tokens, layers, batch_size, **kwargs):
        model = ti.Model(model_obj)
        sites = [model.layers[l] for l in layers]

        for batch, idx in _iter_batches(tokens, batch_size):
            batch_inputs = {k: v for k, v in batch.items() if k != "detection_mask"}
            output = model(**batch_inputs, use_cache=False, get=sites)
            acts = torch.stack([output[s] for s in sites], dim=0)

            # ... existing flat+offsets construction ...
            yield flat_data, flat_det, offsets, idx
```

probelab's datasets, tokenization, masks, probes, metrics, and pooling are completely unchanged. Only the model access layer is replaced. The backend automatically supports every architecture because `model.layers[l]` works on everything.

---

## 11. File Structure and Line Budget

```
tinyinterp/
├── __init__.py      [  20 lines]  Exports: Model, Server, find, replace/add/zero/...
├── model.py         [ 180 lines]  Model + _ModuleProxy + _ModuleListProxy + Output
├── hooks.py         [  80 lines]  HookState + _extract + _replace
├── maps.py          [  40 lines]  Named maps + head utilities
├── renames.py       [  40 lines]  Pre-built rename packs (llm, vision)
├── utils.py         [  20 lines]  find, find_all, children
├── batch.py         [ 120 lines]  ti.batch() context + planning
├── debug.py         [  80 lines]  DEBUG output + GRAPH rendering
├── counters.py      [  40 lines]  ti.Counters
├── context.py       [  25 lines]  ti.context()
├── server/
│   ├── inference.py [ 350 lines]
│   ├── plans.py     [ 180 lines]
│   ├── sessions.py  [ 140 lines]
│   ├── collector.py [  60 lines]
│   └── results.py   [  20 lines]
└── sz.py            [  10 lines]  CI line count enforcer
                          CORE: ~725 lines
                        SERVER: ~400 lines
                     DEBUG/AUX: ~155 lines
                         TOTAL: ~1,280 lines
```

Hard limit: 2,000 lines. We're at ~1,280. Plenty of room for edge cases without bloating.

---

## 12. Code Style

```python
# GOOD: clear, one thing per function, obvious
def _extract(output):
    if isinstance(output, torch.Tensor): return output
    if isinstance(output, tuple): return output[0]
    if hasattr(output, "last_hidden_state"): return output.last_hidden_state
    raise TypeError(f"Cannot extract tensor from {type(output)}")

# BAD: clever, compressed
extract = lambda o: o if isinstance(o, torch.Tensor) else o[0] if isinstance(o, tuple) else getattr(o, "last_hidden_state")
```

- Functions under 30 lines
- Type hints on all public functions
- Errors name what's wrong, what's expected, and what to do
- No global mutable state (hook state lives on the Model object)
- Dependencies: `torch` only for core, `transformers` optional for `ti.Model(string)` and the HuggingFace server path

---

## 13. Development Roadmap

| Phase | What                                                                              | Lines | Duration |
| ----- | --------------------------------------------------------------------------------- | ----- | -------- |
| 0     | `model.py` + `hooks.py` + `maps.py`. `model(input, get=[module])` works on GPT-2. | ~300  | 2 weeks  |
| 1     | `map=` support. `renames.py` with LLM pack. Replicate IOI patching.               | +160  | 2 weeks  |
| 2     | `model.layers`. `ti.find`. `model.generate(map=)`. Test on 10+ architectures.     | +100  | 2 weeks  |
| 3     | `ti.batch()`. Benchmark vs TransformerLens/nnterp (built on NNsight). `model.stream()` removed after benchmarks showed no consistent win vs a normal loop. | +200  | 3 weeks  |
| 4     | Inference server. Compiled plans, collector mode, explicit prefill/decode.        | +400  | 3 weeks  |
| 5     | Debug tools. probelab integration. Migration guide. Release.                      | +155  | 2 weeks  |

**14 weeks to v1.0.**

---

## 14. The Explain-It-In-One-Paragraph Test

tinyinterp wraps any PyTorch model and lets you read and modify internal activations during the forward pass. Navigate to any module using the model's own attribute paths — `model.model.layers[5].self_attn` on LLaMA, `model.transformer.h[5].attn` on GPT-2 — and pass it to `get=` to capture its output or `map=` to transform it. For portable code across architectures, pass a rename dict that aliases model-specific names to canonical ones: `ti.Model(m, rename=ti.renames.llm)` makes `model.model.layers[5].self_attn` work on both. The `model.layers` shortcut finds the repeated block stack on any model. Every call is self-contained (no state leaks), inputs pass untouched to the real model, and the library is ~1,300 lines of Python.
