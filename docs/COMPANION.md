# tinyinterp Developer's Companion

_Everything you need to think about while building this that doesn't fit in the plan._

---

## 1. Architecture Detection Without Hardcoding

### Why `arch=` is wrong

The `arch="llama"` parameter is probelab's approach: a dict of 3 named architectures with hardcoded lambda accessors. It works for 3 architectures. It won't work for 50+. And it forces users to know which architecture family their model belongs to before they can use the library — violating the tinygrad principle of "just works."

### The real structure of HuggingFace models

After reviewing GPT-2, LLaMA, Mistral, Gemma, and others, the layer containers are:

```
GPT-2:    model.transformer.h[i]        (ModuleList named "h")
LLaMA:    model.model.layers[i]         (ModuleList named "layers")
Mistral:  model.model.layers[i]         (same as LLaMA)
Gemma:    model.model.layers[i]         (same as LLaMA)
Gemma 3:  model.model.language_model.layers[i]  (nested deeper)
BERT:     model.bert.encoder.layer[i]   (ModuleList named "layer")
T5:       model.encoder.block[i]        (ModuleList named "block")
GPT-Neo:  model.transformer.h[i]        (same as GPT-2)
Phi:      model.model.layers[i]         (same as LLaMA)
Mamba:    model.backbone.layers[i]      (ModuleList named "layers")
```

**The pattern:** there is ALWAYS a `nn.ModuleList` somewhere in the tree that contains the repeated layers. It might be called `h`, `layers`, `layer`, `block`, but it's always a ModuleList and it's always the biggest one (by count of children).

### The discovery algorithm: walk the module tree

Instead of a hardcoded dict, we WALK the model's module tree and discover the structure:

```python
def discover_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Find the main layer container by walking the module tree.

    Returns the ModuleList and its path (e.g. "model.layers").
    """
    candidates = []
    for path, module in _walk_named_modules(model):
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            candidates.append((path, module, len(module)))

    if not candidates:
        raise ValueError("No ModuleList found in model. Cannot auto-detect layers.")

    # The layer container is the longest ModuleList
    # (attention heads are also ModuleLists but shorter)
    candidates.sort(key=lambda x: x[2], reverse=True)
    path, module_list, count = candidates[0]
    return module_list, path
```

This works for every model listed above without any hardcoding. A new architecture (say, a future LLaMA-4 with a different nesting depth) works automatically as long as it uses a ModuleList for its layers.

### Discovering sub-components within a layer

Once we have the layer ModuleList, we need to find attention, MLP, and layer norms within each layer. Again, we walk the tree:

```python
def discover_layer_components(layer: nn.Module) -> dict[str, str]:
    """Discover attention, MLP, and norms within a single layer.

    Returns a dict mapping canonical names to attribute paths.
    """
    components = {}
    for name, child in layer.named_children():
        child_type = type(child).__name__.lower()

        # Attention: class name contains "attn" or "attention"
        if "attn" in child_type or "attention" in child_type:
            components["attn"] = name

        # MLP: class name contains "mlp" or "feedforward" or "ff"
        elif "mlp" in child_type or "feedforward" in child_type or child_type == "ff":
            components["mlp"] = name

        # Layer norms: class name contains "norm" or "layernorm" or "rmsnorm"
        elif "norm" in child_type or "layernorm" in child_type or "rmsnorm" in child_type:
            if "ln1" not in components and "ln_1" not in components:
                components["ln1"] = name   # first norm = pre-attention
            else:
                components["ln2"] = name   # second norm = pre-MLP

    return components
```

### Discovering attention internals

Within the attention module, we need Q/K/V projections and the output projection:

```python
def discover_attn_components(attn: nn.Module) -> dict[str, str]:
    components = {}
    for name, child in attn.named_children():
        name_lower = name.lower()
        if isinstance(child, nn.Linear):
            if "q" in name_lower and "k" not in name_lower:
                components["q_proj"] = name
            elif "k" in name_lower and "q" not in name_lower:
                components["k_proj"] = name
            elif "v" in name_lower:
                components["v_proj"] = name
            elif "o" in name_lower or "out" in name_lower or "dense" in name_lower:
                components["o_proj"] = name
            # Some models use a single QKV projection
            elif "qkv" in name_lower or "c_attn" in name_lower:
                components["qkv_proj"] = name
    return components
```

### The config as backup

When auto-detection is ambiguous (rare), we fall back to the model's config:

```python
def discover_config(model: nn.Module) -> dict:
    config = getattr(model, "config", None)
    if config is None:
        return {}

    result = {}
    for attr, canonical in [
        ("num_hidden_layers", "n_layers"),
        ("n_layer", "n_layers"),          # GPT-2 uses n_layer
        ("num_layers", "n_layers"),
        ("num_attention_heads", "n_heads"),
        ("n_head", "n_heads"),            # GPT-2 uses n_head
        ("hidden_size", "d_model"),
        ("n_embd", "d_model"),            # GPT-2 uses n_embd
        ("intermediate_size", "d_mlp"),
        ("n_inner", "d_mlp"),             # GPT-2 uses n_inner
        ("num_key_value_heads", "n_kv_heads"),
    ]:
        if hasattr(config, attr):
            result[canonical] = getattr(config, attr)

    # Derive d_head
    if "d_model" in result and "n_heads" in result:
        result["d_head"] = result["d_model"] // result["n_heads"]

    return result
```

### When discovery fails

If the model has a truly exotic structure that auto-detection can't handle, the user can pass a discovery hint — but NOT an `arch=` string. Instead, they pass the actual paths:

```python
# For 99% of models: auto-detection works
model = ti.Model(hf_model)

# For exotic models: pass a hint dict (NOT an arch name)
model = ti.Model(exotic_model, layers="model.custom_backbone.blocks")
```

This is ONE kwarg, not a full architecture spec. It tells the discovery algorithm where to find the layer container, and everything else is still auto-detected within each layer.

---

## 2. The Forward Pass and Input Handling

### The zero-processing principle

Our `__call__` does NOT:

- Tokenize strings
- Move tensors to devices
- Create attention masks
- Handle padding
- Convert between formats

The underlying model does all of this. We just forward.

### How to handle string inputs

Some HuggingFace models accept strings directly (via a built-in tokenizer pipeline), some don't. We don't care. If `model("Hello")` works on the raw model, it works on our wrapper. If it doesn't, it doesn't. We're not a tokenization library.

```python
# This works if the underlying model supports it
output = model("Hello world", get=[model.layers[5].resid])

# This ALWAYS works (pre-tokenized input)
tokens = tokenizer("Hello world", return_tensors="pt").to(model.device)
output = model(**tokens, get=[model.layers[5].resid])
```

### How to handle probelab-style dict inputs

probelab prepares batches as dicts with keys `input_ids`, `attention_mask`, and `detection_mask`. The HF model accepts the first two and ignores the third (it's probelab's metadata). Our wrapper passes the whole dict through — the model takes what it needs, ignores the rest. If the model raises a TypeError for unknown kwargs, we should NOT silently eat it. Let the error propagate. The user or the library above us (probelab) is responsible for filtering their own metadata.

Actually — looking at probelab's code, they DO filter before calling:

```python
model_inputs = {k: v for k, v in batch_inputs.items() if k != "detection_mask"}
```

So probelab would do the same filtering before calling `model(**batch_inputs, get=[...])`. Our wrapper doesn't need to handle this.

### The `use_cache=False` optimization

probelab sets `use_cache=False` when extracting activations because KV cache allocation is wasted memory during activation collection. Should we do this automatically?

**No.** This is a caller's concern. probelab knows it doesn't need the KV cache. A researcher doing generation DOES need it. We don't know which case we're in. If the caller wants `use_cache=False`, they pass it:

```python
output = model(input_ids=ids, attention_mask=mask, use_cache=False, get=[...])
```

### Handling model output types

HuggingFace models return various types: `CausalLMOutputWithPast`, `BaseModelOutput`, `SequenceClassifierOutput`, plain tuples, etc. Our `Output` wrapper must handle all of them.

```python
class Output:
    def __init__(self, model_output, activations):
        self._model_output = model_output
        self._activations = activations

    def __getattr__(self, name):
        # Delegate ALL attribute access to the model output
        # This means output.logits, output.loss, output.hidden_states
        # all work exactly as they would on the raw model output
        return getattr(self._model_output, name)

    def __getitem__(self, site):
        if isinstance(site, Site):
            return self._activations[site]
        # If it's an integer or slice, delegate to model output
        # (some models return tuples)
        return self._model_output[site]
```

**Critically:** if `get=None` and `map=None`, we return the raw model output directly (not wrapped in Output). This means the wrapper is completely invisible when not used. The model behaves exactly like the unwrapped version.

---

## 3. The Hook System in Detail

### Where to place hooks

For reading the residual stream after a layer, we hook the layer MODULE (the block itself), not a sub-component. The layer's forward hook gives us the block's output, which is the residual stream after that block.

For reading attention patterns, we need to hook the attention module specifically and access its `attn_weights` output. Different models return this differently:

- Some models return `(hidden_states, attn_weights, past_kv)` from the attention forward
- Some only return `hidden_states` by default and require `output_attentions=True`
- Flash attention implementations may not return attention weights at all

**Our approach:** for attention patterns, we set `output_attentions=True` on the model config before the forward pass, then hook the attention module's output. If the model doesn't support `output_attentions`, we raise a clear error: "Attention pattern access not available with this model's attention implementation (Flash Attention). Use `get=[model.layers[5].attn]` for the attention output instead."

### The permanent hook with flags, in full detail

```python
class HookState:
    """Mutable state for all hooks. One instance per Model."""
    __slots__ = ("get_flags", "map_fns", "buffers", "n_sites")

    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.get_flags = [False] * n_sites
        self.map_fns = [None] * n_sites       # None means "don't map"
        self.buffers = [None] * n_sites        # Will hold captured tensors

    def activate(self, get_sites: list[int], map_entries: dict[int, Callable]):
        for sid in get_sites:
            self.get_flags[sid] = True
        for sid, fn in map_entries.items():
            self.map_fns[sid] = fn

    def deactivate_and_collect(self) -> dict[int, Tensor]:
        result = {}
        for i in range(self.n_sites):
            if self.get_flags[i]:
                result[i] = self.buffers[i]
                self.buffers[i] = None
                self.get_flags[i] = False
            self.map_fns[i] = None
        return result
```

The hook function itself:

```python
def _make_hook(state: HookState, site_id: int, extract_fn=None):
    """Create a permanent hook for a specific site.

    extract_fn: optional callable to extract the activation from the module output.
    For most modules, output IS the activation (identity).
    For attention, output might be a tuple and we want output[0].
    """
    if extract_fn is None:
        extract_fn = lambda x: x

    def hook(module, input, output):
        act = extract_fn(output)

        if state.get_flags[site_id]:
            # Capture: store reference (zero-copy when possible)
            if state.map_fns[site_id] is not None:
                # If we're also mapping, capture BEFORE the map
                state.buffers[site_id] = act.detach().clone()
            else:
                state.buffers[site_id] = act

        if state.map_fns[site_id] is not None:
            new_act = state.map_fns[site_id](act)
            # Reconstruct the output with the modified activation
            if isinstance(output, tuple):
                return (new_act,) + output[1:]
            return new_act

        return output

    return hook
```

### The extract_fn pattern

Different modules return different shapes. The hook must know how to extract the activation:

| Module type       | Output format                                  | extract_fn                                                |
| ----------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Transformer block | `(hidden_states, ...)` or just `hidden_states` | `lambda o: o[0] if isinstance(o, tuple) else o`           |
| Attention module  | `(attn_output, attn_weights, past_kv)`         | `lambda o: o[0]` for output, `lambda o: o[1]` for pattern |
| MLP module        | `hidden_states`                                | identity                                                  |
| LayerNorm         | `hidden_states`                                | identity                                                  |
| Embedding         | `hidden_states`                                | identity                                                  |

These are discovered during model introspection by checking the module type.

### The get-before-map ordering problem

When both `get` and `map` are requested on the SAME site, what does `get` return? The value BEFORE the map (the clean activation) or AFTER (the modified one)?

**Answer: BEFORE.** The researcher wants to know what the activation was, then modify it. If they wanted the modified value, they can compute it themselves (they have the map function). This matches the intuition: "get the activation at this site, and also transform it."

This is why the hook captures `.detach().clone()` when both get and map are active on the same site. The clone is necessary because the map will modify the tensor in-place (or return a new tensor that replaces it).

---

## 4. The Site Object in Detail

### What a Site IS

A Site is a frozen reference to a specific location in the model where activations can be observed or modified. It is:

- **Hashable** (can be a dict key)
- **Comparable** (can check equality)
- **Inspectable** (has name, module path, shape)
- **NOT mutable** (you can't change what a site points to)

```python
@dataclass(frozen=True)
class Site:
    name: str               # canonical name: "L5.resid", "L5.attn.head[3]"
    id: int                 # integer index into the hook state arrays
    module_path: str        # dot-separated path to the actual nn.Module
    _model_ref: Any         # weak reference to the Model (for shape computation)

    @property
    def shape(self) -> tuple[str, ...]:
        """Lazy shape description. Requires one forward pass if not cached."""
        ...

    def __repr__(self) -> str:
        return f"Site({self.name!r})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Site):
            return self.name == other.name
        return NotImplemented
```

### The Layer object (navigation node, not a site itself)

```python
class Layer:
    """Navigation node for a model layer. NOT a Site — you access sites through it."""

    def __init__(self, idx: int, sites: dict[str, Site]):
        self.idx = idx
        self._sites = sites

    @property
    def resid(self) -> Site:
        return self._sites["resid"]

    @property
    def attn(self) -> AttnAccessor:
        return AttnAccessor(self._sites, self.idx)

    @property
    def mlp(self) -> MlpAccessor:
        return MlpAccessor(self._sites, self.idx)

    # ... ln1, ln2, etc.


class AttnAccessor:
    """Navigation node for attention components."""

    # The attn accessor itself is ALSO a Site (the attention output)
    def __init__(self, sites, layer_idx):
        self._sites = sites
        self._layer_idx = layer_idx
        self._site = sites.get(f"L{layer_idx}.attn")

    # When used as a Site (in get/map), return the attn output site
    @property
    def name(self): return self._site.name
    @property
    def id(self): return self._site.id
    def __hash__(self): return hash(self._site)
    def __eq__(self, other): return self._site == other

    # Navigation to sub-components
    @property
    def pattern(self) -> Site:
        return self._sites[f"L{self._layer_idx}.attn.pattern"]

    @property
    def q(self) -> Site:
        return self._sites[f"L{self._layer_idx}.attn.q"]

    def head(self, idx: int) -> Site:
        return self._sites[f"L{self._layer_idx}.attn.head[{idx}]"]

    @property
    def heads(self) -> list[Site]:
        n = ...  # from config
        return [self.head(i) for i in range(n)]
```

This is ~60 lines of navigation code. No complexity — just attribute routing.

### The head site problem

A "head" is not a separate module. It's a slice of the attention output along the head dimension. You can't hook a single head — you hook the attention module and then slice.

For `get=[model.layers[5].attn.head[3]]`:

- We hook the attention module
- In the hook, we extract `output[:, :, 3, :]` (the head-3 slice)
- We store that slice

For `map={model.layers[5].attn.head[3]: ti.zero()}`:

- We hook the attention module
- In the hook, we apply `output[:, :, 3, :] = 0`
- We return the modified full output

This means head-level sites are **virtual sites** — they don't correspond to a physical module, they correspond to a slice of a physical module's output. The hook system needs to handle this:

```python
# During site registration
if site.is_virtual:
    # Register on the parent module, with a slice spec
    parent_site = site.parent  # e.g., L5.attn
    parent_hook[parent_site].add_virtual(site.id, site.slice_spec)
```

### String-to-Site resolution

For programmatic access:

```python
def site(self, pattern: str) -> Site | list[Site]:
    """Resolve a string pattern to Site objects.

    "L5.resid"        → single Site
    "L*.resid"        → list of Sites (one per layer)
    "L5.attn.head[*]" → list of Sites (one per head)
    "L0:5.resid"      → list of Sites (layers 0-4)
    """
    if "*" not in pattern and ":" not in pattern:
        # Exact match
        if pattern in self._site_map:
            return self._site_map[pattern]
        raise KeyError(f"Site {pattern!r} not found. Available: {self._list_similar(pattern)}")

    # Glob match
    import fnmatch
    matches = [s for name, s in self._site_map.items() if fnmatch.fnmatch(name, pattern)]
    if not matches:
        raise KeyError(f"No sites match pattern {pattern!r}")
    return matches
```

---

## 5. Dependencies: The tinygrad Way

### Core dependencies: as few as possible

tinygrad's core depends on **nothing** (not even numpy — it's optional). Our core depends on **torch** only. Not transformers, not tokenizers, not safetensors.

```toml
[project]
dependencies = ["torch>=2.0"]

[project.optional-dependencies]
transformers = ["transformers>=4.30"]   # for ti.Model(string_name)
server = ["pyzmq>=25"]                  # for ti.Server
dev = ["pytest", "mypy", "ruff"]
```

When the user does `ti.Model("gpt2")`, we try to import transformers. If it's not installed, we raise:

```
ImportError: ti.Model("gpt2") requires `transformers` to load models by name.
Install with: pip install tinyinterp[transformers]
Or pass an already-loaded model: ti.Model(your_model)
```

When the user does `ti.Model(my_already_loaded_model)`, we DON'T import transformers at all. This means tinyinterp can be used with ANY PyTorch model, even non-HuggingFace ones, with zero extra dependencies.

### Import structure

```python
# tinyinterp/__init__.py
# Lazy imports everywhere — nothing heavy at import time

from .model import Model
from .site import Site
from .maps import replace, add, scale, zero, noise
from .batch import batch

# These are conditional — only imported when used
def connect(url, **kwargs):
    from .server.client import RemoteModel
    return RemoteModel(url, **kwargs)

def Server(model, **kwargs):
    from .server.engine import InterpServer
    return InterpServer(model, **kwargs)
```

At `import tinyinterp` time, we import: model.py, site.py, maps.py, batch.py. That's ~600 lines of Python. No torch imports until `ti.Model()` is called. Fast import.

---

## 6. Specific Implementation Problems and Solutions

### Problem: Output tuple unpacking varies by model

Some models return `(hidden_states,)`, others `(hidden_states, attn_weights)`, others `(hidden_states, attn_weights, past_kv)`. The hook must return the SAME tuple structure with a modified first element.

**Solution:** The hook inspects the output type and reconstructs it:

```python
def _replace_in_output(output, new_activation):
    """Replace the activation in whatever output format the module returns."""
    if isinstance(output, tuple):
        return (new_activation,) + output[1:]
    if isinstance(output, dict):
        # Some modules return dicts (rare)
        output = dict(output)
        # Find the key that has the right shape
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.shape == new_activation.shape:
                output[k] = new_activation
                break
        return output
    return new_activation
```

### Problem: Flash Attention doesn't return attention weights

**Solution:** Don't offer attention pattern sites when Flash Attention is active. During discovery, check:

```python
attn_module = layer.self_attn
if hasattr(attn_module, "_flash_attn_uses_top_left_mask") or "flash" in type(attn_module).__name__.lower():
    # Don't register L{i}.attn.pattern site
    pass
```

If the user requests `model.layers[5].attn.pattern` when it's unavailable, raise a clear error.

### Problem: Multi-GPU models with device_map="auto"

Layers live on different GPUs. A hook on layer 5 (GPU 0) captures a tensor on GPU 0. A hook on layer 20 (GPU 1) captures on GPU 1. The user gets tensors on different devices.

**Solution:** probelab's approach is correct — detect multi-GPU and consolidate to CPU:

```python
def _resolve_target_device(self):
    devices = set()
    for site in self._active_get_sites:
        module = self._get_module(site)
        param = next(module.parameters(), None)
        if param is not None:
            devices.add(param.device)

    if len(devices) == 1:
        return devices.pop()  # single GPU, keep on GPU
    return torch.device("cpu")  # multi-GPU, consolidate to CPU
```

### Problem: Quantized models (GPTQ, AWQ, bitsandbytes)

Quantized models store weights in int4/int8 but compute in fp16/fp32. Activations are in the compute dtype, not the storage dtype. Hooks capture activations in the compute dtype, which is correct — the researcher wants the actual floating-point activations.

No special handling needed. The hooks see the dequantized activations.

### Problem: PEFT/LoRA models

LoRA wraps the base model in a `PeftModel`. The layers are accessed via `model.get_base_model().model.layers`. probelab handles this with `model.get_base_model()`.

Our discovery algorithm should unwrap PEFT models:

```python
def _unwrap_model(model):
    """Unwrap PEFT, DataParallel, etc. to get the base model."""
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "module"):  # DataParallel
        model = model.module
    return model
```

### Problem: Encoder-decoder models (T5, BART)

These have BOTH an encoder layer stack AND a decoder layer stack. Our site tree should reflect this:

```python
model.encoder.layers[i].resid     # encoder layer i
model.decoder.layers[i].resid     # decoder layer i
model.decoder.layers[i].cross_attn  # cross-attention
```

The discovery algorithm finds TWO ModuleLists (one in encoder, one in decoder) and creates separate layer trees for each.

### Problem: Vision Transformers (ViT, CLIP)

ViTs have layers but no "attention heads" in the same sense. The residual stream is patch embeddings. Our site tree still works:

```python
model.layers[i].resid     # patch representation after layer i
model.layers[i].attn      # self-attention output
model.layers[i].mlp       # MLP output
```

The site names are the same. The semantics are different (patches instead of tokens). The library doesn't care — it hooks modules and captures tensors.

### Problem: Mamba/SSM models

Mamba doesn't have attention. It has SSM blocks. The site tree adapts:

```python
model.layers[i].resid     # residual stream (always exists)
model.layers[i].ssm       # SSM output (instead of attn)
model.layers[i].conv      # conv output (Mamba-specific)
model.layers[i].mlp       # MLP output (if present)
# model.layers[i].attn   — does NOT exist, accessing it raises clear error
```

The discovery algorithm detects that there's no attention module in the layer and omits the attn site.

---

## 7. The Weight Access System

### Why reshape weights

HuggingFace stores attention weights as `[d_model, d_model]` (one big linear layer). Researchers want `[n_heads, d_model, d_head]` (per-head view). The weight access system reshapes on read:

```python
class WeightAccessor:
    def __getitem__(self, pattern: str) -> Tensor:
        site_name, weight_name = self._parse(pattern)  # "L5.attn.W_Q" → (L5.attn, W_Q)
        module = self._resolve_module(site_name)
        raw = self._get_raw_weight(module, weight_name)
        return self._reshape(raw, weight_name)

    def _reshape(self, raw: Tensor, name: str) -> Tensor:
        if name in ("W_Q", "W_K", "W_V"):
            # [d_model, n_heads * d_head] → [n_heads, d_model, d_head]
            return raw.view(self.config.n_heads, self.config.d_head, -1).transpose(1, 2)
        if name == "W_O":
            # [n_heads * d_head, d_model] → [n_heads, d_head, d_model]
            return raw.view(self.config.n_heads, self.config.d_head, -1)
        return raw
```

### Handling GQA (Grouped Query Attention)

LLaMA 2 70B and Mistral use GQA where `n_kv_heads < n_heads`. W_K and W_V have shape `[n_kv_heads * d_head, d_model]` instead of `[n_heads * d_head, d_model]`. The reshape must use `n_kv_heads`:

```python
if name in ("W_K", "W_V") and self.config.n_kv_heads is not None:
    n_heads = self.config.n_kv_heads
else:
    n_heads = self.config.n_heads
```

---

## 8. The Batch Context Implementation

### How `ti.batch()` works internally

```python
_batch_context: ContextVar[list | None] = ContextVar("batch", default=None)

@contextmanager
def batch():
    """Hint: batch forward passes inside this block."""
    plan = []
    token = _batch_context.set(plan)
    try:
        yield
    finally:
        _batch_context.reset(token)
        _execute_batch(plan)


# Inside Model.__call__:
def __call__(self, *args, get=None, map=None, grad=False, **kwargs):
    plan = _batch_context.get()
    if plan is not None:
        # We're inside ti.batch() — accumulate instead of executing
        future = Future()
        plan.append((args, kwargs, get, map, grad, future))
        return future  # returns a placeholder that resolves after batch exits

    # Normal eager execution
    ...
```

The `Future` object raises an error if accessed before the batch executes:

```python
class Future:
    def __init__(self):
        self._result = _UNSET

    def _set(self, result):
        self._result = result

    def __getattr__(self, name):
        if self._result is _UNSET:
            raise RuntimeError(
                "Accessed batch result before batch completed. "
                "Results are available after the `with ti.batch():` block exits."
            )
        return getattr(self._result, name)
```

### The batch executor

```python
def _execute_batch(plan: list):
    """Execute accumulated calls with batching optimization."""
    if not plan:
        return

    # Group by identical input
    groups = defaultdict(list)
    for entry in plan:
        args, kwargs, get, map, grad, future = entry
        input_key = _hash_inputs(args, kwargs)  # hash the input for grouping
        groups[input_key].append(entry)

    for input_key, entries in groups.items():
        if len(entries) == 1:
            # Single call, execute normally
            ...
        else:
            # Multiple calls with same input — try to batch
            _execute_batched_group(entries)
```

---

## 9. The Streaming Implementation

### Triple-buffered pipeline

```python
def stream(self, dataloader, *, get, batch_size=32, device="cpu"):
    """Yield activation batches with GPU/CPU overlap."""

    # Use two CUDA streams: compute and transfer
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    # Pinned memory buffer for CPU transfer
    pin_buffer = None

    prev_result = None

    for batch in dataloader:
        # Start forward pass on compute stream
        with torch.cuda.stream(compute_stream):
            output = self(batch, get=get)

        # While compute runs, transfer previous batch to CPU
        if prev_result is not None:
            with torch.cuda.stream(transfer_stream):
                cpu_result = {
                    site: act.to(device, non_blocking=True)
                    for site, act in prev_result.items()
                }
            transfer_stream.synchronize()
            yield Output(None, cpu_result)

        compute_stream.synchronize()
        prev_result = {site: output[site] for site in get}

    # Yield the last batch
    if prev_result is not None:
        cpu_result = {site: act.to(device) for site, act in prev_result.items()}
        yield Output(None, cpu_result)
```

---

## 10. Testing Strategy

### What to test (from tinygrad's philosophy)

tinygrad tests **observable behavior**, not internal implementation. Their `test_ops.py` tests that operations produce correct results, not that they use specific code paths.

Our tests should:

1. **Numerical equivalence**: `model(input, get=[site])[site]` should match the activation you'd get by manually hooking the module with PyTorch
2. **Passthrough correctness**: `model(input)` (no get/map) should produce EXACTLY the same output as the raw model
3. **Intervention correctness**: `model(input, map={site: ti.zero()})` should match running the model with a manual zeroing hook
4. **No state leaks**: after any call (including one that raises), the model should be in a clean state
5. **Multi-architecture**: all tests run on at least GPT-2 (small, fast) and one LLaMA variant

### What NOT to test

- Don't test that specific hooks were registered (implementation detail)
- Don't test that specific buffers were allocated (implementation detail)
- Don't test internal flag states (implementation detail)

### The regression test for correctness

```python
def test_get_matches_manual_hook(model_name, site_name, input_text):
    """The activation from ti.get must exactly match a manual PyTorch hook."""
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Manual hook
    manual_result = {}
    def hook(module, input, output):
        manual_result["act"] = output[0] if isinstance(output, tuple) else output
    handle = get_module(hf_model, site_path).register_forward_hook(hook)
    _ = hf_model(tokenize(input_text))
    handle.remove()

    # tinyinterp
    model = ti.Model(hf_model)
    site = model.site(site_name)
    output = model(tokenize(input_text), get=[site])

    assert torch.allclose(output[site], manual_result["act"], atol=1e-6)
```

---

## 11. What We Learn from tinygrad's sz.py

tinygrad has a file `sz.py` that counts lines and enforces the budget in CI. The key insight: **the line count is per-file, not just total.** This prevents any single file from becoming a monster.

Our budgets:

| File       | Max lines | Rationale                                             |
| ---------- | --------- | ----------------------------------------------------- |
| model.py   | 350       | The most complex file (wrapping, calling, generating) |
| adapter.py | 400       | Discovery logic for many architectures                |
| site.py    | 250       | Site + Layer + AttnAccessor + MlpAccessor             |
| hooks.py   | 250       | Hook state + hook creation + extraction               |
| maps.py    | 60        | Named map constructors are one-liners                 |
| batch.py   | 200       | Batch context + planning + execution                  |
| stream.py  | 120       | Triple-buffered streaming                             |
| output.py  | 50        | Output wrapper                                        |
| server/\*  | 500 total | Engine + protocol + client                            |
| **Total**  | **2,180** | Under 2,500 hard limit                                |

Enforced in CI:

```bash
python -c "
import pathlib
total = 0
for f in pathlib.Path('tinyinterp').rglob('*.py'):
    lines = len([l for l in f.read_text().splitlines() if l.strip() and not l.strip().startswith('#')])
    total += lines
    print(f'{f}: {lines}')
print(f'TOTAL: {total}')
assert total < 2500, f'OVER BUDGET: {total}'
"
```
