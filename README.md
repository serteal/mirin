# mirin

Small activation access and intervention wrapper for PyTorch models.

Navigate real module paths, not a separate site tree:

```python
import torch
import mirin as mi

model = mi.Model(hf_model, rename=mi.renames.llm)
out = model(**inputs, get=[model.model.layers[5].self_attn])
acts = out[model.model.layers[5].self_attn]

# Capture-only path: stop after the last requested site.
capture = model(
    **inputs,
    get=[model.model.layers[5].self_attn],
    stop_at_last_get=True,
)
acts = capture[model.model.layers[5].self_attn]
```

Collect activations with one entrypoint:

```python
site = model.model.layers[5]

# One-shot collection.
outs = model.collect(["hello", "world"], get=[site])
acts = outs[0][site]

# Stream a dataset batch by batch.
for step in model.collect(dataset, get=[site], out="gpu", max_tokens=4096):
    mask = step.batch["attention_mask"]
    acts = step[site]
    pooled = (acts * mask.to(acts.device, dtype=acts.dtype).unsqueeze(-1)).sum(dim=1)
    pooled = pooled / mask.sum(dim=1, keepdim=True).clamp(min=1).to(acts.device, dtype=acts.dtype)
    step.release()

# Let mirin handle the loop and run a local postprocess on each step.
def pool_mean(step):
    mask = step.batch["attention_mask"]
    acts = step[site]
    pooled = (acts * mask.to(acts.device, dtype=acts.dtype).unsqueeze(-1)).sum(dim=1)
    pooled = pooled / mask.sum(dim=1, keepdim=True).clamp(min=1).to(acts.device, dtype=acts.dtype)
    return pooled.cpu()

for pooled in model.collect(dataset, get=[site], out="gpu", process=pool_mean, max_tokens=4096):
    train_probe(pooled)

# Stream a huge export to disk.
manifest = model.collect(dataset, get=[site], out="acts/")
```

## Setup

```bash
uv sync --python 3.11
uv run pre-commit install
uv run pytest
```

To include optional model-loading support:

```bash
uv sync --python 3.11 --extra transformers
```

## Runtime Stats

```python
import mirin as mi

model = mi.Model(hf_model)
capacity = model.capacity
stats = model.stats()
```
