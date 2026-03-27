# tinyinterp

Small activation access and intervention wrapper for PyTorch models.

Navigate real module paths, not a separate site tree:

```python
import tinyinterp as ti

model = ti.Model(hf_model, rename=ti.renames.llm)
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

## Server Mode

```python
import tinyinterp as ti

server = ti.Server(hf_model)
plan = server.compile(
    get=["model.layers.5.self_attn"],
    output={"logits": False, "activations": True},
)

collector = server.open_collector(plan=plan, stop_at_last_get=True)
batch_out = collector.collect_batch(inputs)
acts = batch_out.activations["model.layers.5.self_attn"]
many_out = collector.collect_many(
    [
        {"input_ids": inputs["input_ids"][0]},
        {"input_ids": inputs["input_ids"][0]},
    ]
)

session = server.open_session(plan=plan, cache="dynamic")
prefill = server.prefill(session, **inputs)
step = server.decode([session], max_new_tokens=1)[0]

batched = server.generate_many(
    [
        [{"role": "user", "content": "Summarize attention heads."}],
        [{"role": "user", "content": "List safety caveats."}],
    ],
    max_new_tokens=32,
)

stats = server.stats()
```
