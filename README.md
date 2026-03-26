# tinyinterp

Small activation access and intervention wrapper for PyTorch models.

Navigate real module paths, not a separate site tree:

```python
import tinyinterp as ti

model = ti.Model(hf_model, rename=ti.renames.llm)
out = model(**inputs, get=[model.model.layers[5].self_attn])
acts = out[model.model.layers[5].self_attn]
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
