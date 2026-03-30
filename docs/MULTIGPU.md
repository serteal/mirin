# Multi-GPU Status

`docs/PLAN.md` gives the right frame for this topic:

- distribution should be a property of where tensors live, not a second API
- tiny changes beat large frameworks
- no “speedup” claim without benchmark evidence

This file documents the current state, not an aspirational one.

## What Works Today

If the wrapped HuggingFace model already spans devices, tinyinterp mostly rides along:

```python
server = ti.Server(
    "meta-llama/Llama-3.1-70B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

Today that means:

- hooks are registered on modules, not devices
- a hook fires wherever its module runs
- `maps.py` already coerces tensor values onto the activation device
- the wrapped model's own multi-device forward path still decides movement

In other words, tinyinterp does not implement tensor parallelism or pipeline parallelism itself. It wraps whatever layout the model already uses.

## What Is Not Done Yet

There is still no dedicated multi-GPU server layer for:

- consolidating captures from multiple GPUs to one destination
- computing admission or chunk budgets from the bottleneck device
- choosing peer-to-peer vs CPU bounce paths from benchmarked evidence
- remote transport that understands multi-node or multi-host execution

So the right description today is:

```text
multi-GPU model execution may work if the wrapped model works,
but tinyinterp's server-side budgeting and collection policy are still mostly single-device.
```

## Design Direction

The minimal direction is still the one from `docs/PLAN.md`:

1. keep the user-facing API unchanged
2. detect when captured activations span devices
3. consolidate only when needed
4. base server budget on the tightest device, not the sum of all devices
5. benchmark before claiming a win

## Practical Guidance

For now:

- treat `device_map="auto"` support as wrapped-model compatibility, not a finished tinyinterp feature
- verify activation device placement in your own workload before building on it
- benchmark end to end before documenting performance claims

When multi-GPU becomes a first-class server feature, this file should grow only if the code grows. If the diff stays small, the doc should stay small too.
