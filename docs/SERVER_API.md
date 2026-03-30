# tinyinterp Server API

`docs/PLAN.md` is still the bar:

- one obvious front door
- keep the public API small
- push serving complexity into the runtime, not the core wrapper
- benchmark every speed claim

This is the API that exists after the current refactor.

## Public Split

There are two public objects with different jobs:

```python
import tinyinterp as ti

local = ti.Model(hf_model)
socket = ti.Model("unix:///tmp/tinyinterp.sock")
```

- `ti.Model(...)` is the user-facing model API.
- `ti.Server(...)` is the runtime/service API.
- `Server` does not expose `server.model`.

## `ti.Model(...)`

`ti.Model` now covers two cases:

```python
ti.Model(hf_model)                  # local
ti.Model("unix:///tmp/tinyinterp.sock")  # remote Unix socket client
```

`ti.Model(server)` has been removed.

Shared high-level methods:

- `model(...)`
- `model.generate(...)`
- `model.collect(...)`
- `model.capabilities`

`model(...)` returns the shared `tinyinterp.Output` contract on both backends. Local tensors are materialized immediately; remote tensors may resolve lazily when you access `out[site]` or `out.logits`.

`model.generate(...)` returns generated token tensors when no activations are requested. With
`get=...`, it returns `tinyinterp.GenerateOutput`, which exposes:

- `out.sequences`
- `out.generated_ids`
- `out[site]`
- `out.prompt_length`
- `out.generated_length`

Generation capture modes:

- `capture="all"`: prompt activations plus generated-token activations
- `capture="generated"`: generated-token activations only

Shared request forms:

- raw tensor/model kwargs
- strings
- chat/message requests
- request lists

`model.capabilities` reports what the current backend actually supports. Today that mainly matters for `grad`, lazy remote values, and the remote protocol version.

What stays local-only today:

- arbitrary Python object outputs from unusual custom models

What the remote client supports today:

- proxy navigation
- `get=` / built-in `map=`
- `grad=True` for raw tensor `model(...)` calls
- `stop_at_last_get=True` for capture-only calls
- `collect(...)`
- `generate(...)`

## `ti.Server(...)`

`Server` owns runtime concerns:

```python
server = ti.Server(
    hf_model,
    tokenizer=tokenizer,
    gpu_fraction=0.9,
    cpu_fraction=0.8,
)
```

Server-only APIs:

- `compile(...)`
- `call(...)`
- `call_many(...)`
- `open_collector(...)`
- `collect_batch(...)`
- `collect_many(...)`
- `open_session(...)`
- `prefill(...)`
- `prefill_many(...)`
- `decode(...)`
- `generate(...)`
- `generate_many(...)`
- `stats()`
- `budget`
- `serve(sock_path)`

That is the place for:

- scheduler policy
- batching
- KV/cache ownership
- admission control
- memory budgets
- monitoring/utilization stats

Runtime execution is split intentionally:

- stateless `call` / `collect` / `generate` hot paths no longer share one global server lock
- stateful `prefill` / `decode` session work is still guarded together because it mutates runtime-owned cache and family state

## `model.collect(...)`

`collect(...)` is the shared high-level collection helper:

```python
site = model.layers[0]
rows = model.collect(
    ["hello", "world"],
    get=[site],
)
```

Current behavior:

- requires at least one `get=` site
- returns one output per request
- defaults to `stop_at_last_get=True`
- local and remote modes both use the same lowered runtime collector path
- the deployed server adds transport, multi-client scheduling, and stats around that same core

This keeps the user-facing API fixed while deployment-only concerns stay in `Server`.

## Transport

The remote transport is still intentionally small:

- Unix sockets only
- `unix:///path.sock` is the explicit endpoint form

Remote `grad=True` is handle-based:

- the server owns the live autograd tape
- `out[site]` behaves like a tensor proxy and supports `.backward(...)`
- `out[site].grad` and `out.input_grads` fetch gradients back from the server
- prompt/text request forms are still intentionally unsupported for remote grad

The transport is still local-machine oriented. It is not a hardened multi-user RPC layer.

## What This Doc Does Not Claim

This doc does not claim:

- generic remote execution for arbitrary Python model outputs
- multi-host serving
- benchmark wins that are not in `benchmarks/`
