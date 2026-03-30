# tinyinterp Server Optimization Report

_Review of the current in-process inference server, focused on local activation collection, interpretability workflows, batching QoL, memory handling, and GPU utilization for HuggingFace LLMs._

---

## 1. Executive Summary

The current server already has several real optimizations:

- compiled interpretability plans
- narrow result policies
- a collector fast path using `stop_at_last_get=True`
- server-owned session state and HuggingFace cache objects
- decode grouping by compatibility
- basic token-budget chunking and admission control

Those are meaningful wins. The project benchmark artifacts show that the collector path already beats a plain manual HuggingFace hook loop by a large margin.

The biggest remaining bottlenecks are not missing kernels. They are:

1. the server is still effectively single-threaded
2. decode batching is manual and call-local, not a real continuous scheduler
3. dynamic-cache microbatching copies KV tensors every decode step
4. static-cache decode cannot microbatch at all
5. batching quality collapses once session lengths diverge
6. collector CPU copies are synchronous and unpinned

If this server is mainly for local interpretability workflows, the highest-value work is:

- better collector pipelines
- better cache batching
- better prefill/decode scheduling
- less GPU memory churn

It is **not** transport or networking work. The current server is in-process, and that is good.

---

## 2. What Is Already Optimized

The current implementation is not a naive wrapper. It already includes several useful optimizations.

### 2.1 Compiled plans

Location:

- `tinyinterp/server/plans.py:66-100`

What it does:

- resolves paths once
- validates supported map ops once
- computes a stable plan fingerprint
- stores pre-resolved proxies and map dicts

Why it matters:

- removes repeated path traversal from hot execution paths
- gives the scheduler a stable compatibility key

Assessment:

- good design
- keep it

### 2.2 Narrow output policies

Location:

- `tinyinterp/server/plans.py:28-46`
- `tinyinterp/server/plans.py:145-187`
- `tinyinterp/server/inference.py:819-845`

What it does:

- allows requests to avoid returning tokens, logits, or activations unless needed
- allows optional CPU offload for activations/logits
- uses last-token logits for decode-oriented results

Why it matters:

- output size is a major local bottleneck in activation workflows
- returning only the last-token logits is much cheaper than returning full `[batch, seq, vocab]`

Assessment:

- good optimization
- should become stricter, not looser

### 2.3 Collector capture-only fast path

Location:

- `tinyinterp/server/inference.py:143-158`
- `tinyinterp/server/inference.py:166-199`
- `tinyinterp/model.py:269-337`

What it does:

- automatically enables `stop_at_last_get=True` for capture-only collection when safe
- disables it when `map=` is present

Why it matters:

- this is probably the single most important optimization for offline activation extraction
- it turns "run the whole model" into "stop as soon as the last requested site fires"

Assessment:

- excellent
- this is the right direction for local interp workloads

### 2.4 Collector token-budget chunking

Location:

- `tinyinterp/server/collector.py:33-65`
- `tinyinterp/server/inference.py:172-183`

What it does:

- splits oversized batches by an approximate token budget

Why it matters:

- avoids obvious OOM cases
- gives users a basic dataset-handling QoL improvement

Assessment:

- useful first step
- much too coarse to count as "adaptive batching"

### 2.5 Session-owned caches

Location:

- `tinyinterp/server/inference.py:205-315`
- `tinyinterp/server/sessions.py:55-86`

What it does:

- opens persistent sessions
- supports `cache="dynamic"`, `cache="static"`, and `cache="none"`
- uses HuggingFace cache classes when available

Why it matters:

- this is the correct serving abstraction
- it avoids sending `past_key_values` around as user inputs

Assessment:

- correct architecture
- implementation still leaves a lot of performance on the table

### 2.6 Decode grouping and microbatch chunking

Location:

- `tinyinterp/server/inference.py:584-608`
- `tinyinterp/server/scheduler.py:102-118`
- `tinyinterp/server/sessions.py:44-52`

What it does:

- groups pending sessions by compatibility key
- optionally chunks large groups by token budget

Why it matters:

- gives the server a path to multi-session decode

Assessment:

- important foundation
- current grouping rules are too restrictive

### 2.7 Admission control and queue metrics

Location:

- `tinyinterp/server/scheduler.py:121-225`
- `tinyinterp/server/inference.py:927-963`

What it does:

- estimates KV-cache bytes and activation bytes
- can reject requests before execution
- tracks queue depth, wait time, service time, total tokens, and session counts

Why it matters:

- gives the server some memory awareness
- exposes useful operational stats

Assessment:

- good guardrail
- estimates are still too rough for aggressive packing

### 2.8 HuggingFace-native prefill/decode plumbing

Location:

- `tinyinterp/server/inference.py:550-582`
- `tinyinterp/server/inference.py:758-808`

What it does:

- uses `prepare_inputs_for_generation`
- handles `cache_position` for static caches
- keeps prefill/decode logic explicit instead of delegating to `generate()`

Why it matters:

- this is the right abstraction boundary
- it keeps tinyinterp in control of scheduling

Assessment:

- correct approach
- needs more batching and less copying around it

---

## 3. Benchmark Evidence From The Repo

Source artifacts:

- `benchmarks/results/server_runtime-after-steps.json`
- `benchmarks/results/server_runtime-smoke-after-step3.json`

The exact numbers vary by model and benchmark revision, but the pattern is consistent.

### 3.1 Collector path

In `server_runtime-after-steps.json`, `server_collector` is faster than `hf_hook_loop` by:

- about 3.7x on Llama 3.1 8B
- about 1.7x on Gemma 3 4B
- about 17.5x on Qwen 3.5 4B

That is real. The collector fast path is already paying off.

Relative to the local non-server `tinyinterp_capture_loop`, the current server collector is much closer to parity. Depending on model and benchmark artifact, it is sometimes faster and sometimes slower. That means the server is already a QoL and control-plane win, but not yet a decisive execution-engine win for pure local capture.

### 3.2 Single-session generation

The current server is around parity with plain HuggingFace `generate()` for single-session decode:

- sometimes slightly faster
- sometimes slightly slower

This suggests the current hot path is not fundamentally broken, but also not yet adding major serving-side value for the single-user case.

### 3.3 Multi-session generation

The current `server_generate_multi_session` beats sequential HuggingFace generation on Llama and Gemma, and is about parity on Qwen.

But it still trails HuggingFace's already-batched multi-example generate path by a lot:

- about 1.7x slower on Llama in `server_runtime-after-steps.json`
- about 2.1x slower on Gemma
- about 3.8x slower on Qwen

This is the clearest sign that the remaining bottleneck is scheduler/cache implementation, not hook overhead.

### 3.4 Static cache

Static-cache multi-session decode is currently much worse than dynamic-cache multi-session decode where supported:

- about 2.9x slower on Llama in `server_runtime-after-steps.json`
- about 1.6x slower on Gemma

That lines up with the code: static cache exists, but static-cache groups do not actually microbatch.

---

## 4. Salient Bottlenecks

This section ranks the bottlenecks by likely impact on local interpretability workflows.

### 4.1 High: Global execution lock serializes all work

Location:

- `tinyinterp/server/inference.py:97`
- `tinyinterp/server/inference.py:927-963`

What is happening:

- every `call`, `collect_batch`, `prefill`, and `decode` runs under one global lock
- queue metrics are tracked, but there is still only one active execution section

Why this matters:

- a collection job can block decode
- decode cannot accumulate work from independently arriving callers while another op is running
- there is no overlap between different server activities

Why it is especially salient for local use:

- local users often run one process with multiple tasks: dataset extraction, experiments, and interactive debugging
- the current design turns those into strict serialized phases

Optimization direction:

- keep a single GPU executor if desired, but move to a real request scheduler
- queue work first, then build batches from the queue
- do not let the caller's synchronous method call define the batch boundary

### 4.2 High: Decode batching is call-local, not true continuous batching

Location:

- `tinyinterp/server/inference.py:317-398`
- `tinyinterp/server/inference.py:584-608`

What is happening:

- decode batches only the sessions passed into one explicit `server.decode([...])` call
- the server does not own an asynchronous decode queue

Why this matters:

- batching quality depends on the caller already knowing how to batch
- independent callers will not coalesce naturally
- this is not "continuous batching" in the vLLM/SGLang sense

Why it is especially salient:

- local interpretability workflows often involve many small repeated decode requests
- the current API gives QoL improvements only to already-batched callers

Optimization direction:

- add a real decode queue
- let the server accumulate compatible pending requests for a short scheduling window
- keep the public API synchronous if desired, but make batching server-owned

### 4.3 High: Dynamic-cache microbatching copies KV tensors every decode step

Location:

- `tinyinterp/server/sessions.py:89-123`
- `tinyinterp/server/sessions.py:126-157`
- `tinyinterp/server/inference.py:690-703`

What is happening:

- `merge_caches()` concatenates per-session keys and values layer by layer
- `split_cache()` slices and clones them back into per-session caches after the forward

Why this matters:

- this is massive memory traffic
- it creates allocator churn every decode step
- it scales with layer count, KV size, and active sessions

Why it is especially salient:

- for LLM decode, memory traffic is often the real bottleneck
- this design throws away the main advantage of persistent server-side caches

This is likely the single most important decode bottleneck in the current implementation.

Optimization direction:

- stop materializing merged and split cache objects every token
- move toward one of:
  - a persistent batched cache layout
  - a paged/block cache manager
  - a static bucketed cache layout that stays batched across steps

### 4.4 High: Static cache cannot microbatch

Location:

- `tinyinterp/server/inference.py:601-605`
- `tinyinterp/server/inference.py:1165-1188`

What is happening:

- `_can_microbatch_hf_group()` only returns true for `DynamicCache`
- static-cache sessions fall back to per-session eager-like advancement

Why this matters:

- static cache is supposed to be a performance tool
- here it becomes a batching anti-optimization in multi-session decode

Why it is especially salient:

- benchmark artifacts already show static multi-session decode is much slower than dynamic

Optimization direction:

- either implement real static-cache microbatching or stop advertising static cache as a multi-session optimization

### 4.5 High: Sessions only batch when lengths exactly match

Location:

- `tinyinterp/server/sessions.py:44-52`

What is happening:

- dynamic-cache sessions use `(plan.fingerprint, cache_mode, current_length)` as the compatibility key
- once lengths diverge, they stop batching together

Why this matters:

- prompt lengths differ
- EOS appears at different times
- real workloads drift out of lockstep quickly

Why it is especially salient:

- this causes batch quality to decay over time
- it makes the best case look much better than the steady-state case

Optimization direction:

- batch by bucketed decode length rather than exact current length
- or use a cache layout that supports mixed-length batched decode properly

### 4.6 High: Prefill is not batched across sessions

Location:

- `tinyinterp/server/inference.py:239-315`

What is happening:

- `prefill()` operates on one session at a time
- there is chunking within a session, but no cross-session prefill batching

Why this matters:

- large prompt ingestion workloads are common in interpretability jobs
- interactive multi-session serving also spends real time in prefill

Why it is especially salient:

- users dealing with many prompts do not get the same batching help they get for decode

Optimization direction:

- add a prefill queue
- batch compatible prefills by plan, backend, and length bucket
- allow chunked prefill batches so decode can still cut in

### 4.7 Medium-High: Session token history and masks are kept and grown on GPU

Location:

- `tinyinterp/server/inference.py:485-486`
- `tinyinterp/server/inference.py:525-526`
- `tinyinterp/server/inference.py:614-629`
- `tinyinterp/server/inference.py:713-715`
- `tinyinterp/server/inference.py:751-754`
- `tinyinterp/server/inference.py:377-384`

What is happening:

- input ids and attention masks are cloned and stored per session
- they are repeatedly extended with `torch.cat`
- generated token outputs are also accumulated via repeated `torch.cat`

Why this matters:

- repeated `torch.cat` creates O(n^2)-style small reallocation churn over long decode
- GPU memory is consumed by token history that does not need to stay on GPU once cached

Why it is especially salient:

- local users often keep many sessions open
- token history is cheap on CPU and expensive on GPU

Optimization direction:

- keep session token history on CPU or in Python lists
- store only `current_length`, `pending_input_ids`, and cache state on GPU
- accumulate output tokens in lists and stack once at the end

### 4.8 Medium-High: Collector CPU copies are synchronous and unpinned

Location:

- `tinyinterp/server/inference.py:828-839`
- `tinyinterp/server/inference.py:1032-1046`

What is happening:

- activation and logits offload uses plain `.cpu()`
- no pinned buffers
- no non-blocking transfer
- no overlap with the next forward

Why this matters:

- for activation collection, D2H can dominate total time
- especially true for large site sets or wide hidden states

Optimization direction:

- pinned destination buffers
- non-blocking copies
- optional double-buffered writer path for collector mode

### 4.9 Medium: Collector batching is too coarse

Location:

- `tinyinterp/server/collector.py:33-65`
- `tinyinterp/server/inference.py:172-183`

What is happening:

- batches are only split by `token_budget // seq_len`
- no length bucketing across dataset
- no packing by activation size
- no consideration of number of requested sites

Why this matters:

- `get=[layer_8]` and `get=[many_layers]` are treated almost the same
- sequence-length variance wastes batch capacity

Optimization direction:

- add dataset-aware bucketing
- split by estimated activation bytes, not only prompt tokens
- optionally pack multiple small source batches into one collector batch

### 4.10 Medium: Admission estimates are rough

Location:

- `tinyinterp/server/scheduler.py:121-225`

What is happening:

- activation size is estimated as `batch * seq * hidden * num_sites`
- this ignores actual module output shapes
- this ignores `stop_at_last_get=True` depth
- this ignores extra clone cost for `get+map`

Why this matters:

- estimates can underutilize memory
- or fail to protect against real activation-heavy requests

Optimization direction:

- cache per-site shape profiles after first execution
- distinguish residual/attention/logit-like sites
- incorporate output policy and stop depth into the estimate

### 4.11 Medium: Repeated Python overhead in hot paths

Location:

- `tinyinterp/server/inference.py:986-995`
- `tinyinterp/server/inference.py:1191-1203`
- `tinyinterp/model.py:310-312`
- `tinyinterp/hooks.py:136-162`

What is happening:

- `inspect.signature()` runs on every filtered call
- nested structures are recursively walked to move tensors
- model execution uses `torch.no_grad()` instead of `torch.inference_mode()`
- `get+map` captures clone activations before applying the map

Why this matters:

- these costs are not the main bottleneck on big models
- but they add up in local workflows with many small batches and many short decode steps

Optimization direction:

- cache the allowed-kwargs set per model class
- short-circuit `_move_tensors_to()` when values already live on the target device
- measure `torch.inference_mode()` for non-grad server execution
- avoid unnecessary activation clones where semantics permit it

### 4.12 Low-Medium: `call()` does not use the server's better batching paths

Location:

- `tinyinterp/server/inference.py:111-132`

What is happening:

- `call()` is just an immediate wrapped forward
- it does not use collector splitting
- it does not use admission estimates
- it does not queue with other compatible work

Why this matters:

- advanced users may reach for `call()` for big local jobs and miss the optimized collection path

Optimization direction:

- either document `call()` as the simple path
- or give it optional admission/batching behavior

---

## 5. Recommended Optimization Order

This is the order that makes the most sense for a local in-process interpretability server.

### Priority 1: Replace call-local decode batching with a real decode scheduler

Why first:

- highest impact on user-visible serving behavior
- required before "continuous batching" is a true claim

Concrete work:

- add a decode queue
- collect pending compatible sessions for a short scheduling window
- let decode batching happen across independent callers
- preserve synchronous API by blocking on futures

### Priority 2: Eliminate per-step cache merge/split copies

Why second:

- likely the biggest per-token bottleneck
- directly responsible for poor multi-session scaling

Concrete work:

- persistent batched cache layout for dynamic sessions
- or paged/block cache manager
- or real static-cache batched decode

Do not keep `torch.cat` + clone on every layer and every step.

### Priority 3: Batch prefill across sessions

Why third:

- prompt ingestion is a real workload
- required for "large dataset" and multi-prompt QoL

Concrete work:

- prefill queue
- length bucketing
- chunked prefill batches
- prefill/decode interleaving policy

### Priority 4: Move session history off GPU

Why fourth:

- easy win
- improves memory and allocator behavior

Concrete work:

- CPU-side token history
- GPU-side `current_length` plus cache only
- list-based token accumulation rather than repeated `torch.cat`

### Priority 5: Improve collector output pipeline

Why fifth:

- directly impacts the main local-use case

Concrete work:

- pinned CPU output buffers
- non-blocking D2H
- optional mmap writer path
- optional server-side reducers

### Priority 6: Make collector batching truly adaptive

Why sixth:

- better GPU utilization on mixed datasets
- better memory stability

Concrete work:

- length bucketing
- activation-byte-aware splitting
- optional packer for small batches

### Priority 7: Clean up Python overhead

Why seventh:

- worthwhile, but not the main problem

Concrete work:

- cache signature filtering
- use `torch.inference_mode()`
- reduce recursive structure walking

---

## 6. What I Would Not Prioritize Yet

### 6.1 Network transport work

This server is in-process today. For the stated local-use goal, transport is not the limiting factor.

### 6.2 Generalizing the hot path beyond HuggingFace CausalLMs

The performance path should stay specialized until the LLM use case is genuinely strong.

### 6.3 Fancy output-object abstractions

Result narrowing is already the right idea. More object model complexity will not fix throughput.

### 6.4 Custom kernels before fixing scheduler and cache layout

The current benchmark story says memory movement and batching are the first-order problems.

---

## 7. Bottom Line

The current server already has the right basic shape for local interpretability work:

- compiled plans
- collector fast path
- server-owned sessions
- explicit prefill/decode split
- basic batching hooks

The collector path is already valuable.

The decode path is where the biggest issues remain. Right now it is limited less by HuggingFace itself than by:

- manual batching boundaries
- serialized scheduling
- per-step cache copying
- weak mixed-length batching

If the goal is "optimized for activation collection and similar interpretability workflows, with good batching and memory handling for local HuggingFace LLM use," the next big wins are all server-internal:

1. real decode/prefill queues
2. persistent batched cache layout
3. stronger collector pipeline
4. less GPU history churn

That is where the effort should go next.
