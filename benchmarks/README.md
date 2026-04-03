# Benchmarks

The benchmark directory is local-only now. It exists to validate `mirin.Model(...)`
collection throughput, chunking, and memory guardrails.

Useful entrypoints:

```bash
uv run python benchmarks/testbed_collect.py --model toy-llama --requests 256
uv run python benchmarks/runtime_collect_stress.py --model toy-llama
uv run python benchmarks/runtime_oom_guardrails.py --json
```

These scripts are repo-local testbeds, not installed product surface.
