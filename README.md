# vLLM latency simulator (no GPU)

A pure-Python discrete-event simulator of the vLLM **v1** engine's request
lifecycle and timing. It reproduces continuous batching, the token-budget
scheduler, paged-KV admission/preemption, and chunked prefill, and predicts
step latency from a **calibrated cost model** instead of running a GPU.

Use it for cost/latency estimation: TTFT, TPOT/ITL, end-to-end latency, and
throughput vs. request rate.

## Layout

| file | role |
|------|------|
| `cost_model.py` | `t_step = a + b·C + c·B + d·S [+ e·C²]` with least-squares `fit()` |
| `kv.py` | KV-block accounting using vLLM's real page/block/concurrency math |
| `workload.py` | gamma/Poisson arrivals + prompt/output length sampling |
| `simulator.py` | the discrete-event scheduler + step loop |
| `calibrate.py` | emit a shape grid, fit coefficients, or seed a roofline model |
| `demo.py` | runnable example (Llama-3-8B-ish on H100-ish), rate sweep |

## Quick start

```bash
python demo.py        # runs with an analytical seed cost model
```

## Calibrate against real vLLM (do this for real numbers)

The seed model is roofline-only (~30% off). For accurate estimates, fit the
coefficients to measured forward times on your target GPU/model/dtype/TP:

```bash
# 1. emit the batch-shape grid to sweep
python calibrate.py grid > shapes.csv

# 2. measure each shape's step time with vLLM's forward benchmark, on a GPU:
#    vllm bench forward --model <model> ...   (vllm/benchmarks/forward/)
#    Fill the t_seconds column -> measured.csv with header C,B,S,t_seconds

# 3. fit and save
python calibrate.py fit measured.csv cost_model.json
#   -> prints R^2 / MAPE; aim for MAPE < ~0.15
```

Then load it:

```python
from cost_model import CostModel
cost = CostModel.load("cost_model.json")
```

Re-fit per (model, GPU, dtype, tensor-parallel degree).

## Getting the KV block count

Either derive it (`kv.num_blocks_from_memory`) or — simpler — read the real
value from the vLLM startup log line `# GPU blocks: N` and pass it straight to
`Simulator(..., num_gpu_blocks=N)`.

## Validate

Compare simulated TTFT/TPOT percentiles against a real `vllm bench serve` run at
2–3 request rates; tune `a` (fixed overhead) and the preemption behavior until
they line up.

## Prefix caching

Hash-based, like vLLM: full prompt blocks are keyed by a content hash, shared
across requests, refcounted, and evicted LRU when unreferenced. The hit rate is
an *emergent output*, not an input. Drive it from the workload:

```python
make_workload(..., num_prefix_groups=8, shared_prefix_len=1536, block_size=16)
```

Each request joins a group and its leading `shared_prefix_len` tokens collide
with its group-mates (model shared system prompts / few-shot templates). The
first request of a group pays full prefill (cold); later ones reuse its blocks
(warm) until memory pressure evicts them. `metrics["prefix_hit_rate"]` reports
the realized cached-token fraction. Leave `num_prefix_groups=0` to fall back to
the flat per-request `prefix_hit_frac` approximation.

## Scope / simplifications

- Preemption is recompute-only (no swap-to-CPU).
- Single engine; no pipeline/data parallelism overlap (`step_with_batch_queue`).
- Speculative decoding not modelled (extend `needed_tokens`/cost `C`).
- Prompt blocks are promoted to the cache at prefill completion (vLLM registers
  each block as it fills; equivalent for prompts, which fill during prefill).

## Prior art

- **Vidur** (MLSys 2024) — calibrated vLLM-like inference simulator.
- **Sarathi** / chunked-prefill paper — per-step cost derivation.
