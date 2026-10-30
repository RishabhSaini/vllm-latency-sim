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
| `cost_model.py` | two step-latency models with `fit()`: linear and regime-aware roofline |
| `kv.py` | KV-block accounting using vLLM's real page/block/concurrency math |
| `workload.py` | gamma/Poisson arrivals + prompt/output length sampling |
| `simulator.py` | the discrete-event scheduler + step loop |
| `calibrate.py` | emit a shape grid, fit coefficients, or seed a roofline model |
| `demo.py` | runnable example (Llama-3-8B-ish on H100-ish), rate sweep |

## Quick start

```bash
pip install -r requirements.txt
python demo.py        # runs with an analytical seed cost model
```

## Usage example

```python
from kv import ModelConfig, num_blocks_from_memory
from cost_model import CostModel
from workload import make_workload
from simulator import Simulator, SchedulerConfig

# 1. Describe the model + hardware (Llama-3-8B-ish, TP=1).
cfg = ModelConfig(num_layers=32, num_kv_heads=8, head_size=128,
                  block_size=16, max_model_len=8192)

# 2. Cost model: load a calibrated one, or use coefficients directly.
#    cost = CostModel.load("cost_model.json")
cost = CostModel(a=5e-4, b=3e-5, c=1e-4, d=4e-8)

# 3. KV blocks from free memory (or read "# GPU blocks: N" from vLLM's log).
num_blocks = num_blocks_from_memory(cfg, available_kv_bytes=int(50e9))

# 4. Build a workload: 15 req/s, 2048-tok prompts, a shared 1536-tok prefix
#    across 8 variants (emergent prefix-cache hits).
reqs = make_workload(n=300, rate=15, burstiness=1.0,
                     prompt_len=(2048, 128), output_len=(128, 64),
                     num_prefix_groups=8, shared_prefix_len=1536,
                     block_size=cfg.block_size, seed=0)

# 5. Simulate and read the metrics.
m = Simulator(cfg, SchedulerConfig(max_num_seqs=256,
                                   max_num_batched_tokens=8192),
              cost, num_blocks).run(reqs)

print(f"throughput : {m['throughput_tok_s']:.0f} tok/s")
print(f"prefix hit : {m['prefix_hit_rate']*100:.1f}%")
print(f"TTFT p50/99: {m['ttft_s']['p50']*1e3:.0f} / {m['ttft_s']['p99']*1e3:.0f} ms")
print(f"TPOT p50   : {m['tpot_s']['p50']*1e3:.1f} ms")
```

## Pareto sweep

Sweep request rate across scheduler configs and plot the throughput-vs-latency
trade-off with the non-dominated frontier:

```bash
python pareto.py   # writes pareto.png and pareto.csv
```

![Throughput vs latency Pareto frontier](pareto.png)

The frontier answers "for a given TTFT SLO, what is the most throughput I can get
and with which config?" The example compares a tight vs. wide concurrency cap: the
tight cap saturates earlier; the wide cap reaches higher throughput before its own
latency knee, so the frontier switches configs past the SLO line.

## Cost models: linear vs. roofline

Two interchangeable step-latency models (same `step_time(C, B, S, A)` signature):

- **`CostModel`** (linear): `t = a + b·C + c·B + d·S`. Simple, interpretable,
  calibratable from few points. Smears the prefill/decode crossover into one line.
- **`RooflineCostModel`** (regime-aware): `t = max(compute, memory)` with an
  attention feature `A = Σ chunkᵢ·contextᵢ` (quadratic for prefill, linear for
  decode). Fit by alternating segmented regression. More accurate near the
  compute/memory crossover and for long context.

`calibrate.py fit` fits both and keeps the lower-MAPE one; the simulator uses
either transparently. On regime-split data the roofline model recovers the
generating function to ~4% MAPE where the linear fit is ~24%.

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
