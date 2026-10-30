"""End-to-end demo: Llama-3-8B-ish on an H100-ish GPU, no GPU required.

Uses an analytical *seed* cost model (replace with a calibrated one from
calibrate.py for real estimates). Sweeps request rate and prints TTFT / TPOT /
throughput so you can see the latency-vs-load knee.
"""

from __future__ import annotations

from calibrate import seed_model
from cost_model import CostModel
from kv import ModelConfig, num_blocks_from_memory, max_concurrency
from simulator import SchedulerConfig, Simulator
from workload import make_workload

# Model and hardware description
cfg = ModelConfig(
    num_layers=32,
    num_kv_heads=8,      # GQA, per-GPU (TP=1)
    head_size=128,
    kv_dtype_bytes=2,
    block_size=16,
    max_model_len=8192,
)

NUM_PARAMS = 8.03e9
PEAK_FLOPS = 9.89e14       # H100 bf16 dense
HBM_BW = 3.35e12           # bytes/s

# Available KV memory after weights/activations (80GB * 0.9 util - weights - act).
avail_kv_bytes = int(80e9 * 0.9 - NUM_PARAMS * 2 - 4e9)
num_blocks = num_blocks_from_memory(cfg, avail_kv_bytes)

cost: CostModel = seed_model(cfg, NUM_PARAMS, PEAK_FLOPS, HBM_BW, mfu=0.5)

print(f"KV bytes/token : {cfg.kv_bytes_per_token():,}")
print(f"page bytes/block: {cfg.page_size_bytes():,}")
print(f"GPU KV blocks  : {num_blocks:,}")
print(f"max concurrency : {max_concurrency(cfg, num_blocks):.1f} reqs @ max_len")
print(f"seed cost model : {cost}\n")

sched = SchedulerConfig(
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    enable_chunked_prefill=True,
)

print("=== rate sweep (unique prompts, no prefix sharing) ===")
print(f"{'rate':>6} {'thrpt tok/s':>12} {'TTFT p50':>10} {'TTFT p99':>10} "
      f"{'TPOT p50':>10} {'E2E p50':>10}")
for rate in (2, 5, 10, 20, 40):
    reqs = make_workload(
        n=400, rate=rate, burstiness=1.0,
        prompt_len=(512, 256), output_len=(128, 64), seed=1,
    )
    sim = Simulator(cfg, sched, cost, num_blocks)
    m = sim.run(reqs)
    print(f"{rate:>6} {m['throughput_tok_s']:>12.0f} "
          f"{m['ttft_s']['p50']*1000:>9.1f}m {m['ttft_s']['p99']*1000:>9.1f}m "
          f"{m['tpot_s'].get('p50', 0)*1000:>9.1f}m {m['e2e_s']['p50']:>9.2f}s")

print("\n=== prefix caching: shared 1536-token system prompt, 8 variants ===")
print(f"{'groups':>7} {'hit rate':>9} {'thrpt tok/s':>12} {'TTFT p50':>10} {'TTFT p99':>10}")
for groups in (0, 8, 1):  # 0 = no sharing; fewer groups = more reuse
    reqs = make_workload(
        n=400, rate=15, burstiness=1.0,
        prompt_len=(2048, 128), output_len=(128, 64),
        num_prefix_groups=groups, shared_prefix_len=1536,
        block_size=cfg.block_size, seed=2,
    )
    sim = Simulator(cfg, sched, cost, num_blocks)
    m = sim.run(reqs)
    tag = "none" if groups == 0 else f"{groups}"
    print(f"{tag:>7} {m['prefix_hit_rate']*100:>8.1f}% {m['throughput_tok_s']:>12.0f} "
          f"{m['ttft_s']['p50']*1000:>9.1f}m {m['ttft_s']['p99']*1000:>9.1f}m")
