"""Benchmark the simulator's own speed (how fast it simulates, not GPU time)."""

from __future__ import annotations

import time

from cost_model import CostModel
from kv import ModelConfig, num_blocks_from_memory
from simulator import SchedulerConfig, Simulator
from workload import make_workload


def main(n: int = 4000) -> None:
    cfg = ModelConfig(num_layers=32, num_kv_heads=8, head_size=128,
                      block_size=16, max_model_len=8192)
    cost = CostModel(a=5e-4, b=3e-5, c=1e-4, d=4e-8)
    num_blocks = num_blocks_from_memory(cfg, int(50e9))
    reqs = make_workload(n=n, rate=30, prompt_len=(768, 256),
                         output_len=(160, 80), seed=0)

    t0 = time.perf_counter()
    m = Simulator(cfg, SchedulerConfig(), cost, num_blocks).run(reqs)
    wall = time.perf_counter() - t0

    print(f"requests simulated : {n}")
    print(f"engine steps       : {m['steps']:,}")
    print(f"wall time          : {wall * 1e3:.0f} ms")
    print(f"speed              : {n / wall:,.0f} req/s, {m['steps'] / wall:,.0f} steps/s")


if __name__ == "__main__":
    main()
