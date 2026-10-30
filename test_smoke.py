"""Smoke test: run the simulator end-to-end and check core invariants.

Runnable with plain `python test_smoke.py` (exits non-zero on failure) or under
pytest. No GPU, no heavyweight deps beyond numpy.
"""

from __future__ import annotations

from cost_model import CostModel
from kv import ModelConfig
from simulator import SchedulerConfig, Simulator
from workload import make_workload

CFG = ModelConfig(num_layers=32, num_kv_heads=8, head_size=128,
                  block_size=16, max_model_len=8192)
COST = CostModel(a=5e-4, b=3e-5, c=1e-4, d=4e-8)


def _run(num_blocks: int, **wl):
    reqs = make_workload(**wl)
    sim = Simulator(CFG, SchedulerConfig(max_num_seqs=64,
                                         max_num_batched_tokens=4096),
                    COST, num_blocks)
    return sim, sim.run(reqs), len(reqs)


def test_all_requests_drain_and_metrics_are_sane():
    _, m, n = _run(20000, n=200, rate=10, prompt_len=(512, 128),
                   output_len=(96, 32), seed=1)
    assert m["n_requests"] == n
    assert m["throughput_tok_s"] > 0
    assert m["ttft_s"]["p50"] <= m["ttft_s"]["p99"]
    assert m["tpot_s"]["p50"] > 0


def test_block_accounting_invariant_under_pressure():
    # Tiny pool forces eviction + preemption; the pool must stay consistent.
    sim, m, n = _run(2000, n=300, rate=25, prompt_len=(1024, 256),
                     output_len=(96, 48), num_prefix_groups=4,
                     shared_prefix_len=768, block_size=16, seed=7)
    bm = sim.blocks
    assert m["n_requests"] == n                       # no deadlock
    assert 0 <= bm.free <= bm.num_blocks
    assert bm.free == bm.num_blocks - len(bm.cached) - sum(bm.owned.values())
    assert sum(bm.owned.values()) == 0               # nothing leaked
    assert all(b.refcount >= 0 for b in bm.cached.values())


def test_prefix_cache_reduces_ttft():
    # Shared prefix should yield a positive hit rate and lower TTFT than none.
    common = dict(n=400, rate=15, prompt_len=(2048, 128),
                  output_len=(128, 64), block_size=16, seed=2)
    _, cold, _ = _run(24000, num_prefix_groups=0, shared_prefix_len=0, **common)
    _, warm, _ = _run(24000, num_prefix_groups=8, shared_prefix_len=1536, **common)
    assert cold["prefix_hit_rate"] == 0.0
    assert warm["prefix_hit_rate"] > 0.5
    assert warm["ttft_s"]["p99"] < cold["ttft_s"]["p99"]


if __name__ == "__main__":
    test_all_requests_drain_and_metrics_are_sane()
    test_block_accounting_invariant_under_pressure()
    test_prefix_cache_reduces_ttft()
    print("all smoke tests passed")
