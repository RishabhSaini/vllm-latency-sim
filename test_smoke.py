"""Smoke test: run the simulator end-to-end and check core invariants.

Runnable with plain `python test_smoke.py` (exits non-zero on failure) or under
pytest. No GPU, no heavyweight deps beyond numpy.
"""

from __future__ import annotations

import numpy as np

from cost_model import CostModel, RooflineCostModel
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


def test_roofline_beats_linear_on_regime_split_data():
    # Data from a two-regime roofline model: the roofline fit should recover it
    # far better than a single line, which smears the compute/memory crossover.
    true = RooflineCostModel(cc0=3e-4, cc_C=2.5e-5, cc_A=1e-9,
                             mc0=2e-3, mc_B=5e-5, mc_S=3e-8)
    rng = np.random.default_rng(0)
    C, B, S, A, t = [], [], [], [], []
    for _ in range(120):
        if rng.random() < 0.5:  # decode (memory-bound)
            b = int(rng.integers(1, 256)); ctx = int(rng.integers(128, 8192))
            c, s, a = b, b * ctx, b * ctx
        else:                   # prefill (compute-bound)
            c = int(rng.integers(128, 8192)); b = 1; s = c; a = c * c
        C.append(c); B.append(b); S.append(s); A.append(a)
        t.append(true.step_time(c, b, s, a) * (1 + rng.normal(0, 0.04)))
    arrs = [np.array(x) for x in (C, B, S, A, t)]
    _, lin_d = CostModel.fit(arrs[0], arrs[1], arrs[2], arrs[4])
    _, roof_d = RooflineCostModel.fit(*arrs)
    assert roof_d["mape"] < lin_d["mape"]
    assert roof_d["mape"] < 0.1  # recovers the generating model closely


def test_simulator_runs_with_roofline_model():
    cost = RooflineCostModel(cc0=3e-4, cc_C=2.5e-5, cc_A=1e-9,
                             mc0=2e-3, mc_B=5e-5, mc_S=3e-8)
    reqs = make_workload(n=120, rate=12, prompt_len=(512, 128),
                         output_len=(96, 32), seed=1)
    m = Simulator(CFG, SchedulerConfig(), cost, 20000).run(reqs)
    assert m["n_requests"] == 120
    assert m["throughput_tok_s"] > 0


if __name__ == "__main__":
    test_all_requests_drain_and_metrics_are_sane()
    test_block_accounting_invariant_under_pressure()
    test_prefix_cache_reduces_ttft()
    test_roofline_beats_linear_on_regime_split_data()
    test_simulator_runs_with_roofline_model()
    print("all smoke tests passed")
