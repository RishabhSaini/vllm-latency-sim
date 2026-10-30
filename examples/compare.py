"""Reproduce the real-vs-simulator validation from committed data.

Loads the calibrated cost model (gemma4_h200_roofline.json) and the real
`vllm bench serve` measurements (real_serve.json), runs the simulator at the
same request rates, and prints the comparison. With matplotlib installed it also
regenerates latency_compare.png.

    python examples/compare.py          # table
    python examples/compare.py --plot   # table + chart

No GPU or cluster required; everything needed is in this directory.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cost_model import RooflineCostModel  # noqa: E402
from kv import ModelConfig  # noqa: E402
from simulator import SchedulerConfig, Simulator  # noqa: E402
from workload import make_workload  # noqa: E402

real = json.load(open(os.path.join(HERE, "real_serve.json")))
cost = RooflineCostModel.load(os.path.join(HERE, "gemma4_h200_roofline.json"))

# gemma4 architecture; KV blocks set to vLLM's reported capacity (-> 9.17x conc).
cfg = ModelConfig(num_layers=60, num_kv_heads=16, head_size=256,
                  block_size=16, max_model_len=real["max_model_len"])
num_blocks = round(real["gpu_kv_cache_tokens"] / cfg.block_size)
sched = SchedulerConfig(
    max_num_seqs=256, max_num_batched_tokens=8192, enable_chunked_prefill=True,
    request_overhead_s=0.035,            # fixed per-request latency (tokenization etc.)
    admission_headroom_blocks_per_running=4.0,  # avoid preemption spiral at overload
)
wl = real["workload"]
CAPACITY = 10.0

rows = []
print(f"{real['model']} on {real['hardware']} | real (median) vs simulator\n")
hdr = (f"{'rate':>4} {'region':>10} | {'req/s R/S':>13} | {'TTFT R/S ms':>14} | "
       f"{'TPOT R/S ms':>14} | {'E2E R/S ms':>16}")
print(hdr); print("-" * len(hdr))
for rec in real["by_rate"]:
    rate = rec["offered_rate"]
    reqs = make_workload(n=wl["num_prompts"], rate=rate, burstiness=1.0,
                         prompt_len=(wl["input_len"], 20),
                         output_len=(wl["output_len"], 5), seed=1)
    m = Simulator(cfg, sched, cost, num_blocks).run(reqs)
    sim = dict(rps=m["n_requests"] / m["makespan_s"],
               ttft=m["ttft_s"]["p50"] * 1e3, tpot=m["tpot_s"]["p50"] * 1e3,
               e2e=m["e2e_s"]["p50"] * 1e3)
    region = "saturated" if rec["request_throughput"] < 0.9 * rate else "normal"
    rows.append((rate, rec, sim))
    print(f"{rate:>4} {region:>10} | {rec['request_throughput']:>5.1f}/{sim['rps']:>6.1f} | "
          f"{rec['median_ttft_ms']:>6.0f}/{sim['ttft']:>6.0f} | "
          f"{rec['median_tpot_ms']:>6.1f}/{sim['tpot']:>6.1f} | "
          f"{rec['median_e2el_ms']:>7.0f}/{sim['e2e']:>7.0f}")

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REAL, SIM, INK, GRID = "#0072B2", "#E69F00", "#1a1a1a", "#e6e6e6"
    rate = [r[0] for r in rows]
    panels = [("TTFT (ms)", "median_ttft_ms", "ttft", True),
              ("TPOT / ITL (ms)", "median_tpot_ms", "tpot", False),
              ("End-to-end latency (ms)", "median_e2el_ms", "e2e", True),
              ("Throughput (req/s)", "request_throughput", "rps", False)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, rk, sk, logy) in zip(axes.ravel(), panels):
        ax.set_axisbelow(True); ax.grid(True, color=GRID, lw=0.8)
        ax.axvspan(CAPACITY, max(rate), color="#f3f3f3", zorder=0)
        ax.plot(rate, [r[1][rk] for r in rows], "-o", color=REAL, lw=2, ms=6,
                label="real vLLM (H200)")
        ax.plot(rate, [r[2][sk] for r in rows], "--s", color=SIM, lw=2, ms=6,
                label="simulator (CPU)")
        if logy:
            ax.set_yscale("log"); title += ", log"
        ax.set_title(title, color=INK); ax.set_xlabel("offered request rate (req/s)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("gemma-4-31B-it on 1xH200: real vLLM vs simulator (shaded = saturated)",
                 color=INK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(HERE, "latency_compare.png"), dpi=150)
    print("\nwrote latency_compare.png")
