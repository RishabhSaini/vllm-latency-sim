"""Latency / throughput Pareto sweep.

Sweeps request rate across one or more scheduler configs, runs the simulator at
each point (no GPU), and plots the throughput-vs-latency trade-off plus the
non-dominated Pareto frontier. Writes `pareto.png` and `pareto.csv`.

    python pareto.py

The frontier answers "for a given TTFT SLO, what is the most throughput I can
get, and with which config?" -- the core capacity/cost question.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless; render straight to file
import matplotlib.pyplot as plt  # noqa: E402

from cost_model import CostModel  # noqa: E402
from kv import ModelConfig, num_blocks_from_memory  # noqa: E402
from simulator import SchedulerConfig, Simulator  # noqa: E402
from workload import make_workload  # noqa: E402

# Okabe-Ito: a published colorblind-safe categorical palette, used in fixed
# order (not cycled). Text stays in ink; color only carries series identity.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
INK = "#1a1a1a"
MUTED = "#8a8a8a"
FRONTIER = "#1a1a1a"


@dataclass
class Point:
    config: str
    rate: float
    throughput: float   # tokens/s  (maximize)
    ttft_p99_ms: float  # milliseconds (minimize)
    tpot_p50_ms: float


def sweep(cfg, cost, num_blocks, configs, rates, workload_kwargs) -> list[Point]:
    pts: list[Point] = []
    for name, sched in configs.items():
        for rate in rates:
            reqs = make_workload(rate=rate, block_size=cfg.block_size, **workload_kwargs)
            m = Simulator(cfg, sched, cost, num_blocks).run(reqs)
            pts.append(Point(
                config=name, rate=rate,
                throughput=m["throughput_tok_s"],
                ttft_p99_ms=m["ttft_s"]["p99"] * 1e3,
                tpot_p50_ms=m["tpot_s"].get("p50", 0.0) * 1e3,
            ))
    return pts


def pareto_front(pts: list[Point]) -> list[Point]:
    """Non-dominated points: maximize throughput, minimize TTFT p99."""
    front: list[Point] = []
    best_ttft = float("inf")
    for p in sorted(pts, key=lambda p: (-p.throughput, p.ttft_p99_ms)):
        if p.ttft_p99_ms < best_ttft:
            front.append(p)
            best_ttft = p.ttft_p99_ms
    return sorted(front, key=lambda p: p.throughput)


def plot(pts: list[Point], configs, out_png: str, slo_ms: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)

    for i, name in enumerate(configs):
        series = sorted((p for p in pts if p.config == name), key=lambda p: p.throughput)
        color = PALETTE[i % len(PALETTE)]
        ax.plot([p.throughput for p in series], [p.ttft_p99_ms for p in series],
                "-o", color=color, markersize=6, linewidth=2, label=name, zorder=3)

    front = pareto_front(pts)
    ax.plot([p.throughput for p in front], [p.ttft_p99_ms for p in front],
            "--", color=FRONTIER, linewidth=2, zorder=4, label="Pareto frontier")
    ax.scatter([p.throughput for p in front], [p.ttft_p99_ms for p in front],
               s=120, facecolors="none", edgecolors=FRONTIER, linewidths=2, zorder=5)

    if slo_ms is not None:
        ax.axhline(slo_ms, color=MUTED, linestyle=":", linewidth=1.5, zorder=2)
        ax.text(ax.get_xlim()[1], slo_ms, f" SLO {slo_ms:.0f} ms",
                va="bottom", ha="right", color=MUTED, fontsize=9)

    ax.set_yscale("log")  # TTFT spans orders of magnitude; keep the SLO knee legible
    ax.set_xlabel("Throughput (output tokens / s)", color=INK)
    ax.set_ylabel("TTFT p99 (ms, log scale)", color=INK)
    ax.set_title("Throughput vs. latency trade-off (simulated)", color=INK, fontsize=13)
    ax.tick_params(colors=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def write_csv(pts: list[Point], out_csv: str) -> None:
    front_ids = {id(p) for p in pareto_front(pts)}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "rate", "throughput_tok_s", "ttft_p99_ms",
                    "tpot_p50_ms", "on_frontier"])
        for p in sorted(pts, key=lambda p: (p.config, p.rate)):
            w.writerow([p.config, p.rate, round(p.throughput, 1),
                        round(p.ttft_p99_ms, 1), round(p.tpot_p50_ms, 2),
                        int(id(p) in front_ids)])
    print(f"wrote {out_csv}")


def main() -> None:
    cfg = ModelConfig(num_layers=32, num_kv_heads=8, head_size=128,
                      block_size=16, max_model_len=8192)
    cost = CostModel(a=5e-4, b=3e-5, c=1e-4, d=4e-8)
    num_blocks = num_blocks_from_memory(cfg, available_kv_bytes=int(50e9))

    # Two configs that genuinely trade off: a tight concurrency cap keeps TTFT
    # low but plateaus throughput early; a wide cap reaches higher throughput at
    # the cost of latency under load. The frontier picks between them per SLO.
    configs = {
        "max_seqs=24": SchedulerConfig(max_num_seqs=24, max_num_batched_tokens=8192),
        "max_seqs=256": SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=8192),
    }
    rates = [2, 5, 8, 12, 16, 20, 28, 36, 48]
    workload_kwargs = dict(n=400, burstiness=1.0, prompt_len=(768, 256),
                           output_len=(128, 64), seed=3)

    pts = sweep(cfg, cost, num_blocks, configs, rates, workload_kwargs)
    write_csv(pts, "pareto.csv")
    plot(pts, configs, "pareto.png", slo_ms=200)


if __name__ == "__main__":
    main()
