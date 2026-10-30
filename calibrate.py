"""Calibrate the cost model against real vLLM forward-pass measurements.

Workflow:
  1. Emit a grid of batch shapes to sweep:
         python calibrate.py grid > shapes.csv
     Feed those shapes to `vllm bench forward` on your target GPU/model/dtype/TP
     and record one measured step time per shape. See vllm/benchmarks/forward/.
  2. Collect results into a CSV with header:  C,B,S,t_seconds
  3. Fit:
         python calibrate.py fit measured.csv cost_model.json
     Prints R^2 / MAPE and writes the coefficients.

For a rough pre-calibration seed (analytical roofline), use `seed_model`.
Always re-fit per (model, GPU, dtype, tensor-parallel degree).
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from cost_model import CostModel
from kv import ModelConfig


def shape_grid() -> list[tuple[int, int, int]]:
    """A spread of (C, B, S) shapes covering decode, prefill, and chunked mixes.

    C = query rows, B = requests, S = summed context. Tweak to your workload;
    include points near your real max_num_batched_tokens and max context.
    """
    shapes: list[tuple[int, int, int]] = []
    # Pure decode: B reqs, 1 token each, varying context length.
    for B in (1, 8, 32, 64, 128, 256):
        for ctx in (128, 512, 2048, 8192):
            shapes.append((B, B, B * ctx))
    # Pure prefill: one request, C = prompt chunk.
    for C in (128, 512, 1024, 2048, 4096, 8192):
        shapes.append((C, 1, C))
    # Mixed (chunked prefill + decodes): one prefill chunk + Bd decodes.
    for C in (256, 1024, 4096):
        for Bd in (8, 32, 128):
            shapes.append((C + Bd, 1 + Bd, C + Bd * 1024))
    return shapes


def cmd_grid() -> None:
    w = csv.writer(sys.stdout)
    w.writerow(["C", "B", "S", "t_seconds"])
    for C, B, S in shape_grid():
        w.writerow([C, B, S, ""])  # fill t_seconds from vllm bench forward


def cmd_fit(in_csv: str, out_json: str, quadratic: bool = False) -> None:
    C, B, S, t = [], [], [], []
    with open(in_csv) as f:
        for row in csv.DictReader(f):
            if not row.get("t_seconds"):
                continue
            C.append(float(row["C"]))
            B.append(float(row["B"]))
            S.append(float(row["S"]))
            t.append(float(row["t_seconds"]))
    model, diag = CostModel.fit(
        np.array(C), np.array(B), np.array(S), np.array(t), quadratic=quadratic
    )
    model.save(out_json)
    print(f"fit: {model}")
    print(f"diagnostics: R^2={diag['r2']:.4f} MAPE={diag['mape']:.3f} "
          f"p95_APE={diag['p95_ape']:.3f} n={diag['n']}")
    print(f"saved -> {out_json}")


def seed_model(
    cfg: ModelConfig,
    num_params: float,
    peak_flops: float,
    hbm_bw_bytes_s: float,
    mfu: float = 0.5,
    weight_bytes: int = 2,
    fixed_overhead_s: float = 5e-4,
) -> CostModel:
    """Analytical roofline seed (use only until you have real measurements).

    b ~ compute per query token = 2*N / (peak_flops*mfu)
    d ~ KV read per context token = kv_bytes_per_token / hbm_bw
    a ~ fixed overhead + weight read amortized per step
    """
    b = 2.0 * num_params / (peak_flops * mfu)
    d = cfg.kv_bytes_per_token() / hbm_bw_bytes_s
    a = fixed_overhead_s + (num_params * weight_bytes) / hbm_bw_bytes_s
    return CostModel(a=a, b=b, c=0.0, d=d, e=0.0)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "grid":
        cmd_grid()
    elif len(sys.argv) >= 4 and sys.argv[1] == "fit":
        quad = "--quadratic" in sys.argv
        cmd_fit(sys.argv[2], sys.argv[3], quadratic=quad)
    else:
        print(__doc__)
        sys.exit(1)
