"""Calibrated per-step latency cost model for the vLLM v1 engine.

One ``EngineCore.step()`` == one ``scheduler.schedule()`` + exactly one forward
pass. In v1 there is no prefill/decode "phase": every request just has
``num_computed_tokens`` chasing its target, so a prefill chunk and a decode are
the same op with a different token count. We therefore model step latency as a
single linear function of the batch's *shape*, not of a phase:

    t_step = a + b*C + c*B + d*S   [+ e*C^2]

    C = total query rows this step  (= total_num_scheduled_tokens)
    B = number of scheduled requests
    S = sum of context lengths over scheduled requests (KV bytes read)
    C^2 = optional attention curvature term for long prefill chunks

Interpretation:
    a : fixed per-step overhead (launches, Python, sampling) -> tiny batches
    b : compute slope (GEMM FLOPs ~ query tokens)            -> prefill-heavy
    c : per-request decode overhead                          -> many small reqs
    d : KV-traffic slope (HBM read ~ resident context)       -> long context

Fit the coefficients to real measurements from ``vllm bench forward`` (see
calibrate.py). Pure analytical roofline is ~30% off; a calibrated linear fit is
typically within ~10-15% and is what you want for cost/latency estimation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CostModel:
    """Linear per-step latency model. Times are in seconds."""

    a: float = 0.0  # fixed overhead
    b: float = 0.0  # per query-token
    c: float = 0.0  # per scheduled request
    d: float = 0.0  # per unit of summed context (KV read)
    e: float = 0.0  # optional C^2 attention term (0 => disabled)

    def step_time(self, C: float, B: float, S: float) -> float:
        """Predicted wall time (s) for one forward step of this shape.

        Args:
            C: Total query rows scheduled this step.
            B: Number of scheduled requests.
            S: Sum of context lengths over scheduled requests.

        Returns:
            Predicted step latency in seconds (floored at a small positive
            value so the simulation clock always advances).
        """
        t = self.a + self.b * C + self.c * B + self.d * S + self.e * C * C
        return max(t, 1e-6)

    @classmethod
    def fit(
        cls,
        C: np.ndarray,
        B: np.ndarray,
        S: np.ndarray,
        t: np.ndarray,
        quadratic: bool = False,
    ) -> tuple["CostModel", dict]:
        """Least-squares fit of the coefficients to measured step times.

        Args:
            C, B, S: 1-D arrays of batch-shape features, one row per measured
                forward.
            t: 1-D array of measured step latencies (seconds).
            quadratic: If True, also fit the C^2 term.

        Returns:
            A (CostModel, diagnostics) pair. Diagnostics contains R^2 and the
            mean / p95 absolute percentage error.
        """
        C, B, S, t = map(lambda x: np.asarray(x, float).ravel(), (C, B, S, t))
        cols = [np.ones_like(C), C, B, S]
        if quadratic:
            cols.append(C * C)
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        e = float(coef[4]) if quadratic else 0.0
        model = cls(a=float(coef[0]), b=float(coef[1]),
                    c=float(coef[2]), d=float(coef[3]), e=e)

        pred = A @ coef
        resid = t - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((t - t.mean()) ** 2)) or 1.0
        ape = np.abs(resid) / np.clip(np.abs(t), 1e-9, None)
        diag = {
            "r2": 1.0 - ss_res / ss_tot,
            "mape": float(np.mean(ape)),
            "p95_ape": float(np.percentile(ape, 95)),
            "n": int(len(t)),
        }
        return model, diag

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CostModel":
        with open(path) as f:
            return cls(**json.load(f))
