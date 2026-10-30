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

    def step_time(self, C: float, B: float, S: float, A: float = 0.0) -> float:
        """Predicted wall time (s) for one forward step of this shape.

        Args:
            C: Total query rows scheduled this step.
            B: Number of scheduled requests.
            S: Sum of context lengths over scheduled requests.
            A: Attention feature (ignored by this model; present so the linear
                and roofline models share one call signature).

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
            json.dump({"kind": "linear", **asdict(self)}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CostModel":
        with open(path) as f:
            d = json.load(f)
        d.pop("kind", None)
        return cls(**d)


@dataclass
class RooflineCostModel:
    """Regime-aware model: t = overhead + max(compute_time, memory_time).

    A transformer step is compute-bound (prefill) or memory-bound (decode); the
    true cost is the max of the two, not a single line. Each branch is affine:

        compute_time = cc0 + cc_C * C + cc_A * A
        memory_time  = mc0 + mc_B * B + mc_S * S

    where A = sum over scheduled requests of (chunk_i * context_i) is the
    attention feature. For a decode token (chunk=1) A contributes context_i
    (linear); for a prefill chunk of n tokens over context c it contributes n*c
    (the quadratic the linear model misses). Fit with `fit()` (alternating
    segmented regression, numpy only).
    """

    cc0: float = 0.0
    cc_C: float = 0.0
    cc_A: float = 0.0
    mc0: float = 0.0
    mc_B: float = 0.0
    mc_S: float = 0.0

    def step_time(self, C: float, B: float, S: float, A: float = 0.0) -> float:
        tc = self.cc0 + self.cc_C * C + self.cc_A * A
        tm = self.mc0 + self.mc_B * B + self.mc_S * S
        return max(tc, tm, 1e-6)

    @classmethod
    def fit(
        cls,
        C: np.ndarray,
        B: np.ndarray,
        S: np.ndarray,
        A: np.ndarray,
        t: np.ndarray,
        max_iter: int = 50,
    ) -> tuple["RooflineCostModel", dict]:
        """Fit the two affine branches by alternating assignment.

        Each point is assigned to whichever branch currently predicts the larger
        time; branches are re-fit on their assigned points until the assignment
        stops changing (a few iterations in practice).
        """
        C, B, S, A, t = (np.asarray(x, float).ravel() for x in (C, B, S, A, t))
        Xc = np.stack([np.ones_like(C), C, A], axis=1)
        Xm = np.stack([np.ones_like(C), B, S], axis=1)
        # Init: more tokens than requests => likely compute (prefill) bound.
        compute = C > B

        def _ls(X, y):
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            return coef

        bc = _ls(Xc, t)
        bm = _ls(Xm, t)
        for _ in range(max_iter):
            if compute.sum() >= 3:
                bc = _ls(Xc[compute], t[compute])
            if (~compute).sum() >= 3:
                bm = _ls(Xm[~compute], t[~compute])
            new_compute = (Xc @ bc) >= (Xm @ bm)
            if np.array_equal(new_compute, compute):
                break
            compute = new_compute

        model = cls(cc0=float(bc[0]), cc_C=float(bc[1]), cc_A=float(bc[2]),
                    mc0=float(bm[0]), mc_B=float(bm[1]), mc_S=float(bm[2]))
        pred = np.maximum(Xc @ bc, Xm @ bm)
        resid = t - pred
        ss_tot = float(np.sum((t - t.mean()) ** 2)) or 1.0
        ape = np.abs(resid) / np.clip(np.abs(t), 1e-9, None)
        diag = {
            "r2": 1.0 - float(np.sum(resid**2)) / ss_tot,
            "mape": float(np.mean(ape)),
            "p95_ape": float(np.percentile(ape, 95)),
            "n": int(len(t)),
            "compute_frac": float(compute.mean()),
        }
        return model, diag

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": "roofline", **asdict(self)}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RooflineCostModel":
        with open(path) as f:
            d = json.load(f)
        d.pop("kind", None)
        return cls(**d)


def load_cost_model(path: str):
    """Load either model type, dispatching on the saved ``kind`` tag."""
    with open(path) as f:
        kind = json.load(f).get("kind", "linear")
    return RooflineCostModel.load(path) if kind == "roofline" else CostModel.load(path)
