"""Workload / arrival generation.

Mirrors ``vllm bench serve``: gamma inter-arrivals with a ``burstiness``
parameter (burstiness=1 => exponential => Poisson arrivals), plus prompt and
output length distributions. You can also build requests directly from a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Request:
    req_id: int
    arrival: float
    prompt_len: int
    output_len: int
    prefix_hit_frac: float = 0.0  # flat-model fallback (ignored if block_hashes set)
    block_hashes: list[int] = field(default_factory=list)  # one per full prompt block

    # --- mutable simulation state ---
    computed: int = 0             # tokens with KV computed (num_computed_tokens)
    produced: int = 0            # output tokens sampled so far
    status: str = "waiting"      # waiting | running | finished
    ttft: float | None = None     # first output token time - arrival
    token_times: list[float] = field(default_factory=list)  # decode timestamps
    finish_time: float | None = None

    @property
    def target_prefill(self) -> int:
        # With hash-based prefix caching the discount comes from `computed` being
        # seeded at admission, so the whole prompt is the prefill target here.
        if self.block_hashes:
            return self.prompt_len
        # Flat fallback: approximate cached tokens as a fixed fraction.
        return max(1, round(self.prompt_len * (1.0 - self.prefix_hit_frac)))

    def needed_tokens(self) -> int:
        """Tokens whose KV is not yet computed (num_tokens - num_computed)."""
        return (self.target_prefill + self.produced) - self.computed

    def context_len(self) -> int:
        """Positions attended to (drives KV read in the cost model)."""
        return self.computed + self.prompt_len - self.target_prefill


def poisson_arrivals(n: int, rate: float, burstiness: float, rng: np.random.Generator) -> np.ndarray:
    """Cumulative arrival times for n requests at `rate` req/s.

    burstiness=1 gives exponential gaps (Poisson). <1 is burstier, >1 smoother.
    rate<=0 means all requests arrive at t=0 (closed-loop / max load).
    """
    if rate <= 0:
        return np.zeros(n)
    theta = 1.0 / (rate * burstiness)
    gaps = rng.gamma(shape=burstiness, scale=theta, size=n)
    return np.cumsum(gaps)


def _block_hashes(req_id: int, prompt_len: int, group: int, shared_len: int, bs: int) -> list[int]:
    """Per-full-block content hashes.

    Blocks fully inside the group's shared prefix are keyed by (group, index) so
    requests in the same group collide and share them; blocks past the prefix are
    keyed by (req_id, index) so they are unique. This reproduces vLLM's
    contiguous-prefix matching without real tokenization.
    """
    nblocks = prompt_len // bs  # only full blocks are cacheable
    shared_full = shared_len // bs
    return [
        hash(("g", group, j)) if j < shared_full else hash(("r", req_id, j))
        for j in range(nblocks)
    ]


def make_workload(
    n: int,
    rate: float = 1.0,
    burstiness: float = 1.0,
    prompt_len: tuple[int, int] = (512, 128),   # (mean, std) lognormal-ish
    output_len: tuple[int, int] = (128, 64),
    prefix_hit_frac: float = 0.0,
    num_prefix_groups: int = 0,     # >0 enables hash-based shared prefixes
    shared_prefix_len: int = 0,     # tokens of shared prefix per group
    block_size: int = 16,
    seed: int = 0,
) -> list[Request]:
    """Synthesize a list of Requests sorted by arrival time.

    Set ``num_prefix_groups`` and ``shared_prefix_len`` to model shared system
    prompts / few-shot templates: each request is assigned to a group and its
    leading ``shared_prefix_len`` tokens hash-collide with its group-mates, so
    the prefix-cache hit rate emerges from arrival timing and memory pressure.
    """
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(n, rate, burstiness, rng)

    def sample_len(mean: int, std: int) -> np.ndarray:
        if std <= 0:
            return np.full(n, mean)
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma**2
        return np.clip(rng.lognormal(mu, sigma, n).round(), 1, None)

    plens = sample_len(*prompt_len).astype(int)
    olens = sample_len(*output_len).astype(int)
    groups = (
        rng.integers(0, num_prefix_groups, n) if num_prefix_groups > 0 else [None] * n
    )
    out = []
    for i in range(n):
        hashes = (
            _block_hashes(i, int(plens[i]), int(groups[i]), shared_prefix_len, block_size)
            if num_prefix_groups > 0
            else []
        )
        out.append(
            Request(
                req_id=i,
                arrival=float(arrivals[i]),
                prompt_len=int(plens[i]),
                output_len=int(olens[i]),
                prefix_hit_frac=prefix_hit_frac,
                block_hashes=hashes,
            )
        )
    return out
