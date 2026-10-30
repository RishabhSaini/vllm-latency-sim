"""Discrete-event simulator of the vLLM v1 scheduler + step loop.

One iteration of the main loop == one ``EngineCore.step()``:
  1. schedule(): PASS 1 over RUNNING reqs, then PASS 2 admitting WAITING reqs,
     subject to the token budget, the active-request cap, and free KV blocks.
  2. cost model -> dt for the resulting batch shape (stands in for the forward).
  3. update: advance num_computed_tokens, sample tokens, free finished reqs.

This reproduces vLLM's continuous batching (prefill chunks and decodes share one
forward) and preemption-on-OOM so timing matches a real engine once the cost
model is calibrated. See AGENTS-style references in kv.py / cost_model.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from cost_model import CostModel
from kv import BlockManager, ModelConfig
from workload import Request


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256                 # active-request cap (max_num_active_reqs)
    max_num_batched_tokens: int = 8192      # per-step token budget
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0   # 0 => no single-chunk cap
    policy: str = "fcfs"                     # "fcfs" | "priority"


class Simulator:
    def __init__(
        self,
        cfg: ModelConfig,
        sched: SchedulerConfig,
        cost: CostModel,
        num_gpu_blocks: int,
        watermark: float = 0.01,
    ):
        self.cfg = cfg
        self.sched = sched
        self.cost = cost
        self.blocks = BlockManager(cfg, num_gpu_blocks, watermark)

    def run(self, requests: list[Request]) -> dict:
        reqs = sorted(requests, key=lambda r: r.arrival)
        n = len(reqs)
        next_idx = 0
        clock = 0.0
        waiting: deque[Request] = deque()
        running: list[Request] = []
        finished: list[Request] = []
        steps = 0

        single_chunk = self.sched.long_prefill_token_threshold or 10**12

        while finished.__len__() < n:
            # Admit arrivals up to the current clock.
            while next_idx < n and reqs[next_idx].arrival <= clock + 1e-12:
                waiting.append(reqs[next_idx])
                next_idx += 1

            if not running and not waiting:
                if next_idx < n:
                    clock = reqs[next_idx].arrival  # fast-forward idle gap
                    continue
                break  # all arrived and drained (shouldn't reach with finished<n)

            token_budget = self.sched.max_num_batched_tokens
            scheduled: list[tuple[Request, int, int]] = []  # (req, n, need_before)
            preempted = False

            # ---- PASS 1: RUNNING ----
            i = 0
            while i < len(running) and token_budget > 0:
                r = running[i]
                need = r.needed_tokens()
                nt = min(need, token_budget, single_chunk)
                if nt <= 0:
                    i += 1
                    continue
                if self.blocks.can_grow(r, r.computed + nt, is_admission=False):
                    scheduled.append((r, nt, need))
                    token_budget -= nt
                    i += 1
                else:
                    # Out of KV blocks: preempt (FCFS: last; PRIORITY: lowest).
                    victim = self._pick_victim(running)
                    self._preempt(victim, running, waiting)
                    preempted = True
                    break

            # ---- PASS 2: WAITING (only if no preemption this step) ----
            if not preempted:
                while (
                    waiting
                    and len(running) < self.sched.max_num_seqs
                    and token_budget > 0
                ):
                    r = waiting[0]
                    if r.computed == 0:  # fresh admission: reuse cached prefix
                        r.computed = self.blocks.match_prefix(r)
                    need = r.needed_tokens()
                    if self.sched.enable_chunked_prefill:
                        nt = min(need, token_budget, single_chunk)
                    else:
                        if need > token_budget:
                            break  # cannot fit whole prompt this step -> wait
                        nt = need
                    if not self.blocks.can_grow(r, r.computed + nt, is_admission=True):
                        self.blocks.release(r)  # undo prefix match, retry later
                        r.computed = 0
                        break
                    waiting.popleft()
                    running.append(r)
                    r.status = "running"
                    scheduled.append((r, nt, need))
                    token_budget -= nt

            if not scheduled:
                # Blocked (e.g. all budget consumed by preemption). Advance time.
                if next_idx < n:
                    clock = max(clock, reqs[next_idx].arrival)
                    continue
                break

            # ---- COST MODEL: one forward pass ----
            C = sum(nt for _, nt, _ in scheduled)
            B = len(scheduled)
            S = sum(self._end_context(r, nt) for r, nt, _ in scheduled)
            dt = self.cost.step_time(C, B, S)
            clock += dt
            steps += 1

            # ---- UPDATE: advance state, sample, retire ----
            done_now = []
            for r, nt, need in scheduled:
                r.computed += nt
                self.blocks.grow(r, r.computed)
                if nt == need:  # caught up -> sample one output token
                    if r.produced == 0:
                        r.ttft = clock - r.arrival
                    r.produced += 1
                    r.token_times.append(clock)
                    if r.produced >= r.output_len:
                        r.status = "finished"
                        r.finish_time = clock
                        self.blocks.release(r)
                        done_now.append(r)
            for r in done_now:
                running.remove(r)
                finished.append(r)

        return self._metrics(finished, clock, steps)

    # ---- helpers ----
    def _end_context(self, r: Request, nt: int) -> int:
        cached_prefix = r.prompt_len - r.target_prefill
        return cached_prefix + r.computed + nt

    def _pick_victim(self, running: list[Request]) -> Request:
        if self.sched.policy == "priority":
            # Lowest priority first; tie-break on latest arrival (youngest).
            return max(running, key=lambda r: (-getattr(r, "priority", 0), r.arrival))
        return running[-1]  # FCFS preempts the most-recently-added

    def _preempt(self, victim: Request, running: list[Request], waiting: deque) -> None:
        running.remove(victim)
        self.blocks.release(victim)
        victim.computed = 0
        victim.produced = 0
        victim.status = "waiting"
        victim.ttft = None
        victim.token_times.clear()
        waiting.appendleft(victim)  # recompute from scratch, keep ahead of fresh

    def _metrics(self, finished: list[Request], makespan: float, steps: int) -> dict:
        ttft = np.array([r.ttft for r in finished if r.ttft is not None])
        tpot = []
        for r in finished:
            if r.produced > 1:
                tpot.append((r.token_times[-1] - r.token_times[0]) / (r.produced - 1))
        tpot = np.array(tpot)
        e2e = np.array([r.finish_time - r.arrival for r in finished])
        out_tokens = sum(r.produced for r in finished)

        def pct(a):
            if len(a) == 0:
                return {}
            return {
                "mean": float(a.mean()),
                "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)),
            }

        return {
            "n_requests": len(finished),
            "steps": steps,
            "makespan_s": makespan,
            "output_tokens": out_tokens,
            "throughput_tok_s": out_tokens / makespan if makespan > 0 else 0.0,
            "prefix_hit_rate": self.blocks.hit_rate(),
            "ttft_s": pct(ttft),
            "tpot_s": pct(tpot),       # inter-token latency (decode)
            "e2e_s": pct(e2e),
        }
