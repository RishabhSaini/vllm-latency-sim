"""KV-cache block accounting, mirroring vLLM v1 memory math.

The hard concurrency ceiling in vLLM comes from the paged KV cache. We reproduce
the same arithmetic vLLM uses so the simulator preempts at the same points a
real engine would.

References (current checkout):
  - page bytes:    AttentionSpec.page_size_bytes
                   = block_size * n_kv_heads * (head_size + head_size_v)
                     * dtype_bytes, summed over layers   (kv_cache_interface.py)
  - num_blocks:    available_memory // bytes_per_block    (kv_cache_planning.py)
  - max concurrency: num_blocks / blocks_per_request@max_model_len
                   (kv_cache_utils.get_max_concurrency_for_kv_cache_config)
  - prefix cache:  BlockPool hashes full blocks and reuses them across requests;
                   unreferenced cached blocks are evicted LRU from the free queue
                   (block_pool.py)
  - block_size default 16 tokens                          (config/cache.py)

``BlockManager`` here is *hash-aware*: full prompt blocks are keyed by a content
hash, so prefix sharing (and its hit rate) emerges from the workload rather than
being a fixed per-request fraction. A request references cached prefix blocks for
free; it promotes its own newly-filled prompt blocks into the cache when it
finishes prefill; on completion it decrefs (blocks stay cached and evictable).
"""

from __future__ import annotations

from dataclasses import dataclass


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class ModelConfig:
    """Minimal model/hardware description needed for KV + cost modelling."""

    num_layers: int
    num_kv_heads: int           # after TP sharding, i.e. per-GPU
    head_size: int
    head_size_v: int | None = None
    kv_dtype_bytes: int = 2     # fp16/bf16=2, fp8=1
    block_size: int = 16
    max_model_len: int = 8192

    def kv_bytes_per_token(self) -> int:
        hv = self.head_size_v if self.head_size_v is not None else self.head_size
        # (head_size + head_size_v) already sums the K and V states.
        return self.num_layers * self.num_kv_heads * (self.head_size + hv) * self.kv_dtype_bytes

    def page_size_bytes(self) -> int:
        return self.block_size * self.kv_bytes_per_token()


def num_blocks_from_memory(cfg: ModelConfig, available_kv_bytes: int) -> int:
    """Number of GPU KV blocks given bytes left after weights/activations."""
    return available_kv_bytes // cfg.page_size_bytes()


def max_concurrency(cfg: ModelConfig, num_blocks: int) -> float:
    """Requests that fit if each were at max_model_len (vLLM's ceiling)."""
    blocks_per_request = cdiv(cfg.max_model_len, cfg.block_size)
    return num_blocks / blocks_per_request


class _CachedBlock:
    __slots__ = ("refcount", "last_used")

    def __init__(self) -> None:
        self.refcount = 0
        self.last_used = 0


class BlockManager:
    """Hash-aware paged KV accounting with a watermark reserve for admissions.

    Invariant: free == num_blocks - len(cached) - sum(owned.values()), where
    ``cached`` are physical blocks holding hashed (shareable) content and
    ``owned`` are a request's private blocks (partial tail + output tokens).
    """

    def __init__(self, cfg: ModelConfig, num_blocks: int, watermark: float = 0.01):
        self.cfg = cfg
        self.bs = cfg.block_size
        self.num_blocks = int(num_blocks)
        self.free = int(num_blocks)
        self.watermark_blocks = int(watermark * num_blocks)
        self.cached: dict[int, _CachedBlock] = {}  # block hash -> entry
        self.owned: dict[int, int] = {}            # req_id -> private blocks
        self.refs: dict[int, list[int]] = {}       # req_id -> cached hashes held
        self.promoted: set[int] = set()            # req_ids past prefill
        self._tick = 0
        # stats
        self.hit_tokens = 0
        self.prompt_tokens = 0

    def blocks_for_tokens(self, tokens: int) -> int:
        return cdiv(max(tokens, 1), self.bs)

    def _touch(self) -> int:
        self._tick += 1
        return self._tick

    def _evictable(self) -> int:
        return sum(1 for b in self.cached.values() if b.refcount == 0)

    def _reclaim(self, need: int) -> bool:
        """Evict LRU unreferenced cached blocks until `need` free blocks exist."""
        if self.free >= need:
            return True
        victims = sorted(
            (h for h, b in self.cached.items() if b.refcount == 0),
            key=lambda h: self.cached[h].last_used,
        )
        for h in victims:
            if self.free >= need:
                break
            del self.cached[h]
            self.free += 1
        return self.free >= need

    # ---- admission ----
    def match_prefix(self, req) -> int:
        """Reference the longest cached block-prefix of `req`; return its tokens.

        Keeps at least one prompt token to recompute (vLLM never schedules zero
        new tokens), so a full-prompt hit still pays for the last block.
        """
        self.owned.setdefault(req.req_id, 0)
        refs = self.refs.setdefault(req.req_id, [])
        self.prompt_tokens += req.prompt_len
        if not req.block_hashes:
            return 0
        max_blocks = max(0, (req.prompt_len - 1) // self.bs)
        matched = 0
        for h in req.block_hashes:
            if matched >= max_blocks or h not in self.cached:
                break
            matched += 1
        for j in range(matched):
            b = self.cached[req.block_hashes[j]]
            b.refcount += 1
            b.last_used = self._touch()
            refs.append(req.block_hashes[j])
        self.hit_tokens += matched * self.bs
        return matched * self.bs

    def can_grow(self, req, computed_after: int, is_admission: bool = False) -> bool:
        target_owned = self.blocks_for_tokens(computed_after) - len(self.refs.get(req.req_id, []))
        need = target_owned - self.owned.get(req.req_id, 0)
        if need <= 0:
            return True
        reserve = self.watermark_blocks if is_admission else 0
        return need + reserve <= self.free + self._evictable()

    # ---- growth / promotion ----
    def grow(self, req, computed: int) -> None:
        rid = req.req_id
        if computed >= req.prompt_len and rid not in self.promoted:
            self._promote(req)
            self.promoted.add(rid)
        target_owned = self.blocks_for_tokens(computed) - len(self.refs.get(rid, []))
        cur = self.owned.get(rid, 0)
        if target_owned > cur:
            need = target_owned - cur
            self._reclaim(need)
            self.free -= need
            self.owned[rid] = target_owned
        elif target_owned < cur:
            self.free += cur - target_owned
            self.owned[rid] = target_owned

    def _promote(self, req) -> None:
        """Move this request's full prompt blocks (beyond the matched prefix)
        into the shared cache so later requests can hit them."""
        rid = req.req_id
        refs = self.refs[rid]
        full_prompt_blocks = min(req.prompt_len // self.bs, len(req.block_hashes))
        for j in range(len(refs), full_prompt_blocks):
            h = req.block_hashes[j]
            b = self.cached.get(h)
            if b is not None:  # another request already cached it -> dedup
                b.refcount += 1
                b.last_used = self._touch()
                self.owned[rid] -= 1
                self.free += 1
            else:              # convert our owned block into a cache entry
                nb = _CachedBlock()
                nb.refcount = 1
                nb.last_used = self._touch()
                self.cached[h] = nb
                self.owned[rid] -= 1
            refs.append(h)

    # ---- release ----
    def release(self, req) -> None:
        rid = req.req_id
        for h in self.refs.get(rid, []):
            b = self.cached.get(h)
            if b is not None:
                b.refcount -= 1
                b.last_used = self._touch()
        self.free += self.owned.get(rid, 0)
        self.owned.pop(rid, None)
        self.refs.pop(rid, None)
        self.promoted.discard(rid)

    def hit_rate(self) -> float:
        return self.hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
