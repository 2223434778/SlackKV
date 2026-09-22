# Phase 3S implementation

The research endpoint accepts an optional `lease_key`. When present it is carried as `slackkv_residency_lease_key` in `kv_transfer_params`; omission preserves the previous endpoint behavior.

The scheduler maintains `_slackkv_residency_leases: dict[str, list[KVCacheBlock]]`. At successful prefetch completion it obtains the request's block groups, removes null blocks, deduplicates by `block_id`, and calls `block_pool.touch()` before the prefetch request is freed. Duplicate keys and empty leases fail loudly.

```text
prefetch request
      |
      v
lease acquire via touch()
      |
      v
prefetch request free
      |
      v
lease retains blocks
      |
      v
cache pressure
      |
      v
real resume prefix hit + allocate_slots()
      |
      v
real request owns blocks
      |
      v
lease release via free_blocks()
      |
      v
real request remains owner
```

The release hook runs only after `allocate_slots()` has successfully touched/allocated the real resume's locally matched prefix. It requires every leased block to have `ref_cnt >= 2`, calls `block_pool.free_blocks()`, deletes the mapping exactly once, and then requires `ref_cnt >= 1`. This avoids a refcount-zero eviction window.

Structured `SLACKKV_LEASE_TRACE` events record acquire, held-after-prefetch-free, release-ready, and released boundaries, including real block counts, refcount ranges, active lease count, reason, and CPU operation time.

Phase 3S is **not** the final residency reservation scheduler. It validates only the runtime pin/lease primitive required to implement that scheduler safely. It deliberately adds no TTL or expiry mechanism because session A is guaranteed to resume.
