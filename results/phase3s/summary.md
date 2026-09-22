# Phase 3S — GPU KV residency lease

## Objective

Validate the minimum runtime primitive needed by a future residency-aware scheduler: retain one vLLM block-pool reference after native prefetch, survive ordinary prefix-cache pressure, and hand ownership to the real resume without an unprotected interval.

## Setup

- SlackKV base: `fe5e0603ad9dc6573607aad9cfdb0d551780eca5`
- vLLM source: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` plus the existing research patches and the Phase 3S delta
- RTX A6000, PyTorch 2.11.0+cu126, CUDA 12.6
- One 4096-token session A (117,440,512-byte KV), 32-token resume
- Four fixed 9500-token pressure prompts after native prefetch
- 3 fresh restarts for each of UnleasedPrefetch and LeasedPrefetch

## Results

| Mode | Full local reuse | External restore | Remote wait | Source conservation | 32 tokens |
|---|---:|---:|---:|---:|---:|
| UnleasedPrefetch | 0/3 | 3/3 (4096 tokens each) | 3/3 | 3/3 | 3/3 |
| LeasedPrefetch | 3/3 (4096 tokens each) | 0/3 | 0/3 | 3/3 | 3/3 |

All six native prefetches loaded 4096 external tokens / 117,440,512 bytes with zero model forwards, zero generated tokens, and zero mixed prefetch/model steps. The control pressure was valid in all three runs.

Each leased run acquired and released exactly one 256-block lease. Observed refcounts followed the intended handoff:

```text
acquire:                 min/max = 2/2
after prefetch free:     min/max = 1/1
before lease release:    min/max = 2/2
after lease release:     min/max = 1/1
```

Lease acquire CPU overhead was 17.701 us median / 17.941 us max. Lease release overhead was 13.081 us median / 15.790 us max. Both are far below 1 ms and the previously observed ~21-ms synchronous Prometheus scrape.

Unleased source medians were 0 local / 4096 external tokens. Leased source medians were 4096 local / 0 external tokens.

## Anomalies and limitations

The environment prints a non-fatal `nvidia_cutlass_dsl` `.pth` warning at Python startup. It did not affect tests or serving. Phase 3S intentionally has no TTL, multi-session reservation ledger, or capacity policy; A is guaranteed to resume. Prometheus was used only for before/after measurement snapshots, never as a lease or promotion control signal.

## Conclusion

**Result A.** A retained SlackKV reference is sufficient to protect prefetched GPU KV blocks against ordinary vLLM prefix-cache eviction until the real resume request takes ownership.

Next direction: `RESERVATION_LEDGER_12_SESSION` (not started).
