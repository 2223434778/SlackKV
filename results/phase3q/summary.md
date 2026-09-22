# Phase 3Q — twelve-session async scalability

## Status

INCOMPLETE / integration gate failed. Per the frozen protocol, the 9-run policy matrix was not run.

## Observed integration-gate evidence

Async40-K2 completed all 12 native promotions before resume arrival (0 internal-ready misses). Every native prefetch returned `ok`, loaded 4096 external tokens, executed zero model forwards and zero generated tokens, and had T0-to-T6 below 40 ms (max 32.61 ms).

The capacity-retention requirement failed: aggregate source counters after the workload were 16,384 local-hit tokens and 32,768 external-transfer tokens, instead of 49,152 local and zero external. Sessions E-L entered `WAITING_FOR_REMOTE_KVS` despite their T6 values preceding scheduler arrival by 94.5–150.4 ms. This is READY_BUT_NOT_LOCAL / capacity-retention failure, not a fixed-40-ms service-time failure.

## Integration metrics

- Native promotions: 12/12 correct.
- Resume token count: 12/12 at 32 tokens.
- K2 maximum pre-first-token concurrency: 2.
- Background X active through L first token: PASS.
- Async source local/external tokens: 16,384 / 32,768.
- Async remote waits: 8 (E-L).
- Max issue-to-T6: 32.61 ms.

## Conclusion

Result C (capacity/retention scalability limit). With a 2 GiB external tier and unchanged 1 GiB GPU KV setup, the twelve-session promotion schedule reaches T6 in time but cannot retain all promoted prefixes until their later bursts. Next direction: `CAPACITY_AWARE_PROMOTION`.

No Phase 3R work was started.
