# Phase 3U implementation

Phase 3U reuses the Phase 3S residency-lease vLLM patch unchanged. No vLLM source modification was made.

The driver runs a common 12-session setup and switches only controller policy: OnDemand, synchronous Blocking70, unleased NaiveAsync40, or ReservationLedger2 Async40 with two 256-block leases. Three X-only runs establish the slowdown denominator. Four non-statistical integration runs precede the 12 measured runs.

All H2D totals are derived from actual native-call counts plus measured external source-token counters. Duplicate traffic is `max(0, total - 1344 MiB)`. Promotion usefulness is derived from T6, scheduler arrival, and real remote-wait traces.

Phase 3U is an internal mechanism baseline matrix. It is not yet the final external SOTA comparison against MORI / Continuum / CacheWise / KVFlow-style systems.
