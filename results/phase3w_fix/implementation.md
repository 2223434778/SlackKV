# Phase 3W-FIX implementation

The controller layer implements two fixes without changing vLLM scheduling or Phase 3S physical lease semantics. Each session has a lock-protected lifecycle. Immediately before native-prefetch commitment, the promotion worker atomically checks whether the real resume has started; stale work is classified `STALE_PROMOTION_CANCELLED` and falls back to OnDemand without a reservation, lease, or native transfer.

Logical reservations are reclaimed by a client-observed first streamed token signal, after post-hoc validation that physical `LEASE_RELEASED` occurred first. Predicted-deadline-plus-100-ms timeout expiry remains the fallback. The promotion loop drains runtime release events before each admission decision. All server logs use stage/policy/repetition-specific immutable paths; failed preliminary attempts are preserved under `failures/` and are excluded from measurements.

No new vLLM source changes were made. Phase 3S physical leases and the Phase 3W frozen workload, prediction errors, Async40 service budget, 200-ms margin, K2, and cache capacities were reused unchanged.
