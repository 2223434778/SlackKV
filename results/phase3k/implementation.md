Phase 3K is the clean retry of Phase 3I. It reuses the Phase 3J load-aware promotion scheduler with frozen p_busy=70 ms and resume_guard=110 ms, plus the Phase 3H K-capacity gate. No vLLM source changes were made.

The runner executes ResumeAware-K3, FirstTokenGate-K1, and FirstTokenGate-K2 for five fresh server restarts each. X is started before t0, t0 is recorded after X first token, and A/B/C use the same 4096-token prefixes and 32-token continuations as earlier phases. Native endpoint response fields are copied directly into native_prefetch_metrics.csv.

Promotion validity is a hard gate before interpreting the admission ranking: every run must have zero deadline misses, blackout conflicts, and service-budget overruns, while preserving zero native model forwards/generated tokens/mixed steps. X must remain active through C first token. No dynamic-K or Phase 3L work is included.
