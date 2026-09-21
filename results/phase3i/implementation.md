No scheduler algorithm changes. Phase 3H generalized K-capacity gate is reused unchanged. The measured policies are K3 (unmarked), K1, and K2 using the existing request metadata.

X is an ordinary un-gated 128-token decode started before t0. The client sets t0 only after X produces its first token. A/B/C then use the existing ResumeAware promotion worker, p_safe=30 ms, resume_guard=110 ms, and the native /slackkv/prefetch endpoint. Scheduler logs are parsed using raw monotonic timestamps minus the saved t0; no first-event rebasing is used.

The background request carries trace_resume/session=X but no first-token gate marker, so it cannot acquire a K slot. Its first-token, finish, decode duration, TPOT, generated-token count, and slowdown relative to the 3-run X-only baseline are stored in background_metrics.csv. Native promotion rows are copied from endpoint responses and are not postprocessed constants.

The frozen validity rule is applied: if ordinary X forward causes promotion misses, blackout conflicts, or long native service, the K comparison is contaminated and Result D is reported. No dynamic-K or Phase 3J work is started.
