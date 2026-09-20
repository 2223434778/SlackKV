# Phase 3G implementation

Phase 3G reused the Phase 3F-Fix in-engine FirstTokenGate and Phase 3B
connector-first native promotion without changing vLLM source. Real resumes
were sent at the original deadlines with `max_tokens=32`,
`temperature=0`, and `ignore_eos=true`; no client Event, lock, semaphore, or
fixed spacing was used.

The control stage produced six deterministic 32-token continuations. The
required A/B/C integration gate then showed the intended scheduler ordering:
A admitted, A first token, B admitted before A finished, B first token, C
admitted before B finished, and post-first-token decode overlap. However, the C
measured continuation diverged from its control sequence after 32 tokens even
though the count was 32. Per the Phase 3G protocol this is a correctness
failure, so the nine-run matrix was not started.

All trace times retain raw server `time.monotonic()` values and are aligned to
the client workload `t0_monotonic_s`; no first-event rebasing is used.
