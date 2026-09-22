# Phase 3N — internal KV-ready handoff

Phase 3M T0–T7 instrumentation was reused unchanged; no new vLLM source changes were made.

- Valid measured runs: 15/15.
- PRE_READY / INTERNAL_READY_PRE_CLIENT_RETURN / POST_CLIENT_RETURN: 0 / 6 / 9.
- Critical-window local hit successes: 6/6; remote-wait events: 0.
- Median TTFT (PRE / critical / post): None / 59.82600443530828 / 56.423360016196966 ms. No PRE_READY sample occurred under the frozen offsets; no additional offset was added.
- Median internal T6-ready latency: 31.756389886140823 ms; median T6→client-return gap: 25.01799911260605 ms.
- Result A; next direction: ASYNC_PROMOTION_E2E.

The critical-window median is within 10% of the POST_CLIENT_RETURN median, so the scheduler-visible T6 boundary is sufficient for local reuse; waiting for the HTTP endpoint return is not required.
Interpretation: cache outcomes are based on the real `vllm:prompt_tokens_by_source_total` local/external source-counter deltas around A resume, plus actual scheduler remote-wait trace events; they are not inferred from latency.
