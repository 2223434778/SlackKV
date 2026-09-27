# Phase 4F-A implementation

PSA-1 is a research-only extension of the existing guarded native-prefetch endpoint. The controller selects the earliest-deadline episode whose first token has not been observed, then attaches its exact turn-qualified lease key and `ceil((Turn1 ISL + 1)/16)` progress footprint to a guarded promotion. The vLLM scheduler reads its own exact free-block count and actual protected lease block IDs before allocation or connector loading. It evaluates physical fit first, then incremental progress headroom. A rejected request uses the existing prefetch completion path with zero transfer, lease, model forward, and generated tokens.

The old physical-capacity guard remains the path when `progress_guard` is absent. Ordinary requests, connector behavior, lease handoff, first-token K2 admission, waiting queue ordering, and `allocate_slots()` failure behavior are unchanged. The source delta relative to the pre-Phase-4F SlackKV vLLM working tree is in `vllm_phase4f_a_delta.patch`.

The experiment runner uses fresh servers and the frozen eight boundary-0 jobs. P0 tests rejection and candidate-awareness; P1/P2 deterministically reconstruct the RCA lease set with and without X; P3 protects E04 itself; P4 uses the existing queue-aware JIT target with concurrent native HTTP promotions but no logical reservation; P5 adds the unchanged 3510-block logical reservation. Both rejected statuses use OnDemand fallback. Logical reservations are reclaimed immediately on rejection or after exact physical lease-release evidence at resume handoff.

This phase tests liveness and mechanism correctness only. It does not make a latency superiority claim.
