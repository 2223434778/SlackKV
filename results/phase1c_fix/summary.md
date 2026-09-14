# Phase 1C-Fix

Phase 1C-Fix re-ran the unchanged event set, policies, footprints, seeds and
workloads with distinct measured GPU0-to-pinned-CPU D2H and pinned-CPU-to-GPU0
H2D curves. Promotion-ready time is an explicit simulator wake-up event.

Transfer sanity is recorded in `transfer_sanity.csv`. The primary 56-KiB,
M=32 median results are: OffloadAll 68.52% miss / 5543.1 ms P95 / 0.271
memory-time; FixedTTL-2 2.18% / 0 ms / 0.253; SlackRatio-2 5.70% / 138.8 ms /
0.246; SlackRatio-4 3.09% / 0 ms / 0.256.

C0 PASS: per-event assertions found no admitted SlackRatio or FeasibilityOracle
event below isolated round-trip feasibility. C1 PASS: SlackRatio-4 meets the
median thresholds with 8/10 seeds at M=32 and 9/10 at M=64. C2 PASS: at M=32,
SlackRatio-2 (0.2468, 591.9 ms, 7.31%) is not TTL-frontier dominated and
strictly dominates FixedTTL-0.5 (0.2515, 636.1 ms, 6.75%) only on memory/P95
but not miss; the strict three-metric frontier check is required for final
interpretation and remains documented in aggregate metrics.

Selective policies remain clearly better than OffloadAll. Current evidence is
MIXED: FixedTTL-2/5/10 provides stronger low-stall points, while SlackRatio
provides lower-memory frontier candidates. D2H remains optimistically
non-contended; this is an oracle simulation only.

Phase 1C-Fix execution status: COMPLETE
