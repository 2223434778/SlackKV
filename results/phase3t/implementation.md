# Phase 3T implementation

Phase 3T reuses the Phase 3S vLLM lease implementation without changes. The only new mechanism is a controller-side reservation ledger.

Each reservation represents 256 blocks over `[QueueJIT target, deadline + 100 ms]`. Before every QueueJIT decision the controller expires completed intervals, sums active blocks, and admits the next EDF job only when adding 256 blocks stays within the frozen 512-block budget. Admission issues `native_prefetch(prefix, lease_key=session)`; rejection performs no prefetch and is never retried.

The decision path is entirely in memory. It does not call Prometheus, `nvidia-smi`, or a capacity endpoint. The physical lease remains independent: the controller interval expires conservatively, while the vLLM lease is released only after the real resume has acquired its cached blocks.

Real A–L resume requests are submitted at their original deadlines and never wait for prefetch HTTP completion, futures, or controller events. FirstTokenGate is explicitly K=2. Skipped sessions follow normal OnDemand external recovery.

Phase 3T does not implement adaptive budgets, retries, repacking, or the full baseline matrix.
