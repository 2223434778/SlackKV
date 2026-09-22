# Phase 3R implementation

Adds the frozen 1024 MiB / 112 MiB / 32 MiB reserve capacity rule to the Phase 3Q Async scheduler using vLLM kv_cache_usage_perc. The current committed Stage A evidence is marked invalid because per-target snapshots were not emitted into a persistent CSV; it must not be used as a result claim.
