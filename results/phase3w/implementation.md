# Phase 3W implementation and stop record

Phase 3S physical KV lease patch reused unchanged; no vLLM source files were modified. The Phase 3V serving harness was reused for the fixed A–L workload, offload and pressure, real X-derived t0, K2 resumes, native response, source counters, traces, and server lifecycle. The Phase 3W callback changes only promotion scheduling deadlines and reservation expiry anchors.

The promotion callback consumes only `scheduler_deadlines_ms` and `reservation_end_ms`; actual deadlines remain with the unchanged resume dispatcher and post-run validation. Oracle uses actual deadline by definition. Raw uses predicted deadline. Safe uses predicted−200 ms. Raw/Safe reservation expiry is predicted+100 ms. Resume requests remain at actual deadlines.

The zero-error regression passed. Mixed integration stopped the measured stage: Oracle satisfied 6 promotions/6 skips; Safe had 5 promotions/7 skips with no timing or lease failures, and therefore missed the required 6/6 integration criterion; Raw triggered the existing connector-first scheduler runtime assertion while promotion overlapped resumed model work. Logs are preserved. No vLLM changes or policy retuning were made, and the 9-run matrix was not started.
