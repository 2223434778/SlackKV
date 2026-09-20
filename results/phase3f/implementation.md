# Phase 3F implementation

Pinned vLLM source commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`. The Phase 3B connector-first implementation was reused unchanged; therefore there is no new vLLM source patch. Native prefetch requests still execute through the existing connector-only path with zero model forwards and zero generated tokens.

The FirstTokenGate is a research-only admission gate in `experiments/phase3f/run_clean.py`: resumed A–F requests become eligible at their original oracle deadlines, and only one gated request is admitted until its first generated token is observed. Ordinary OnDemand/ResumeAware requests and native prefetch requests are not gated. With `max_tokens=1`, first-token release is approximately request completion; this phase does not claim multi-token decode overlap.

Memory residency is measured from native promotion completion to actual resume request admission/send, so intentional gate waiting is charged. Scheduler trace and step CSVs contain client-visible admission/first-token transitions; this runtime did not expose internal `WAITING_FOR_REMOTE_KVS` state without intrusive instrumentation.
